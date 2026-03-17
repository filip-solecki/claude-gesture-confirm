#!/usr/bin/env python3
"""
Claude Code – Gesture Confirmation Hook
────────────────────────────────────────
Double right wink (×2 within 1 s) → ALLOW
Double left wink  (×2 within 1 s) → REJECT
Natural blink (both eyes)          → ignored
Timeout (30 s)                     → REJECT  (safe default)

Install deps:  pip install mediapipe opencv-python

Hook receives tool info as JSON on stdin and outputs a Claude Code
permissionDecision JSON on stdout. Runs as a PreToolUse hook.
"""

import os
import sys
import json
import time
import threading
import queue

# ── Read tool info from stdin ─────────────────────────────────────────────────
# Claude Code passes {"tool_name": "...", "tool_input": {...}} via stdin.
try:
    tool_info = json.loads(sys.stdin.read())
except Exception:
    tool_info = {}

tool_name  = tool_info.get("tool_name", "Unknown")
tool_input = tool_info.get("tool_input", {})


def _preview(name, inp, limit=200):
    """Format a human-readable preview of the tool input."""
    if name == "Bash" and "command" in inp:
        return "$ " + inp["command"].strip()
    if name == "Write" and "file_path" in inp:
        lines = inp.get("content", "").splitlines()
        preview = "\n".join(lines[:6])
        if len(lines) > 6:
            preview += f"\n… ({len(lines)} lines total)"
        return f"{inp['file_path']}\n{preview}"
    if name == "Edit" and "file_path" in inp:
        return f"{inp['file_path']}\n- {inp.get('old_string','')[:80].strip()}\n+ {inp.get('new_string','')[:80].strip()}"
    for key in ("file_path", "pattern", "url"):
        if key in inp:
            return inp[key]
    s = json.dumps(inp, indent=2)
    return (s[:limit] + "…") if len(s) > limit else s


tool_preview = _preview(tool_name, tool_input)

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import cv2
    import mediapipe as mp
except ImportError:
    sys.stderr.write("[gesture_confirm] run: pip install mediapipe opencv-python\n")
    # Fail open: if deps are missing don't block Claude entirely.
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "permissionDecisionReason": "gesture_confirm: deps missing – auto-allowed",
    }}))
    sys.exit(0)

import tkinter as tk

# ── Tunable constants ─────────────────────────────────────────────────────────
TIMEOUT            = 30    # seconds before auto-reject if no gesture
BLINK_THRESHOLD    = 0.35  # MediaPipe eyeBlink blendshape score → eye "closed"
WINK_OPEN_MAX      = 0.30  # the OTHER eye must stay below this to confirm a wink
                            # (prevents a normal blink from counting as a wink)
DOUBLE_WINK_WINDOW = 1.0   # max seconds between the two winks of a double-wink
MODEL_PATH         = os.path.expanduser("~/.claude/face_landmarker.task")

# ── UI colours (dark theme) ───────────────────────────────────────────────────
BG     = "#111827"   # background
FG     = "#e5e7eb"   # primary text
ACCENT = "#6366f1"   # indigo highlight
GREEN  = "#22c55e"   # allow / open eye
RED    = "#ef4444"   # reject / closed eye
MUTED  = "#6b7280"   # secondary text
DIM    = "#374151"   # inactive dots

# ── Shared state between detection thread and UI thread ──────────────────────
result_q: "queue.Queue[str]" = queue.Queue()

# Updated by the detection thread, read by the UI poll loop (80 ms interval).
# CPython's GIL makes these plain-dict updates safe without an explicit lock.
eye_state: dict = {
    "face":    False,   # whether a face is currently detected
    "left":    None,    # True = left eye closed, False = open, None = no face
    "right":   None,
    "l_count": 0,       # completed winks in the current window (0 or 1)
    "r_count": 0,
}


def _send_decision(decision: str, reason: str) -> None:
    """Print the Claude Code hook decision JSON to stdout."""
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
        "permissionDecisionReason": reason,
    }}))


# ── Gesture detection (runs in a background thread) ──────────────────────────
def _detect() -> None:
    """
    Capture webcam frames, run MediaPipe Face Landmarker, and detect
    double-wink gestures via eyeBlink blendshape scores.

    A "wink" is one close→open cycle of a single eye while the other eye
    stays clearly open (score < WINK_OPEN_MAX).  Two winks of the same eye
    within DOUBLE_WINK_WINDOW seconds fires the decision.
    """
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        result_q.put("allow")   # no camera available — fail open
        return
    time.sleep(0.3)             # brief warmup so first frames aren't corrupt

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Per-eye tracking variables
    r_was_closed   = l_was_closed   = False   # eye state in previous frame
    r_wink_count   = l_wink_count   = 0       # completed winks in current window
    r_window_start = l_window_start = None    # timestamp of first wink in window
    t0 = time.time()

    try:
        with mp_vision.FaceLandmarker.create_from_options(options) as lmk:
            while (time.time() - t0) < TIMEOUT:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms  = int((time.time() - t0) * 1000)
                result = lmk.detect_for_video(mp_img, ts_ms)

                if not result.face_blendshapes:
                    # No face — reset all state and update overlay
                    eye_state.update(face=False, left=None, right=None,
                                     r_count=0, l_count=0)
                    r_was_closed = l_was_closed = False
                    r_wink_count = l_wink_count = 0
                    r_window_start = l_window_start = None
                    continue

                # Extract per-eye blink scores (0.0 = open, 1.0 = fully closed)
                bs      = {b.category_name: b.score for b in result.face_blendshapes[0]}
                l_score = bs.get("eyeBlinkLeft",  0.0)   # subject's left eye
                r_score = bs.get("eyeBlinkRight", 0.0)   # subject's right eye
                l_cls   = l_score > BLINK_THRESHOLD
                r_cls   = r_score > BLINK_THRESHOLD
                now     = time.time()

                eye_state.update(face=True, left=l_cls, right=r_cls)

                # Both eyes closed = natural blink → ignore, reset counters
                if l_cls and r_cls:
                    r_was_closed = l_was_closed = False
                    r_wink_count = l_wink_count = 0
                    r_window_start = l_window_start = None
                    eye_state["r_count"] = eye_state["l_count"] = 0
                    continue

                # ── Right eye double-wink → ALLOW ────────────────────────
                # A wink is detected on the rising edge: eye was closed last
                # frame and is now open (close→open transition).
                r_winking = r_cls and l_score < WINK_OPEN_MAX
                if not r_winking and r_was_closed:
                    # Wink just completed — count it
                    if r_window_start is None or (now - r_window_start) > DOUBLE_WINK_WINDOW:
                        r_wink_count, r_window_start = 1, now   # start fresh window
                    else:
                        r_wink_count += 1
                    eye_state["r_count"] = r_wink_count
                    if r_wink_count >= 2:
                        result_q.put("allow"); return
                elif not r_winking and r_window_start and (now - r_window_start) > DOUBLE_WINK_WINDOW:
                    # Window expired without a second wink — reset
                    r_wink_count = 0; r_window_start = None
                    eye_state["r_count"] = 0
                r_was_closed = r_winking

                # ── Left eye double-wink → REJECT ────────────────────────
                l_winking = l_cls and r_score < WINK_OPEN_MAX
                if not l_winking and l_was_closed:
                    if l_window_start is None or (now - l_window_start) > DOUBLE_WINK_WINDOW:
                        l_wink_count, l_window_start = 1, now
                    else:
                        l_wink_count += 1
                    eye_state["l_count"] = l_wink_count
                    if l_wink_count >= 2:
                        result_q.put("block"); return
                elif not l_winking and l_window_start and (now - l_window_start) > DOUBLE_WINK_WINDOW:
                    l_wink_count = 0; l_window_start = None
                    eye_state["l_count"] = 0
                l_was_closed = l_winking

    finally:
        cap.release()

    result_q.put("block")   # timeout reached → safe default is reject


# ── Overlay UI (runs on the main thread via tkinter) ─────────────────────────
class Overlay:
    """
    Frameless always-on-top window shown in the top-right corner.
    Displays the tool name, a live eye-state indicator, wink-count dots,
    and a countdown timer.  Polls result_q every 80 ms.
    """

    SYMBOL_OPEN   = "●"   # eye open
    SYMBOL_CLOSED = "—"   # eye closed
    DOT_FILLED    = "●"   # wink registered
    DOT_EMPTY     = "○"   # wink not yet registered

    def __init__(self):
        self._start = time.time()
        r = tk.Tk()
        self.root = r
        r.title("")
        r.configure(bg=BG)
        r.attributes("-topmost", True)
        r.attributes("-alpha", 0.93)
        r.overrideredirect(True)   # remove OS window chrome for a cleaner look

        W = 420
        sw = r.winfo_screenwidth()
        # Height is determined by content; position top-right
        r.geometry(f"+{sw - W - 20}+20")

        self._build()
        threading.Thread(target=_detect, daemon=True).start()
        self._poll()
        r.mainloop()

    def _lbl(self, parent, text, fg=FG, size=11, bold=False, **kw):
        return tk.Label(parent, text=text, bg=BG, fg=fg,
                        font=("Helvetica", size, "bold" if bold else "normal"), **kw)

    def _build(self):
        r, pad = self.root, dict(padx=18)

        # Coloured accent bar at the top
        tk.Frame(r, bg=ACCENT, height=4).pack(fill="x")

        # Tool info header
        self._lbl(r, "⚡  Claude needs permission",
                  fg=FG, size=12, bold=True).pack(anchor="w", pady=(10, 2), **pad)
        self._lbl(r, f"Tool: {tool_name}",
                  fg=ACCENT, size=10, bold=True).pack(anchor="w", **pad)
        tk.Label(r, text=tool_preview, bg=BG, fg=MUTED,
                 font=("Helvetica Neue", 9), justify="left",
                 wraplength=380, anchor="w").pack(anchor="w", pady=(1, 8), **pad)

        # Eye indicators grid
        ef = tk.Frame(r, bg=BG)
        ef.pack()
        self._lbl(ef, "YOUR LEFT EYE",  fg=MUTED, size=8).grid(row=0, column=0, padx=36)
        self._lbl(ef, "YOUR RIGHT EYE", fg=MUTED, size=8).grid(row=0, column=2, padx=36)

        self.l_lbl = tk.Label(ef, text=self.SYMBOL_OPEN, bg=BG, fg=GREEN,
                               font=("Helvetica", 30, "bold"))
        self.l_lbl.grid(row=1, column=0, padx=36, pady=2)
        tk.Label(ef, bg=BG, width=2).grid(row=1, column=1)   # spacer
        self.r_lbl = tk.Label(ef, text=self.SYMBOL_OPEN, bg=BG, fg=GREEN,
                               font=("Helvetica", 30, "bold"))
        self.r_lbl.grid(row=1, column=2, padx=36, pady=2)

        # Wink count dots: ○ ○ → ● ○ after first wink → fires on second
        self.l_dots = tk.Label(ef, text=f"{self.DOT_EMPTY} {self.DOT_EMPTY}",
                                bg=BG, fg=DIM, font=("Helvetica", 14))
        self.l_dots.grid(row=2, column=0, padx=36, pady=(0, 4))
        self.r_dots = tk.Label(ef, text=f"{self.DOT_EMPTY} {self.DOT_EMPTY}",
                                bg=BG, fg=DIM, font=("Helvetica", 14))
        self.r_dots.grid(row=2, column=2, padx=36, pady=(0, 4))

        self._lbl(r, "Double right wink → ALLOW   |   Double left wink → REJECT",
                  fg=MUTED, size=9).pack(pady=(4, 2))

        self.status = self._lbl(r, f"Looking for face…  {TIMEOUT}s", fg=MUTED, size=9)
        self.status.pack(pady=(2, 10))

    def _dots_str(self, count):
        """Return dot string showing how many winks have been registered (max shown: 1)."""
        return f"{self.DOT_FILLED if count >= 1 else self.DOT_EMPTY} {self.DOT_EMPTY}"

    def _poll(self):
        """Called every 80 ms on the main thread to refresh the UI and check for a result."""
        remaining = max(0.0, TIMEOUT - (time.time() - self._start))
        s = eye_state

        if s["face"]:
            def style(closed):
                return (self.SYMBOL_CLOSED, RED) if closed else (self.SYMBOL_OPEN, GREEN)
            lt, lc = style(s["left"])
            rt, rc = style(s["right"])
            self.l_lbl.config(text=lt, fg=lc)
            self.r_lbl.config(text=rt, fg=rc)
            self.l_dots.config(text=self._dots_str(s["l_count"]),
                               fg=RED   if s["l_count"] > 0 else DIM)
            self.r_dots.config(text=self._dots_str(s["r_count"]),
                               fg=GREEN if s["r_count"] > 0 else DIM)
            self.status.config(text=f"Waiting for gesture…  {remaining:.0f}s", fg=MUTED)
        else:
            self.l_lbl.config(text="?", fg=MUTED)
            self.r_lbl.config(text="?", fg=MUTED)
            self.status.config(text=f"No face detected…  {remaining:.0f}s",
                               fg=RED if remaining < 10 else MUTED)

        try:
            result = result_q.get_nowait()
            self._finish(result)
            return
        except queue.Empty:
            pass

        self.root.after(80, self._poll)

    def _finish(self, result: str):
        """Show brief confirmation then exit."""
        if result == "allow":
            self.status.config(text="✅  ALLOWED", fg=GREEN)
            self.r_dots.config(text=f"{self.DOT_FILLED} {self.DOT_FILLED}", fg=GREEN)
        else:
            self.status.config(text="❌  REJECTED", fg=RED)
            self.l_dots.config(text=f"{self.DOT_FILLED} {self.DOT_FILLED}", fg=RED)
        self.root.after(900, lambda: self._exit(result))

    def _exit(self, result: str):
        self.root.destroy()
        if result == "allow":
            _send_decision("allow", "Double right wink – approved by user")
        else:
            _send_decision("deny", "Double left wink or timeout – rejected")
        sys.exit(0)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Overlay()
