#!/usr/bin/env python3
"""
Claude Code – Gesture Confirmation Hook
────────────────────────────────────────
Double right wink (×2 within 1 s)   → ALLOW
Double left wink  (×2 within 1 s)   → REJECT
Both eyes closed for 1 s            → ALLOW + add to allowlist
Natural blink (both eyes, < 1 s)    → ignored
Click "Always allow"                → ALLOW + add to allowlist
Timeout (30 s)                      → REJECT  (safe default)

Allowlist stored in: ~/.claude/gesture_confirm_allowlist.json
  - Bash entries matched by exact command string
  - Other tools matched by tool name

Install deps:  pip install mediapipe opencv-python
"""

import os
import sys
import json
import time
import threading
import queue

# ── Read tool info from stdin ─────────────────────────────────────────────────
try:
    tool_info = json.loads(sys.stdin.read())
except Exception:
    tool_info = {}

tool_name  = tool_info.get("tool_name", "Unknown")
tool_input = tool_info.get("tool_input", {})

# ── Allowlist ─────────────────────────────────────────────────────────────────
ALLOWLIST_PATH = os.path.expanduser("~/.claude/gesture_confirm_allowlist.json")


def _load_allowlist() -> list:
    try:
        with open(ALLOWLIST_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _save_to_allowlist() -> None:
    """Add the current tool/command to the allowlist and persist it."""
    entries = _load_allowlist()
    entry = {"tool": tool_name}
    if tool_name == "Bash" and "command" in tool_input:
        entry["command"] = tool_input["command"].strip()
    if entry not in entries:
        entries.append(entry)
    with open(ALLOWLIST_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def _is_allowlisted() -> bool:
    """Return True if this tool call should be auto-allowed without showing the overlay."""
    for entry in _load_allowlist():
        if entry.get("tool") != tool_name:
            continue
        # Bash: match by exact command
        if tool_name == "Bash":
            if entry.get("command") == tool_input.get("command", "").strip():
                return True
        else:
            # Other tools: match by tool name alone
            return True
    return False


# ── Short preview line for the overlay ───────────────────────────────────────
def _preview(name, inp) -> str:
    """One concise line describing what the tool will do."""
    if name == "Bash":
        cmd = inp.get("command", "").replace("\n", " ").strip()
        return f"$ {cmd[:70]}{'…' if len(cmd) > 70 else ''}"
    if name in ("Write", "Edit"):
        return inp.get("file_path", "")
    if name == "WebFetch":
        url = inp.get("url", "")
        return url[:80] + ("…" if len(url) > 80 else "")
    for key in ("file_path", "pattern", "url"):
        if key in inp:
            return inp[key]
    return ""


tool_preview = _preview(tool_name, tool_input)

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import cv2
    import mediapipe as mp
except ImportError:
    sys.stderr.write("[gesture_confirm] run: pip install mediapipe opencv-python\n")
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "permissionDecisionReason": "gesture_confirm: deps missing – auto-allowed",
    }}))
    sys.exit(0)

import tkinter as tk

# ── Tunable constants ─────────────────────────────────────────────────────────
TIMEOUT            = 30    # seconds before auto-reject if no gesture
BLINK_CLOSE        = 0.40  # score rises above this → eye transitions to "closed"
BLINK_OPEN         = 0.20  # score falls below this → eye transitions back to "open"
WINK_OPEN_MAX      = 0.30  # the OTHER eye must stay below this to confirm a wink
DOUBLE_WINK_WINDOW = 1.0   # max seconds between the two winks of a double-wink
HOLD_ALWAYS_SECS   = 1.0   # seconds to hold both eyes closed → always allow
MODEL_PATH         = os.path.expanduser("~/.claude/face_landmarker.task")

# ── UI colours (dark theme) ───────────────────────────────────────────────────
BG     = "#111827"
FG     = "#e5e7eb"
ACCENT = "#6366f1"
GREEN  = "#22c55e"
RED    = "#ef4444"
MUTED  = "#6b7280"
DIM    = "#374151"

# ── Shared state between detection thread and UI thread ──────────────────────
result_q: "queue.Queue[str]" = queue.Queue()

eye_state: dict = {
    "face":       False,
    "left":       None,
    "right":      None,
    "l_count":    0,
    "r_count":    0,
    "both_hold":  0.0,   # 0.0–1.0 progress toward always-allow (both eyes closed)
}


def _send_decision(decision: str, reason: str) -> None:
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": decision,
        "permissionDecisionReason": reason,
    }}))


# ── Gesture detection (background thread) ────────────────────────────────────
def _detect() -> None:
    """
    Capture webcam frames and detect gestures via MediaPipe eyeBlink blendshapes.
    Puts "allow", "always_allow", or "block" into result_q.
    """
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        result_q.put("allow")
        return
    time.sleep(0.3)

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Hysteresis state: eye is "closed" once score > BLINK_CLOSE,
    # and stays closed until score < BLINK_OPEN. Prevents flickering.
    r_closed = l_closed = False
    r_wink_count   = l_wink_count   = 0
    r_window_start = l_window_start = None
    both_hold_start = None   # when both eyes started being held closed
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
                    eye_state.update(face=False, left=None, right=None,
                                     r_count=0, l_count=0, both_hold=0.0)
                    r_closed = l_closed = False
                    r_wink_count = l_wink_count = 0
                    r_window_start = l_window_start = None
                    both_hold_start = None
                    continue

                bs      = {b.category_name: b.score for b in result.face_blendshapes[0]}
                l_score = bs.get("eyeBlinkLeft",  0.0)
                r_score = bs.get("eyeBlinkRight", 0.0)
                now     = time.time()

                # Hysteresis: transition closed→open only when score drops below BLINK_OPEN
                if r_closed:
                    r_cls = r_score > BLINK_OPEN
                else:
                    r_cls = r_score > BLINK_CLOSE
                if l_closed:
                    l_cls = l_score > BLINK_OPEN
                else:
                    l_cls = l_score > BLINK_CLOSE
                # Capture previous state before updating (needed for transition detection)
                r_prev, l_prev = r_closed, l_closed
                r_closed, l_closed = r_cls, l_cls

                eye_state.update(face=True, left=l_cls, right=r_cls)

                # ── Both eyes closed: short = natural blink, hold 1s = ALWAYS ALLOW ──
                if l_cls and r_cls:
                    if both_hold_start is None:
                        both_hold_start = now
                    held = now - both_hold_start
                    eye_state["both_hold"] = min(held / HOLD_ALWAYS_SECS, 1.0)
                    if held >= HOLD_ALWAYS_SECS:
                        result_q.put("always_allow"); return
                    continue   # don't count single-eye gestures while both are closed
                else:
                    both_hold_start = None
                    eye_state["both_hold"] = 0.0

                # ── Right eye double-wink → ALLOW ────────────────────────
                # Transition: was closed last frame, open now, other eye was open → wink
                if r_prev and not r_cls and l_score < WINK_OPEN_MAX:
                    if r_window_start is None or (now - r_window_start) > DOUBLE_WINK_WINDOW:
                        r_wink_count, r_window_start = 1, now
                    else:
                        r_wink_count += 1
                    eye_state["r_count"] = r_wink_count
                    if r_wink_count >= 2:
                        result_q.put("allow"); return
                elif not r_cls and r_window_start and (now - r_window_start) > DOUBLE_WINK_WINDOW:
                    r_wink_count = 0; r_window_start = None
                    eye_state["r_count"] = 0

                # ── Left eye double-wink → REJECT ────────────────────────
                if l_prev and not l_cls and r_score < WINK_OPEN_MAX:
                    if l_window_start is None or (now - l_window_start) > DOUBLE_WINK_WINDOW:
                        l_wink_count, l_window_start = 1, now
                    else:
                        l_wink_count += 1
                    eye_state["l_count"] = l_wink_count
                    if l_wink_count >= 2:
                        result_q.put("block"); return
                elif not l_cls and l_window_start and (now - l_window_start) > DOUBLE_WINK_WINDOW:
                    l_wink_count = 0; l_window_start = None
                    eye_state["l_count"] = 0

    finally:
        cap.release()

    result_q.put("block")   # timeout → reject


# ── Overlay UI ────────────────────────────────────────────────────────────────
class Overlay:
    SYMBOL_OPEN   = "●"
    SYMBOL_CLOSED = "—"
    DOT_FILLED    = "●"
    DOT_EMPTY     = "○"

    def __init__(self):
        self._start = time.time()
        r = tk.Tk()
        self.root = r
        r.title("")
        r.configure(bg=BG)
        r.attributes("-topmost", True)
        r.attributes("-alpha", 0.93)
        r.overrideredirect(True)

        W = 400
        sw = r.winfo_screenwidth()
        r.geometry(f"+{(sw - W) // 2}+20")

        self._build()
        threading.Thread(target=_detect, daemon=True).start()
        self._poll()
        r.mainloop()

    def _lbl(self, parent, text, fg=FG, size=12, bold=False, **kw):
        return tk.Label(parent, text=text, bg=BG, fg=fg,
                        font=("Helvetica", size, "bold" if bold else "normal"), **kw)

    def _build(self):
        r, pad = self.root, dict(padx=18)

        # Accent bar
        tk.Frame(r, bg=ACCENT, height=4).pack(fill="x")

        # Tool name + preview
        self._lbl(r, "⚡  Claude needs permission",
                  fg=FG, size=14, bold=True).pack(anchor="w", pady=(12, 2), **pad)
        self._lbl(r, tool_name, fg=ACCENT, size=12, bold=True).pack(anchor="w", **pad)
        if tool_preview:
            self._lbl(r, tool_preview, fg=MUTED, size=11).pack(
                anchor="w", pady=(2, 10), **pad)

        # Eye indicators
        ef = tk.Frame(r, bg=BG)
        ef.pack(pady=(0, 4))
        self._lbl(ef, "LEFT EYE",  fg=MUTED, size=9).grid(row=0, column=0, padx=40)
        self._lbl(ef, "RIGHT EYE", fg=MUTED, size=9).grid(row=0, column=2, padx=40)

        self.l_lbl = tk.Label(ef, text=self.SYMBOL_OPEN, bg=BG, fg=GREEN,
                               font=("Helvetica", 32, "bold"))
        self.l_lbl.grid(row=1, column=0, padx=40, pady=2)
        tk.Label(ef, bg=BG, width=2).grid(row=1, column=1)
        self.r_lbl = tk.Label(ef, text=self.SYMBOL_OPEN, bg=BG, fg=GREEN,
                               font=("Helvetica", 32, "bold"))
        self.r_lbl.grid(row=1, column=2, padx=40, pady=2)

        # Wink count dots
        self.l_dots = tk.Label(ef, text=f"{self.DOT_EMPTY} {self.DOT_EMPTY}",
                                bg=BG, fg=DIM, font=("Helvetica", 16))
        self.l_dots.grid(row=2, column=0, padx=40, pady=(0, 2))
        self.r_dots = tk.Label(ef, text=f"{self.DOT_EMPTY} {self.DOT_EMPTY}",
                                bg=BG, fg=DIM, font=("Helvetica", 16))
        self.r_dots.grid(row=2, column=2, padx=40, pady=(0, 2))

        # Hold progress bar centered (fills as you hold both eyes closed)
        BAR_W = 160
        bar_frame = tk.Frame(r, bg=BG)
        bar_frame.pack(pady=(0, 4))
        self._lbl(bar_frame, "Both eyes closed:", fg=MUTED, size=9).pack(side="left", padx=(0, 6))
        bar_bg = tk.Frame(bar_frame, bg=DIM, width=BAR_W, height=5)
        bar_bg.pack(side="left")
        bar_bg.pack_propagate(False)
        self.both_hold_bar = tk.Frame(bar_bg, bg=ACCENT, height=5, width=0)
        self.both_hold_bar.place(x=0, y=0)

        # Instructions (two lines for clarity)
        self._lbl(r, "Right ×2 → Allow   |   Left ×2 → Reject",
                  fg=MUTED, size=10).pack()
        self._lbl(r, "Close both eyes 1s → Always allow",
                  fg=ACCENT, size=10).pack()

        # Always allow button
        tk.Button(
            r, text="Always allow this",
            bg=DIM, fg=MUTED, relief="flat",
            font=("Helvetica", 10),
            activebackground=ACCENT, activeforeground=FG,
            cursor="hand2",
            command=self._always_allow,
        ).pack(pady=(6, 4))

        # Status / timer
        self.status = self._lbl(r, f"Looking for face…  {TIMEOUT}s", fg=MUTED, size=10)
        self.status.pack(pady=(0, 12))

    def _dots_str(self, count):
        d1 = self.DOT_FILLED if count >= 1 else self.DOT_EMPTY
        d2 = self.DOT_FILLED if count >= 2 else self.DOT_EMPTY
        return f"{d1} {d2}"

    def _always_allow(self):
        """Save to allowlist and allow immediately."""
        _save_to_allowlist()
        result_q.put("always_allow")

    def _poll(self):
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
            self.both_hold_bar.config(width=int(s["both_hold"] * 160))
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
        if result == "always_allow":
            self.status.config(text="✅  ALWAYS ALLOWED", fg=ACCENT)
            self.both_hold_bar.config(width=160, bg=ACCENT)
        elif result == "allow":
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
        elif result == "always_allow":
            _send_decision("allow", "Added to allowlist – always allowed")
        else:
            _send_decision("deny", "Double left wink or timeout – rejected")
        sys.exit(0)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check allowlist before showing the overlay at all
    if _is_allowlisted():
        _send_decision("allow", "Allowlisted – auto-approved")
        sys.exit(0)

    Overlay()
