#!/usr/bin/env python3
"""
Claude Code – Gesture Confirmation Hook
────────────────────────────────────────
Right wink (right eye only) → ALLOW
Left wink  (left eye only)  → REJECT
Blink (both eyes)           → ignored  (natural blink, no trigger)
Timeout (30 s)              → REJECT   (safe default)

Install deps:  pip install mediapipe opencv-python
"""

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

tool_name    = tool_info.get("tool_name", "Unknown")
tool_input   = tool_info.get("tool_input", {})

def _preview(name, inp, limit=54):
    if name == "Bash" and "command" in inp:
        cmd = inp["command"].replace("\n", " ")
        return ("$ " + cmd)[:limit + 2]
    for key in ("file_path", "pattern", "url"):
        if key in inp:
            v = inp[key]
            return v[-limit:] if len(v) > limit else v
    s = json.dumps(inp, separators=(",", ":"))
    return (s[:limit] + "…") if len(s) > limit else s

tool_preview = _preview(tool_name, tool_input)

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import cv2                          # noqa: E402
    import mediapipe as mp             # noqa: E402
    import numpy as np                 # noqa: E402
except ImportError:
    sys.stderr.write(
        "[gesture_confirm] Missing deps – run: pip install mediapipe opencv-python\n"
        "Falling back to ALLOW.\n"
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": "gesture_confirm: deps missing, auto-allowed",
        }
    }))
    sys.exit(0)

import tkinter as tk  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
TIMEOUT         = 30     # seconds until auto-reject
BLINK_THRESHOLD = 0.35   # blendshape score above this → eye closed
WINK_OPEN_MAX   = 0.30   # other eye must be BELOW this score to confirm wink
FRAMES_NEEDED   = 2      # consecutive closed frames to confirm gesture
MODEL_PATH      = "/Users/filipsolecki/.claude/face_landmarker.task"

# ── Colours ───────────────────────────────────────────────────────────────────
BG     = "#111827"
FG     = "#e5e7eb"
ACCENT = "#6366f1"
GREEN  = "#22c55e"
RED    = "#ef4444"
MUTED  = "#6b7280"

# ── Shared state ──────────────────────────────────────────────────────────────
result_q: "queue.Queue[str]" = queue.Queue()
eye_state: dict = {"left": None, "right": None, "face": False}

def _output(decision: str, reason: str):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }))

# ── Gesture detection (background thread) ─────────────────────────────────────
def _detect():
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

    r_wink_f = l_wink_f = 0
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
                    eye_state.update(left=None, right=None, face=False)
                    r_wink_f = l_wink_f = 0
                    continue

                bs      = {b.category_name: b.score for b in result.face_blendshapes[0]}
                l_score = bs.get("eyeBlinkLeft",  0.0)
                r_score = bs.get("eyeBlinkRight", 0.0)
                l_cls   = l_score > BLINK_THRESHOLD
                r_cls   = r_score > BLINK_THRESHOLD
                eye_state.update(left=l_cls, right=r_cls, face=True)

                both_closed = l_cls and r_cls

                if both_closed:
                    # Natural blink — ignore, reset counters
                    r_wink_f = l_wink_f = 0
                elif r_cls and l_score < WINK_OPEN_MAX:
                    # Right wink (right eye closed, left clearly open) → ALLOW
                    r_wink_f += 1; l_wink_f = 0
                elif l_cls and r_score < WINK_OPEN_MAX:
                    # Left wink (left eye closed, right clearly open) → REJECT
                    l_wink_f += 1; r_wink_f = 0
                else:
                    if r_wink_f >= FRAMES_NEEDED:
                        result_q.put("allow"); return
                    if l_wink_f >= FRAMES_NEEDED:
                        result_q.put("block"); return
                    r_wink_f = l_wink_f = 0
    finally:
        cap.release()

    result_q.put("block")   # timeout → reject


# ── Overlay UI (main thread) ──────────────────────────────────────────────────
class Overlay:

    EYE_OPEN   = "●"
    EYE_CLOSED = "—"

    def __init__(self):
        self._start = time.time()
        r = tk.Tk()
        self.root = r
        r.title("")
        r.configure(bg=BG)
        r.attributes("-topmost", True)
        r.attributes("-alpha", 0.93)
        r.overrideredirect(True)

        W, H = 390, 230
        sw = r.winfo_screenwidth()
        r.geometry(f"{W}x{H}+{sw - W - 20}+20")

        self._build()
        threading.Thread(target=_detect, daemon=True).start()
        self._poll()
        r.mainloop()

    # ── layout ────────────────────────────────────────────────────────────────
    def _lbl(self, parent, text, fg=FG, size=11, bold=False, **kw):
        return tk.Label(
            parent, text=text, bg=BG, fg=fg,
            font=("Helvetica", size, "bold" if bold else "normal"), **kw
        )

    def _build(self):
        r = self.root
        pad = dict(padx=18)

        # accent bar
        tk.Frame(r, bg=ACCENT, height=4).pack(fill="x")

        # header
        self._lbl(r, "⚡  Claude needs permission",
                  fg=FG, size=12, bold=True).pack(anchor="w", pady=(10, 2), **pad)
        self._lbl(r, f"Tool: {tool_name}",
                  fg=ACCENT, size=10, bold=True).pack(anchor="w", **pad)
        self._lbl(r, tool_preview,
                  fg=MUTED, size=9).pack(anchor="w", pady=(1, 10), **pad)

        # eye indicators
        ef = tk.Frame(r, bg=BG)
        ef.pack()
        self._lbl(ef, "YOUR LEFT EYE",  fg=MUTED, size=8).grid(row=0, column=0, padx=36)
        self._lbl(ef, "YOUR RIGHT EYE", fg=MUTED, size=8).grid(row=0, column=2, padx=36)

        self.l_lbl = tk.Label(ef, text=self.EYE_OPEN, bg=BG, fg=GREEN,
                               font=("Helvetica", 30, "bold"))
        self.l_lbl.grid(row=1, column=0, padx=36, pady=4)
        tk.Label(ef, bg=BG, width=2).grid(row=1, column=1)
        self.r_lbl = tk.Label(ef, text=self.EYE_OPEN, bg=BG, fg=GREEN,
                               font=("Helvetica", 30, "bold"))
        self.r_lbl.grid(row=1, column=2, padx=36, pady=4)

        # instructions
        self._lbl(r, "Right wink → ALLOW   |   Left wink → REJECT",
                  fg=MUTED, size=9).pack(pady=(4, 2))

        # status
        self.status = self._lbl(r, f"Looking for face…  {TIMEOUT}s", fg=MUTED, size=9)
        self.status.pack(pady=(2, 10))

    # ── poll loop (80 ms) ─────────────────────────────────────────────────────
    def _poll(self):
        remaining = max(0.0, TIMEOUT - (time.time() - self._start))
        s = eye_state

        if s["face"]:
            def style(closed):
                return (self.EYE_CLOSED, RED) if closed else (self.EYE_OPEN, GREEN)
            lt, lc = style(s["left"])
            rt, rc = style(s["right"])
            self.l_lbl.config(text=lt, fg=lc)
            self.r_lbl.config(text=rt, fg=rc)
            self.status.config(text=f"Waiting for gesture…  {remaining:.0f}s", fg=MUTED)
        else:
            self.l_lbl.config(text="?", fg=MUTED)
            self.r_lbl.config(text="?", fg=MUTED)
            self.status.config(
                text=f"No face detected…  {remaining:.0f}s",
                fg=RED if remaining < 10 else MUTED
            )

        try:
            result = result_q.get_nowait()
            self._done(result)
            return
        except queue.Empty:
            pass

        self.root.after(80, self._poll)

    # ── finish ────────────────────────────────────────────────────────────────
    def _done(self, result: str):
        if result == "allow":
            self.status.config(text="✅  ALLOWED", fg=GREEN)
            self.l_lbl.config(text=self.EYE_OPEN, fg=GREEN)
            self.r_lbl.config(text=self.EYE_OPEN, fg=GREEN)
        else:
            self.status.config(text="❌  REJECTED", fg=RED)
            self.l_lbl.config(text=self.EYE_CLOSED, fg=RED)
            self.r_lbl.config(text=self.EYE_CLOSED, fg=RED)
        self.root.after(900, lambda: self._exit(result))

    def _exit(self, result: str):
        self.root.destroy()
        if result == "allow":
            _output("allow", "Blink detected – approved by user")
        else:
            _output("deny", "Rejected by user gesture (left wink or timeout)")
        sys.exit(0)


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Overlay()
