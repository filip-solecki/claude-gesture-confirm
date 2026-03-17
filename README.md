# Claude Gesture Confirm

Approve or reject [Claude Code](https://claude.ai/code) tool requests with a blink or wink — no keyboard needed.

When Claude wants to run a tool (bash command, file edit, web fetch, etc.), a small overlay appears in the top-right corner of your screen. Look at your camera and gesture:

| Gesture | Action |
|---|---|
| **Blink** (both eyes) | ✅ Allow |
| **Left wink** (left eye only) | ❌ Reject |
| No gesture for 30 s | ❌ Reject (safe default) |

![overlay screenshot placeholder](https://placehold.co/390x230/111827/6366f1?text=Claude+needs+permission)

## Requirements

- macOS (uses AVFoundation for camera)
- Python 3.9+
- Camera access granted to your terminal app

## Install

```bash
git clone https://github.com/filip-solecki/claude-gesture-confirm
cd claude-gesture-confirm
bash setup.sh
```

`setup.sh` will:
1. Install Python deps (`mediapipe`, `opencv-python`)
2. Copy `gesture_confirm.py` to `~/.claude/`
3. Download the MediaPipe face landmark model (~3.6 MB) to `~/.claude/`
4. Patch `~/.claude/settings.json` with the `PreToolUse` hook

### Camera permission

On first run, macOS will prompt for camera access for your terminal app (Terminal, iTerm2, Warp, etc.). Grant it once in **System Settings → Privacy & Security → Camera**.

## Test

```bash
echo '{"tool_name":"Bash","tool_input":{"command":"ls -la"}}' | python3 ~/.claude/gesture_confirm.py
```

The overlay should appear. Blink to allow, left wink to reject.

## How it works

Claude Code supports [hooks](https://docs.claude.ai/en/docs/claude-code/hooks) — shell commands that run before tool execution. The `PreToolUse` hook in `~/.claude/settings.json` calls `gesture_confirm.py` for every tool call.

The script:
1. Reads tool info from stdin (name + input preview)
2. Opens the camera with OpenCV + AVFoundation
3. Runs MediaPipe Face Landmarker (VIDEO mode) to track `eyeBlinkLeft` / `eyeBlinkRight` blendshape scores in real time
4. Detects blink (both > 0.35) or left wink (left > 0.35 and right < 0.10)
5. Outputs `{"hookSpecificOutput": {"permissionDecision": "allow"|"deny", ...}}` and exits

The overlay is a frameless `tkinter` window that shows live eye state (● open / — closed) and a countdown timer.

## Configuration

Edit the constants at the top of `gesture_confirm.py`:

| Constant | Default | Description |
|---|---|---|
| `TIMEOUT` | `30` | Seconds before auto-reject |
| `BLINK_THRESHOLD` | `0.35` | Blendshape score to count as closed |
| `WINK_OPEN_MAX` | `0.10` | Max score for "open" eye in wink detection |
| `FRAMES_NEEDED` | `2` | Consecutive frames to confirm gesture |

## Uninstall

Remove the `PreToolUse` entry from `~/.claude/settings.json` and delete `~/.claude/gesture_confirm.py` and `~/.claude/face_landmarker.task`.
