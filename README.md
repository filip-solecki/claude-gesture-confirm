# Claude Gesture Confirm

Approve or reject [Claude Code](https://claude.ai/code) tool requests with eye gestures — no keyboard needed.

When Claude wants to run a tool (bash command, file edit, web fetch, etc.), an overlay appears at the top-center of your screen. Look at the camera and gesture:

| Gesture | Action |
|---|---|
| **Double right wink** (×2 within 1 s) | ✅ Allow |
| **Double left wink** (×2 within 1 s) | ❌ Reject |
| **Both eyes closed for 1 s** | ✅ Always allow (saves to allowlist) |
| No gesture for 30 s | ❌ Reject (safe default) |

Natural blinks are ignored. Previously always-allowed commands skip the overlay entirely.

## Requirements

- macOS (uses AVFoundation for camera access)
- Python 3.9+
- Camera permission granted to your terminal app

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

On first run macOS will prompt for camera access for your terminal app (Terminal, iTerm2, Warp, etc.). Grant it once in **System Settings → Privacy & Security → Camera**.

## Test

```bash
echo '{"tool_name":"Bash","tool_input":{"command":"ls -la"}}' | python3 ~/.claude/gesture_confirm.py
```

The overlay should appear. Double right wink to allow, double left wink to reject.

## How it works

Claude Code supports [hooks](https://docs.claude.ai/en/docs/claude-code/hooks) — shell commands that run before every tool execution. The `PreToolUse` hook in `~/.claude/settings.json` calls `gesture_confirm.py`.

The script:
1. Reads tool info from stdin (tool name + a one-line preview of what it will do)
2. Checks the allowlist — if the command was previously always-allowed, exits immediately
3. Opens the camera with OpenCV + AVFoundation
4. Runs MediaPipe Face Landmarker (VIDEO mode) tracking `eyeBlinkLeft` / `eyeBlinkRight` blendshape scores
5. Uses hysteresis to debounce scores and detect double-wink or both-eyes-hold gestures
6. Outputs `{"hookSpecificOutput": {"permissionDecision": "allow"|"deny"}}` and exits

The overlay is a frameless `tkinter` window (always on top) showing live eye state and a countdown timer.

## Configuration

Edit the constants near the top of `gesture_confirm.py`:

| Constant | Default | Description |
|---|---|---|
| `TIMEOUT` | `30` | Seconds before auto-reject if no gesture |
| `BLINK_CLOSE` | `0.40` | Blendshape score that marks an eye as closed |
| `BLINK_OPEN` | `0.20` | Score that marks an eye as open again (hysteresis) |
| `WINK_OPEN_MAX` | `0.30` | Max score the other eye may have during a wink |
| `DOUBLE_WINK_WINDOW` | `1.0` | Max seconds between the two winks of a double-wink |
| `HOLD_ALWAYS_SECS` | `1.0` | Seconds to hold both eyes closed to trigger always-allow |

## Allowlist

Approved-always entries are stored in `~/.claude/gesture_confirm_allowlist.json`.

- **Bash** commands are matched by exact command string
- **Other tools** (Write, Edit, etc.) are matched by tool name alone

To clear the allowlist: `rm ~/.claude/gesture_confirm_allowlist.json`

## Uninstall

1. Remove the `PreToolUse` block from `~/.claude/settings.json`
2. Delete `~/.claude/gesture_confirm.py` and `~/.claude/face_landmarker.task`
