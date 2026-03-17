#!/bin/bash
# Claude Gesture Confirm – one-shot setup
set -e

CLAUDE_DIR="$HOME/.claude"
SCRIPT="$CLAUDE_DIR/gesture_confirm.py"
MODEL="$CLAUDE_DIR/face_landmarker.task"
SETTINGS="$CLAUDE_DIR/settings.json"

echo "==> Installing Python dependencies…"
pip install mediapipe opencv-python -q

echo "==> Copying gesture_confirm.py to $CLAUDE_DIR…"
cp "$(dirname "$0")/gesture_confirm.py" "$SCRIPT"
chmod +x "$SCRIPT"

echo "==> Downloading MediaPipe face landmark model (3.6 MB)…"
curl -sL \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
  -o "$MODEL"

echo "==> Patching $SETTINGS with PreToolUse hook…"
python3 - <<'PYEOF'
import json, os, sys

settings_path = os.path.expanduser("~/.claude/settings.json")

with open(settings_path) as f:
    cfg = json.load(f)

hook_entry = {
    "hooks": [
        {
            "type": "command",
            "command": f"python3 {os.path.expanduser('~/.claude/gesture_confirm.py')}",
            "timeout": 35
        }
    ]
}

hooks = cfg.setdefault("hooks", {})

if "PreToolUse" not in hooks:
    hooks["PreToolUse"] = [hook_entry]
    print("  Added PreToolUse hook.")
else:
    # Check if our command is already there
    cmds = [h["command"] for entry in hooks["PreToolUse"] for h in entry.get("hooks", [])]
    if not any("gesture_confirm" in c for c in cmds):
        hooks["PreToolUse"].append(hook_entry)
        print("  Appended PreToolUse hook.")
    else:
        print("  PreToolUse hook already present, skipping.")

with open(settings_path, "w") as f:
    json.dump(cfg, f, indent=2)
PYEOF

echo ""
echo "Done! Grant camera access to your terminal app if prompted."
echo "Test with:"
echo "  echo '{\"tool_name\":\"Bash\",\"tool_input\":{\"command\":\"ls\"}}' | python3 ~/.claude/gesture_confirm.py"
