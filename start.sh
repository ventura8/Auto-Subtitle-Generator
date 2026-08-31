#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY=""

if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    VENV_PY="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
    VENV_PY="$SCRIPT_DIR/.venv/Scripts/python.exe"
fi

if [ -z "$VENV_PY" ]; then
    echo "=================================================================="
    echo "Auto-Subtitle-Generator: Virtual environment not found."
    echo "Starting automated environment and dependency installation..."
    echo "=================================================================="
    bash "$SCRIPT_DIR/install_dependencies.sh"
    if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
        VENV_PY="$SCRIPT_DIR/.venv/bin/python"
    elif [ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
        VENV_PY="$SCRIPT_DIR/.venv/Scripts/python.exe"
    else
        echo "ERROR: Virtual environment setup finished but Python binary not found." >&2
        exit 1
    fi
fi

exec "$VENV_PY" "$SCRIPT_DIR/auto_subtitle.py" "$@"

