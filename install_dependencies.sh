#!/usr/bin/env bash
# Sets up the environment for auto_subtitle.py on Linux and macOS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up Auto-Subtitle Generator Environment (Linux / macOS) ==="

# Step 1: Detect Python 3.12
echo -e "\nStep 1: Checking Python 3.12..."
PYTHON_BIN=""

if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
elif command -v python3 >/dev/null 2>&1; then
    PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [ "$PY_VER" = "3.12" ]; then
        PYTHON_BIN="python3"
    fi
fi

if [ -z "$PYTHON_BIN" ] && [ -f "$HOME/.local/bin/python3.12" ]; then
    PYTHON_BIN="$HOME/.local/bin/python3.12"
elif [ -z "$PYTHON_BIN" ] && command -v uv >/dev/null 2>&1; then
    UV_PY="$(uv python find 3.12 2>/dev/null || true)"
    if [ -n "$UV_PY" ] && [ -x "$UV_PY" ]; then
        PYTHON_BIN="$UV_PY"
    fi
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3.12 is required (>=3.12,<3.13) but was not found." >&2
    echo "Please install Python 3.12 via your package manager (e.g. apt install python3.12 python3.12-venv, or brew install python@3.12)." >&2
    exit 1
fi

echo "Found Python: $($PYTHON_BIN --version) ($PYTHON_BIN)"

# Step 2: Create Virtual Environment
echo -e "\nStep 2: Setting up Python Virtual Environment..."
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

if [ -x "$VENV_PY" ]; then
    VENV_PY_VER="$("$VENV_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [ "$VENV_PY_VER" != "3.12" ]; then
        echo "ERROR: Existing .venv uses Python $VENV_PY_VER. Remove .venv and rerun this script." >&2
        exit 1
    fi
fi

if [ ! -f "$VENV_PY" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# Step 3: Verify FFmpeg
echo -e "\nStep 3: Checking FFmpeg..."
if command -v ffmpeg >/dev/null 2>&1; then
    echo "Found system FFmpeg: $(which ffmpeg)"
elif [ -f "$VENV_DIR/bin/ffmpeg" ]; then
    echo "Found local venv FFmpeg: $VENV_DIR/bin/ffmpeg"
else
    echo "ERROR: FFmpeg is required but was not found." >&2
    echo "Please install FFmpeg via your package manager (e.g. apt install ffmpeg or brew install ffmpeg)." >&2
    exit 1
fi

# Step 4: Install Dependencies via Poetry
echo -e "\nStep 4: Installing Dependencies via Poetry..."
"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install poetry

"$VENV_PY" -m poetry config --local virtualenvs.in-project true
"$VENV_PY" -m poetry config --local virtualenvs.create false

if [ ! -f "poetry.lock" ]; then
    echo "Generating poetry.lock..."
    "$VENV_PY" -m poetry lock --no-interaction
fi

echo "Installing runtime and ML dependencies..."
export POETRY_REQUESTS_TIMEOUT=300
export PIP_DEFAULT_TIMEOUT=300
REQUESTED_GROUPS="${1:-ml}"
if [[ -z "$REQUESTED_GROUPS" || "$REQUESTED_GROUPS" == *,*,,* || "$REQUESTED_GROUPS" == ,* || "$REQUESTED_GROUPS" == *, ]]; then
    echo "ERROR: Dependency groups must be a comma-separated list of ml and/or dev." >&2
    exit 1
fi
IFS=',' read -r -a DEPENDENCY_GROUPS <<< "$REQUESTED_GROUPS"
for group in "${DEPENDENCY_GROUPS[@]}"; do
    if [[ "$group" != "ml" && "$group" != "dev" ]]; then
        echo "ERROR: Unsupported dependency group '$group'. Choose ml and/or dev." >&2
        exit 1
    fi
done
"$VENV_PY" -m poetry install --no-root --with "$REQUESTED_GROUPS" --no-interaction

# Step 5: Validate Faster-Whisper Runtime
echo -e "\nStep 5: Validating AI Runtime..."
"$VENV_PY" -c '
import os
import sys

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA Available: True (Device: {torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple Silicon MPS Available: True")
    else:
        print("Running in CPU mode.")
except Exception as e:
    print(f"Torch load check info: {e}")

try:
    from faster_whisper import WhisperModel
    print("Faster-Whisper import successful.")
except Exception as e:
    sys.stderr.write(f"Faster-Whisper check error: {e}\n")
    sys.exit(1)
'

# Step 6: Create Launcher
echo -e "\nStep 6: Updating Launcher..."
cat << 'EOF' > "$SCRIPT_DIR/start.sh"
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    VENV_PY="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
    VENV_PY="$SCRIPT_DIR/.venv/Scripts/python.exe"
else
    echo "ERROR: Virtual environment not found at $SCRIPT_DIR/.venv" >&2
    echo "Please run ./install_dependencies.sh first." >&2
    exit 1
fi
exec "$VENV_PY" "$SCRIPT_DIR/auto_subtitle.py" "$@"
EOF
chmod +x "$SCRIPT_DIR/start.sh"

echo -e "\n=== Installation Complete! ==="
echo "Run './start.sh' to use the tool."
