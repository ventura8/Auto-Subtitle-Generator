#!/usr/bin/env bash
# Runner for the cross-platform container test suite.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TARGET="${1:-ubuntu}"

# Guest command channel. Ports 5900 (VNC) and 3389 (RDP) are display protocols
# and cannot run commands; both VM services forward a real SSH port instead.
# The guest image must have an SSH daemon enabled and this key authorized.
SSH_USER="${SSH_USER:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
SSH_READY_TIMEOUT="${SSH_READY_TIMEOUT:-900}"
SSH_KNOWN_HOSTS_FILE="$(mktemp)"
WINDOWS_OEM_DIR=""
MACOS_SSH_IMAGE="${MACOS_SSH_IMAGE:-auto-subtitle-macos-ssh-test}"

cleanup_macos_test() {
    docker compose -f docker/docker-compose.test.yml down
    if [[ -n "$SSH_KNOWN_HOSTS_FILE" ]]; then
        rm -f -- "$SSH_KNOWN_HOSTS_FILE"
    fi
}

cleanup_windows_test() {
    docker compose -f docker/docker-compose.test.yml down
    if [[ -n "$WINDOWS_OEM_DIR" ]]; then
        rm -rf -- "$WINDOWS_OEM_DIR"
    fi
    if [[ -n "$SSH_KNOWN_HOSTS_FILE" ]]; then
        rm -f -- "$SSH_KNOWN_HOSTS_FILE"
    fi
}

prepare_windows_ssh() {
    local public_key="${SSH_KEY}.pub"
    SSH_USER="${SSH_USER:-Docker}"
    if [[ ! -r "$SSH_KEY" ]]; then
        echo "ERROR: SSH private key not found or unreadable at $SSH_KEY." >&2
        return 1
    fi
    if [[ ! -r "$public_key" ]]; then
        echo "ERROR: SSH public key not found at $public_key." >&2
        return 1
    fi
    WINDOWS_OEM_DIR="$(mktemp -d)"
    cp docker/windows-oem/install.bat "$WINDOWS_OEM_DIR/install.bat"
    cp docker/windows-oem/configure-ssh.ps1 "$WINDOWS_OEM_DIR/configure-ssh.ps1"
    cp "$public_key" "$WINDOWS_OEM_DIR/authorized_keys"
}

prepare_macos_ssh() {
    local public_key="${SSH_KEY}.pub"
    SSH_USER="${SSH_USER:-user}"
    if [[ ! -r "$SSH_KEY" ]]; then
        echo "ERROR: SSH private key not found or unreadable at $SSH_KEY." >&2
        return 1
    fi
    if [[ ! -r "$public_key" ]]; then
        echo "ERROR: SSH public key not found at $public_key." >&2
        return 1
    fi
    if ! docker image inspect "$MACOS_SSH_IMAGE" >/dev/null 2>&1; then
        echo "ERROR: macOS SSH image $MACOS_SSH_IMAGE is unavailable." >&2
        echo "Build or load a dockurr/macos guest image with Remote Login enabled," >&2
        echo "user $SSH_USER created, and $public_key in its authorized_keys." >&2
        return 1
    fi
}

ssh_guest() {
    local port="$1"
    shift
    if [[ ! -r "$SSH_KEY" ]]; then
        echo "ERROR: SSH private key not found or unreadable at $SSH_KEY." >&2
        return 1
    fi
    ssh -p "$port" \
        -i "$SSH_KEY" \
        -o BatchMode=yes \
        -o StrictHostKeyChecking=accept-new \
        -o UserKnownHostsFile="$SSH_KNOWN_HOSTS_FILE" \
        -o ConnectTimeout=10 \
        "$SSH_USER@127.0.0.1" "$@"
}

wait_for_ssh() {
    local port="$1"
    local deadline=$((SECONDS + SSH_READY_TIMEOUT))
    echo "Waiting for guest SSH on port $port (timeout ${SSH_READY_TIMEOUT}s)..."
    while ((SECONDS < deadline)); do
        if ssh_guest "$port" true >/dev/null 2>&1; then
            echo "Guest SSH is ready."
            return 0
        fi
        sleep 10
    done
    echo "ERROR: Guest SSH on port $port did not become ready in ${SSH_READY_TIMEOUT}s." >&2
    return 1
}

echo "=== Cross-Platform Docker Test Runner ==="
echo "Target: $TARGET"

case "$TARGET" in
    ubuntu)
        echo "Building and running clean Ubuntu container test..."
        docker build -f docker/Dockerfile.ubuntu -t auto-subtitle-ubuntu-test .
        docker run --rm auto-subtitle-ubuntu-test
        echo "Ubuntu container test completed successfully!"
        ;;
    macos)
        prepare_macos_ssh
        echo "Starting macOS test service via Compose and running test suite..."
        trap cleanup_macos_test EXIT
        MACOS_SSH_IMAGE="$MACOS_SSH_IMAGE" SSH_USER="$SSH_USER" docker compose -f docker/docker-compose.test.yml up -d macos_test
        wait_for_ssh 2222
        ssh_guest 2222 "which ffmpeg >/dev/null 2>&1 || brew install ffmpeg; cd /shared && ./install_dependencies.sh && ./.venv/bin/python -m poetry run pytest"
        echo "macOS container test completed successfully!"
        ;;
    windows)
        prepare_windows_ssh
        echo "Starting Windows test service via Compose and running test suite..."
        trap cleanup_windows_test EXIT
        WINDOWS_OEM_DIR="$WINDOWS_OEM_DIR" SSH_USER="$SSH_USER" docker compose -f docker/docker-compose.test.yml up -d windows_test
        wait_for_ssh 2223
        # dockurr/windows exposes the /shared bind mount to the guest as a Samba
        # network share (SAMBA=Y by default), not as a local C: path.
        ssh_guest 2223 "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \\\\host.lan\\Data\\run_local_pipeline.ps1"
        echo "Windows container test completed successfully!"
        ;;
    *)
        echo "Unknown target: $TARGET. Usage: $0 [ubuntu|macos|windows]"
        exit 1
        ;;
esac
