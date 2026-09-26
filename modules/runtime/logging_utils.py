"""Logging and console initialization utilities."""

import contextlib
import ctypes
import os
import platform
import signal
import subprocess
import sys
import time
from ctypes import wintypes
from typing import Any

from ..configuration import config
from ..configuration.version import __version__
from ..media.hardware_utils import get_cpu_name

# Track active subprocesses for cleanup
active_subprocesses: list[subprocess.Popen[Any]] = []

# Win32 console output modes enabled so ANSI escapes render.
_ENABLE_PROCESSED_OUTPUT = 0x1
_ENABLE_WRAP_AT_EOL_OUTPUT = 0x2
_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x4
_ANSI_CONSOLE_MODE = _ENABLE_PROCESSED_OUTPUT | _ENABLE_WRAP_AT_EOL_OUTPUT | _ENABLE_VIRTUAL_TERMINAL_PROCESSING
_STD_OUTPUT_HANDLE = -11

# Win32 console control events that terminate the process.
_CTRL_C_EVENT = 0
_CTRL_BREAK_EVENT = 1
_CTRL_CLOSE_EVENT = 2
_CTRL_LOGOFF_EVENT = 5
_CTRL_SHUTDOWN_EVENT = 6
_TERMINATING_CTRL_EVENTS = frozenset({_CTRL_C_EVENT, _CTRL_BREAK_EVENT, _CTRL_CLOSE_EVENT, _CTRL_LOGOFF_EVENT, _CTRL_SHUTDOWN_EVENT})
_WIN32_CTRL_HANDLER = None


def print_banner(optimizer=None):
    """Prints a stylish ASCII banner for the application."""
    os_info = f"{platform.system()} {platform.release()}"

    # Defaults if optimizer not ready
    cpu_name = get_cpu_name()
    gpu_name = "Unknown"
    vram = "N/A"
    profile = "STANDARD"

    # Config defaults
    precision = "32-bit Float (WAV)"
    mode = "Hybrid"
    batch = "N/A"
    threads = "N/A"

    if optimizer:
        gpu_name = optimizer.gpu_name
        vram = f"{optimizer.vram_gb} GB VRAM"
        profile = optimizer.profile

        batch = optimizer.config.get("nllb_batch", "N/A")
        threads = optimizer.config.get("ffmpeg_threads", "N/A")

    banner = r"""
     _         _          ____        _      _   _ _
    / \  _   _| |_ ___   / ___| _   _| |__  | |_(_) |_ ___
   / _ \| | | | __/ _ \  \___ \| | | | '_ \ | __| | __/ _ \
  / ___ \ |_| | || (_) |  ___) | |_| | |_) || |_| | ||  __/
 /_/   \_\__,_|\__\___/  |____/ \__,_|_.__/  \__|_|\__\___|
"""
    print("=" * 60)
    print(f"   AUTO-SUBTITLE-GENERATOR - v{__version__}")
    print(f"   Running on: {os_info}")
    print("=" * 60 + "\n")

    print("\033[96m" + banner + "\033[0m")

    print("[HARDWARE DETECTED]")
    print(f"   CPU : {os.cpu_count()} Logical Cores ({cpu_name})")
    print(f"   GPU : {gpu_name} ({vram})")
    print("")

    print(f"[AUTO-TUNED SETTINGS -> Profile: {profile} ({gpu_name})]")
    print(f"   Audio Precision : {precision}")
    print(f"   Process Mode    : {mode}")
    print(f"   Batch Size      : {batch}")
    print(f"   Threads         : {threads}")
    print("   Mix Levels      : Vocals=1.0, Background=1.0")
    sep_model = getattr(config, "AUDIO_SEPARATOR_MODEL_ID", "UVR_Model")
    print(f"   Models          : {sep_model} / UVR-DeNoise")
    print("   Config Source   : config.yaml")

    print("\n" + "-" * 60)
    print(" [HOW TO USE]")
    print(" 1. Drag and Drop a video file (or folder) here.")
    print(" 2. Or paste the file path below.")


def register_subprocess(proc):
    """Registers a subprocess to be killed on shutdown."""
    active_subprocesses.append(proc)


def unregister_subprocess(proc):
    """Unregisters a subprocess (e.g., after clean completion)."""
    if proc in active_subprocesses:
        active_subprocesses.remove(proc)


def _cleanup_subprocess(proc):
    """Terminate subprocess and escalate to kill on timeout."""
    print(f"  [Cleanup] Killing subprocess PID: {proc.pid}")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        try:
            proc.kill()
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass


def _cleanup_active_subprocesses():
    """Best-effort cleanup for all active subprocesses."""
    for proc in active_subprocesses.copy():
        if proc.poll() is None:
            try:
                _cleanup_subprocess(proc)
            except OSError as e:
                print(f"  [Cleanup] Error killing process: {e}")


def handle_shutdown(_signum, _frame):
    """Handles termination signals for graceful shutdown."""
    print("\n\n[!] Termination detected. Stopping all processes...")

    _cleanup_active_subprocesses()

    sys.exit(1)


def _reconfigure_streams_utf8():
    """Switch stdout/stderr to UTF-8 with replacement.

    Transcripts are printed live; when stdout is redirected to a file on
    Windows its default encoding is the ANSI code page, which cannot encode
    characters such as "ă" and raised UnicodeEncodeError mid-transcription.
    Streams without ``reconfigure`` (capture objects) are left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def init_console():
    """Initializes the console for ANSI support and UTF-8 output."""
    _reconfigure_streams_utf8()
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(_STD_OUTPUT_HANDLE)
            mode = ctypes.c_uint32(0)
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | _ANSI_CONSOLE_MODE)
        except (AttributeError, OSError):
            pass


def setup_signal_handlers():
    """Registers signal handlers for SIGINT, SIGTERM, and SIGHUP."""
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, handle_shutdown)

    # Windows Console Handler for "X" button
    if sys.platform == "win32":
        try:
            # Define handler type
            handler_routine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

            def ctrl_handler(ctrl_type):
                if ctrl_type in _TERMINATING_CTRL_EVENTS:
                    print("\n\n[!] Termination detected. Stopping all processes...")
                    _cleanup_active_subprocesses()
                    os._exit(1)
                return False

            # Keep reference alive to prevent GC
            globals()["_WIN32_CTRL_HANDLER"] = handler_routine(ctrl_handler)

            kernel32 = ctypes.windll.kernel32
            if not kernel32.SetConsoleCtrlHandler(globals()["_WIN32_CTRL_HANDLER"], True):
                print("[Warning] Failed to set Windows Console Handler")

        except (AttributeError, OSError) as e:
            print(f"[Warning] Error setting up Windows handler: {e}")


def log(message, level="INFO", to_console=True):
    """Logs a message to both console and log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    entry = f"[{timestamp}] [{level}] {message}"

    # If debug logging is ON, everything goes to console.
    # If OFF, DEBUG messages are skipped.
    should_print = to_console and (level != "DEBUG" or config.DEBUG_LOGGING)

    if should_print:
        prefix = {"ERROR": "!!! ", "WARNING": "! ", "CRITICAL": "XXX "}.get(level, "")
        # Use \r\033[K to clear any active progress bar on the current line
        print(f"\r\033[K{prefix}{message}")

    try:
        with open(config.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except OSError:
        print(f"\r\033[K{entry}", file=sys.stderr)
    sys.stdout.flush()
