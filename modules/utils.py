"""
Utility module for Auto Subtitle Generator.
Handles logging, timestamps, progress bars, and FFmpeg operations.
"""
import sys
import time
import math
import os
import signal
import subprocess
from . import config

# =============================================================================
# LOGGING & SIGNALS
# =============================================================================


def print_banner(optimizer=None):
    """Prints a stylish ASCII banner for the application."""
    import platform

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
    print("   Initialization Complete.[████████████████████] 100.0%")
    print("=" * 60)
    print("   AI HYBRID VHS AUDIO RESTORER - v1.0.0")
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
    # Using config.AUDIO_SEPARATOR_MODEL_ID would be ideal if available, else hardcode as example or generic
    sep_model = getattr(config, "AUDIO_SEPARATOR_MODEL_ID", "UVR_Model")
    print(f"   Models          : {sep_model} / UVR-DeNoise")
    print("   Config Source   : config.yaml")

    print("\n" + "-" * 60)
    print(" [HOW TO USE]")
    print(" 1. Drag and Drop a video file (or folder) here.")
    print(" 2. Or paste the file path below.")


# Track active subprocesses for cleanup
active_subprocesses = []


def register_subprocess(proc):
    """Registers a subprocess to be killed on shutdown."""
    active_subprocesses.append(proc)


def unregister_subprocess(proc):
    """Unregisters a subprocess (e.g., after clean completion)."""
    if proc in active_subprocesses:
        active_subprocesses.remove(proc)


def handle_shutdown(signum, frame):
    """Handles termination signals for graceful shutdown."""
    print("\n\n[!] Termination detected. Stopping all processes...")

    # Kill all registered subprocesses
    for proc in active_subprocesses:
        if proc.poll() is None:  # If running
            try:
                print(f"  [Cleanup] Killing subprocess PID: {proc.pid}")
                proc.terminate()
                # Windows might need force kill if SIGTERM is ignored
                if sys.platform == "win32":
                    subprocess.call(
                        ['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
            except Exception as e:
                print(f"  [Cleanup] Error killing process: {e}")

    sys.exit(0)


def init_console():
    """Initializes the console for ANSI support, especially on Windows."""
    if os.name == 'nt':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING (4) | ENABLE_PROCESSED_OUTPUT (1) | ENABLE_WRAP_AT_EOL_OUTPUT (2) = 7
            k32_stdout = -11
            handle = kernel32.GetStdHandle(k32_stdout)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 7)
        except Exception:
            pass


def setup_signal_handlers():
    """Registers signal handlers for SIGINT and SIGTERM."""
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Windows Console Handler for "X" button
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            # Define handler type
            HandlerRoutine = ctypes.WINFUNCTYPE(
                wintypes.BOOL, wintypes.DWORD
            )

            def ctrl_handler(ctrl_type):
                # 0: CTRL_C_EVENT
                # 1: CTRL_BREAK_EVENT
                # 2: CTRL_CLOSE_EVENT
                # 5: CTRL_LOGOFF_EVENT
                # 6: CTRL_SHUTDOWN_EVENT
                if ctrl_type in (0, 1, 2, 5, 6):
                    handle_shutdown(None, None)
                    return True
                return False

            # Keep reference alive to prevent GC
            global _win32_ctrl_handler
            _win32_ctrl_handler = HandlerRoutine(ctrl_handler)

            kernel32 = ctypes.windll.kernel32
            if not kernel32.SetConsoleCtrlHandler(_win32_ctrl_handler, True):
                print("[Warning] Failed to set Windows Console Handler")

        except Exception as e:
            print(f"[Warning] Error setting up Windows handler: {e}")


def log(message, level="INFO", to_console=True):
    """Logs a message to both console and log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    entry = f"[{timestamp}] [{level}] {message}"

    # If debug logging is ON, everything goes to console.
    # If OFF, DEBUG messages are skipped.
    should_print = to_console and (
        level != "DEBUG" or config.DEBUG_LOGGING
    )

    if should_print:
        prefix = {
            "ERROR": "!!! ", "WARNING": "! ", "CRITICAL": "XXX "
        }.get(level, "")
        # Use \r\033[K to clear any active progress bar on the current line
        print(f"\r\033[K{prefix}{message}")

    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")
    sys.stdout.flush()


def _format_time_component(seconds):
    """Formats seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def _get_progress_info(elapsed, speed, speed_unit, eta, timestamp_str, suffix):
    """Gathers and formats all progress metadata."""
    parts = []
    if timestamp_str:
        parts.append(timestamp_str)
    elif elapsed is not None:
        parts.append(_format_time_component(elapsed))

    if eta is not None:
        try:
            if float(eta) > 0:
                parts.append(f'ETA {_format_time_component(float(eta))}')
        except (TypeError, ValueError):
            pass

    if speed is not None:
        try:
            parts.append(f'{float(speed):.2f}{speed_unit}')
        except (TypeError, ValueError):
            pass

    if suffix:
        parts.append(suffix)
    return parts


def print_progress_bar(
    iteration, total, prefix='', suffix='', decimals=1, length=20,
    fill='█', empty='░', elapsed=None, speed=None, speed_unit='x',
    eta=None, no_newline=False, timestamp_str=None
):
    """
    Call in a loop to create terminal progress bar.
    """
    import shutil
    # Defensive handling for non-numeric inputs (e.g. mocks in tests)
    try:
        tot = float(total) if total and float(total) > 0 else 1.0
        it = float(iteration)
    except (TypeError, ValueError):
        it, tot = 0.0, 1.0

    percent_f = 100 * (it / tot)
    percent_s = ("{0:." + str(decimals) + "f}").format(percent_f)
    filled_l = int(length * it // tot)
    bar = fill * filled_l + empty * (length - filled_l)

    info_parts = [f'{percent_s:>5}%']
    info_parts.extend(
        _get_progress_info(elapsed, speed, speed_unit, eta, timestamp_str, suffix)
    )

    info_display = ' | '.join(info_parts)
    full_bar = f'[{bar}] {info_display}'

    # Terminal Width Awareness to prevent wrapping repetition on Windows
    term_width = shutil.get_terminal_size((80, 20)).columns - 1

    # Truncate prefix if needed
    max_prefix = max(10, term_width - len(full_bar) - 5)
    if len(prefix) > max_prefix:
        prefix = "..." + prefix[-(max_prefix - 3):]

    final_str = f'{prefix}{full_bar}'

    # Use \r\033[K for in-place update. print(..., end='', flush=True) is safer for some wrappers.
    try:
        sys.stdout.write(f'\r\033[K{final_str}')
        sys.stdout.flush()
    except UnicodeEncodeError:
        # Fallback to ASCII
        safe_bar = '#' * int(filled_l) + '-' * (int(length) - int(filled_l))
        full_bar_safe = f'[{safe_bar}] {info_display}'
        # Re-calc prefix truncate for safe bar? Assuming similar length.
        final_str_safe = f'{prefix}{full_bar_safe}'
        sys.stdout.write(f'\r{final_str_safe}')
        sys.stdout.flush()

    # Print new line on complete
    if iteration >= total and not no_newline:
        print()


# =============================================================================
# TIME & FILE UTILS
# =============================================================================

def get_ffmpeg_paths():
    """Returns paths to FFmpeg binaries, preferring local venv installation."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_ffmpeg = os.path.join(base, "venv", "ffmpeg", "bin", "ffmpeg.exe")
    venv_ffprobe = os.path.join(base, "venv", "ffmpeg", "bin", "ffprobe.exe")

    if os.path.exists(venv_ffmpeg) and os.path.exists(venv_ffprobe):
        return venv_ffmpeg, venv_ffprobe
    return "ffmpeg", "ffprobe"


FFMPEG_CMD, FFPROBE_CMD = get_ffmpeg_paths()


def get_audio_duration(file_path):
    """Returns duration of audio file in seconds."""
    try:
        cmd = [
            FFPROBE_CMD, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path
        ]
        return float(subprocess.check_output(cmd).decode().strip())
    except Exception:
        return 0.0


def format_timestamp(seconds):
    """Converts seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{math.floor(seconds):02d},"
        f"{milliseconds:03d}"
    )


def parse_timestamp(ts_str):
    """Converts SRT timestamp (HH:MM:SS,mmm) to seconds."""
    try:
        if ':' not in ts_str:
            return 0.0
        h, m, s_ms = ts_str.split(':')
        if ',' in s_ms:
            s, ms = s_ms.split(',')
        elif '.' in s_ms:
            s, ms = s_ms.split('.')
        else:
            s, ms = s_ms, "0"
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    except Exception:
        return 0.0


def _process_ffmpeg_line(line, start_time, total_duration, desc):
    """Helper to parse progress line from FFmpeg."""
    if line and "time=" in line:
        try:
            time_str = line.split("time=")[1].split()[0]
            current_seconds = parse_timestamp(
                time_str.replace('.', ',')
            )
            if total_duration > 0:
                elapsed = time.time() - start_time
                speed = current_seconds / elapsed if elapsed > 0 else 0
                eta = (
                    (total_duration - current_seconds) / speed
                    if speed > 0 else 0
                )

                print_progress_bar(
                    current_seconds, total_duration,
                    prefix=desc,
                    elapsed=elapsed,
                    speed=speed,
                    eta=eta
                )
        except Exception:
            # GC/Cache clear
            import gc
            gc.collect()
            pass


def _monitor_ffmpeg_process(process, start_time, total_duration, desc):
    """Monitors FFmpeg stderr for progress updates."""
    while True:
        line = process.stderr.readline()
        if not line and process.poll() is not None:
            break

        _process_ffmpeg_line(line, start_time, total_duration, desc)


def _finalize_ffmpeg_progress(process, cmd, start_time, total_duration, desc):
    """Handles final progress update and return code check."""
    # Ensure 100% at the end
    if total_duration > 0:
        elapsed = time.time() - start_time
        print_progress_bar(
            total_duration, total_duration,
            prefix=desc,
            elapsed=elapsed,
            speed=total_duration / elapsed if elapsed > 0 else 1.0
        )

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def run_ffmpeg_progress(cmd, desc, total_duration):
    """Executes FFmpeg command with a real-time progress bar UI."""
    try:
        start_time = time.time()
        # Popen to capture progress
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=(
                subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            ),
            encoding="utf-8",
            errors="replace"
        )
        register_subprocess(process)

        _monitor_ffmpeg_process(process, start_time, total_duration, desc)
        _finalize_ffmpeg_progress(process, cmd, start_time, total_duration, desc)

    except Exception as e:
        raise e
    finally:
        if 'process' in locals():
            unregister_subprocess(process)


def _validate_clean_audio_file(temp_wav):
    """Verifies that the extracted audio file is valid."""
    if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) < 1024:
        raise RuntimeError("Extracted audio is invalid/empty.")


def extract_clean_audio(video_path):
    """Extracts audio from video, normalizes volume, and returns WAV path."""
    base_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_wav = os.path.join(base_dir, f"{base_name}_temp.wav")

    # Reuse existing temp file if valid
    if os.path.exists(temp_wav):
        try:
            dur = get_audio_duration(temp_wav)
            if dur > 0:
                log("  [Pre-Process] Found valid existing temp audio.")
                return temp_wav
        except Exception:
            pass

    log("  [Pre-Process] Extracting & Normalizing Audio...", "INFO")

    cmd = [
        FFMPEG_CMD, "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_f32le",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        temp_wav
    ]

    try:
        total_dur = get_audio_duration(video_path)
        run_ffmpeg_progress(cmd, "  [Sample] Extracting Audio", total_dur)
    except Exception as e:
        log(f"Audio extraction failed: {e}", "ERROR")
        if os.path.exists(temp_wav):
            for _ in range(3):
                try:
                    os.remove(temp_wav)
                    break
                except OSError:
                    time.sleep(0.5)
        raise e

    _validate_clean_audio_file(temp_wav)
    return temp_wav


def _is_temp_file(filename, base_name, video_filename):
    """Checks if a file is a temporary file related to the video."""
    if filename == video_filename:
        return False
    if not filename.startswith(base_name):
        return False
    return (
        filename.endswith(".wav") or
        filename.endswith(".mp3") or
        filename.endswith(".json") or
        filename.endswith(".False.srt")
    )


def cleanup_temp_files(folder, base_name, video_filename):
    """Clean up temporary WAV/MP3 files."""
    for f in os.listdir(folder):
        if _is_temp_file(f, base_name, video_filename):
            path = os.path.join(folder, f)
            for _ in range(3):  # Retry loop for Windows locks
                try:
                    os.remove(path)
                    break
                except OSError:
                    time.sleep(0.5)


def get_cpu_name():
    """Returns the processor name."""
    if sys.platform == "win32":
        try:
            import winreg
            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
            processor_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            return processor_name.strip()
        except Exception:
            pass
    import platform
    return platform.processor() or "Unknown CPU"


def save_srt(segments, path):
    """Saves segments to an SRT file using atomic write to prevent corruption."""
    # Safety: Enforce chronological order if caller passed unsorted list
    segments = sorted(segments, key=lambda s: s.start)

    temp_path = path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = format_timestamp(seg.start)
                end = format_timestamp(seg.end)
                f.write(f"{i}\n{start} --> {end}\n{seg.text}\n\n")

        # Atomic replace (handles overwrite on Windows Python 3.3+)
        os.replace(temp_path, path)
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise e


def save_translated_srt(segments, translations, path):
    """Saves translated segments to an SRT file."""
    from .models import Segment

    final_segments = []
    for i, seg in enumerate(segments):
        text = translations[i] if i < len(translations) else "[Missing]"
        final_segments.append(Segment(seg.start, seg.end, text))
    save_srt(final_segments, path)


def _check_srt_corruption(line, next_line=None):
    """Helper to check for specific SRT corruption patterns in a line."""
    # If a line looks like index, its successor MUST be a ts
    if line.isdigit():
        if next_line:
            if " --> " not in next_line:
                return True  # Corruption: Index followed by something else

    if " --> " in line:
        # Check if the timestamp is physically impossible
        # (e.g. 5 digits in hours)
        parts = line.split(" --> ")
        for p in parts:
            if len(p.split(":")[0]) > 3:  # 01:23:45 -> [01]
                return True  # Corruption: Garbage like 31401:58
    return False


def validate_srt(path):
    """Checks for basic SRT markers to filter out obviously undefined files."""
    if not os.path.exists(path):
        return False

    try:
        # Check size (extremely small is suspicious for an SRT)
        if os.path.getsize(path) < 10:
            return False

        # Use utf-8-sig to handle BOM automatically
        with open(path, "r", encoding="utf-8-sig") as f:
            content = f.read(4096)  # Check first 4KB

            stripped = content.strip()
            if not stripped:
                return False

            # Basic SRT Signature:
            # 1. Starts with a number (Index)
            # 2. Contains timestamp separator

            # Check 1: First non-whitespace char is digit
            if not stripped[0].isdigit():
                return False

            # Check 2: Contains " --> "
            if " --> " not in stripped:
                return False

        return True
    except Exception:
        return False


def parse_srt(path):
    """Parses an SRT file back into a list of Segment objects."""
    from .models import Segment
    # CRITICAL: Validate first
    if not validate_srt(path):
        import logging
        logging.getLogger("Antigravity").warning(
            f"  [Guard] Rejected corrupted SRT: {os.path.basename(path)}"
        )
        return []

    segments = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().split("\n\n")
        for chunk in content:
            lines = chunk.strip().split("\n")
            if len(lines) >= 3:
                # 0: Index, 1: Time, 2+: Text
                if not lines[0].strip().isdigit():
                    continue  # Skip garbage chunks

                time_range = lines[1]
                if " --> " in time_range:
                    try:
                        start_str, end_str = time_range.split(" --> ")
                        start = parse_timestamp(start_str)
                        end = parse_timestamp(end_str)
                        text = " ".join(lines[2:])
                        segments.append(Segment(start, end, text))
                    except Exception:
                        continue  # Skip invalid timestamps

    return segments
