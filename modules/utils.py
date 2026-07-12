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
import ctypes
import gc
import importlib
import logging
import platform
import shutil
from ctypes import wintypes

try:
    import winreg
except ImportError:
    winreg = None

from . import config

# =============================================================================
# LOGGING & SIGNALS
# =============================================================================


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
    print("   AI HYBRID VHS AUDIO RESTORER - v1.1.1")
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
_WIN32_CTRL_HANDLER = None


def _get_segment_class():
    """Return the Segment class without a static import to avoid cycles."""
    return importlib.import_module("modules.models").Segment


def register_subprocess(proc):
    """Registers a subprocess to be killed on shutdown."""
    active_subprocesses.append(proc)


def unregister_subprocess(proc):
    """Unregisters a subprocess (e.g., after clean completion)."""
    if proc in active_subprocesses:
        active_subprocesses.remove(proc)


def handle_shutdown(_signum, _frame):
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
                    subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError as e:
                print(f"  [Cleanup] Error killing process: {e}")

    sys.exit(0)


def init_console():
    """Initializes the console for ANSI support, especially on Windows."""
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING (4) | ENABLE_PROCESSED_OUTPUT (1) | ENABLE_WRAP_AT_EOL_OUTPUT (2) = 7
            k32_stdout = -11
            handle = kernel32.GetStdHandle(k32_stdout)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 7)
        except (AttributeError, OSError):
            pass


def setup_signal_handlers():
    """Registers signal handlers for SIGINT and SIGTERM."""
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Windows Console Handler for "X" button
    if sys.platform == "win32":
        try:
            # Define handler type
            handler_routine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

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

    with open(config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")
    sys.stdout.flush()


def _format_time_component(seconds):
    """Formats seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _get_progress_info(progress_options):
    """Gathers and formats all progress metadata."""
    elapsed = progress_options.get("elapsed")
    speed = progress_options.get("speed")
    speed_unit = progress_options.get("speed_unit", "x")
    eta = progress_options.get("eta")
    timestamp_str = progress_options.get("timestamp_str")
    suffix = progress_options.get("suffix")
    parts = []
    if timestamp_str:
        parts.append(timestamp_str)
    elif elapsed is not None:
        parts.append(_format_time_component(elapsed))

    if eta is not None:
        try:
            if float(eta) > 0:
                parts.append(f"ETA {_format_time_component(float(eta))}")
        except (TypeError, ValueError):
            pass

    if speed is not None:
        try:
            parts.append(f"{float(speed):.2f}{speed_unit}")
        except (TypeError, ValueError):
            pass

    if suffix:
        parts.append(suffix)
    return parts


def _get_progress_style(progress_options):
    """Return the display style values for a progress bar."""
    return {
        "prefix": progress_options.get("prefix", ""),
        "suffix": progress_options.get("suffix", ""),
        "decimals": progress_options.get("decimals", 1),
        "length": progress_options.get("length", 20),
        "fill": progress_options.get("fill", "█"),
        "empty": progress_options.get("empty", "░"),
    }


def _normalize_progress_numbers(iteration, total):
    """Normalize progress numbers for display."""
    try:
        normalized_total = float(total) if total and float(total) > 0 else 1.0
        normalized_iteration = float(iteration)
    except (TypeError, ValueError):
        normalized_iteration, normalized_total = 0.0, 1.0
    return normalized_iteration, normalized_total


def _build_progress_bars(normalized_iteration, normalized_total, style, progress_options):
    """Build the rich and ASCII progress-bar variants."""
    percent_f = 100 * (normalized_iteration / normalized_total)
    percent_s = ("{0:." + str(style["decimals"]) + "f}").format(percent_f)
    filled_length = int(style["length"] * normalized_iteration // normalized_total)
    progress_bar = style["fill"] * filled_length + style["empty"] * (style["length"] - filled_length)

    info_parts = [f"{percent_s:>5}%"]
    info_parts.extend(_get_progress_info(progress_options | {"suffix": style["suffix"]}))
    info_display = " | ".join(info_parts)
    rich_bar = f"[{progress_bar}] {info_display}"
    safe_bar = "#" * int(filled_length) + "-" * (int(style["length"]) - int(filled_length))
    return rich_bar, f"[{safe_bar}] {info_display}"


def _truncate_progress_prefix(prefix, bar_text):
    """Trim the prefix to fit the current terminal width."""
    term_width = shutil.get_terminal_size((80, 20)).columns - 1
    max_prefix = max(10, term_width - len(bar_text) - 5)
    if len(prefix) > max_prefix:
        return "..." + prefix[-(max_prefix - 3) :]
    return prefix


def _build_progress_display(iteration, total, progress_options):
    """Build the rendered progress-bar string and completion flag."""
    style = _get_progress_style(progress_options)
    normalized_iteration, normalized_total = _normalize_progress_numbers(iteration, total)
    rich_bar, safe_bar = _build_progress_bars(
        normalized_iteration,
        normalized_total,
        style,
        progress_options,
    )
    prefix = _truncate_progress_prefix(style["prefix"], rich_bar)
    return (
        f"{prefix}{rich_bar}",
        f"{prefix}{safe_bar}",
        normalized_iteration >= normalized_total,
    )


def print_progress_bar(iteration, total, **progress_options):
    """
    Call in a loop to create terminal progress bar.
    """
    no_newline = progress_options.get("no_newline", False)
    final_str, safe_final_str, is_complete = _build_progress_display(
        iteration,
        total,
        progress_options,
    )

    # Use \r\033[K for in-place update. print(..., end='', flush=True) is safer for some wrappers.
    try:
        sys.stdout.write(f"\r\033[K{final_str}")
        sys.stdout.flush()
    except UnicodeEncodeError:
        sys.stdout.write(f"\r{safe_final_str}")
        sys.stdout.flush()

    # Print new line on complete
    if is_complete and not no_newline:
        print()


# =============================================================================
# TIME & FILE UTILS
# =============================================================================


def _format_elapsed_time(seconds):
    """Formats elapsed seconds as HH:MM:SS."""
    safe_seconds = max(0, int(seconds))
    hours = safe_seconds // 3600
    minutes = (safe_seconds % 3600) // 60
    remaining_seconds = safe_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def format_elapsed_time(seconds):
    """Public wrapper for elapsed-time formatting."""
    return _format_elapsed_time(seconds)


def _format_total_processing_speed(media_seconds, elapsed_seconds):
    """Builds the total processing speed text for one input file."""
    if elapsed_seconds <= 0 or media_seconds <= 0:
        return "N/A"

    speed = media_seconds / elapsed_seconds
    return f"{speed:.2f}x realtime"


def format_total_processing_speed(media_seconds, elapsed_seconds):
    """Public wrapper for processing speed summary formatting."""
    return _format_total_processing_speed(media_seconds, elapsed_seconds)


def _resolve_input_path(input_path):
    """Resolve the input path from CLI args or prompt."""
    path = input_path
    if not path:
        print(">> Please Drag & Drop a video file here and press Enter:")
        path = input(">>Path: ").strip().strip('"')
    return path or "input"


def resolve_input_path(input_path):
    """Public wrapper for resolving user-provided input paths."""
    return _resolve_input_path(input_path)


def _collect_video_files(path):
    """Collect supported input videos from a file or directory path."""
    files = []
    supported_extensions = {(ext if str(ext).startswith(".") else f".{ext}").lower() for ext in config.VIDEO_EXTENSIONS}

    if os.path.isfile(path):
        file_name = os.path.basename(path)
        file_stem, file_ext = os.path.splitext(file_name)
        if file_ext.lower() in supported_extensions and not file_stem.lower().endswith("_multilang"):
            files.append(os.path.abspath(path))
        return files

    if os.path.isdir(path):
        for root, _, filenames in os.walk(path):
            for file_name in filenames:
                file_stem, file_ext = os.path.splitext(file_name)
                if file_ext.lower() not in supported_extensions:
                    continue
                if file_stem.lower().endswith("_multilang"):
                    continue
                files.append(os.path.abspath(os.path.join(root, file_name)))
        return files

    log(f"Error: Path not found: {path}", "CRITICAL")
    sys.exit(1)


def collect_video_files(path):
    """Public wrapper for collecting supported video inputs."""
    return _collect_video_files(path)


def classify_batch_result(process_result):
    """Classify a process_video return value for batch counters."""
    if not (isinstance(process_result, tuple) and len(process_result) == 3):
        return "failed"

    segments = process_result[0]
    if segments is None:
        return "failed"
    if segments:
        return "succeeded"
    return "no_speech"


def build_file_summary(video_path, elapsed_seconds, status):
    """Build per-file metrics summary and return message, media duration, and batch item stats."""
    file_name = os.path.basename(video_path)
    elapsed_text = format_elapsed_time(elapsed_seconds)
    try:
        media_seconds = get_audio_duration(video_path)
        speed_summary = format_total_processing_speed(media_seconds, elapsed_seconds)
        media_text = format_elapsed_time(media_seconds) if media_seconds > 0 else "N/A"
        summary_message = (
            f"  [Summary] {file_name} | Total processing speed: {speed_summary} | Media duration: {media_text} | Elapsed: {elapsed_text}"
        )
        file_stats = {
            "file_name": file_name,
            "status": status,
            "media_text": media_text,
            "elapsed_text": elapsed_text,
            "speed_summary": speed_summary,
        }
        return summary_message, max(0.0, float(media_seconds)), file_stats
    except (OSError, ValueError, RuntimeError, TypeError) as e:
        log(f"  [Summary] Warning: failed to compute media metrics for {file_name}: {e}", "WARNING")
        return (
            f"  [Summary] {file_name} | Total processing speed: N/A | Media duration: N/A | Elapsed: {elapsed_text}",
            0.0,
            {
                "file_name": file_name,
                "status": status,
                "media_text": "N/A",
                "elapsed_text": elapsed_text,
                "speed_summary": "N/A",
            },
        )


def log_batch_summary(total_files, counters, total_media_seconds, batch_start_time, file_stats):
    """Log aggregate batch statistics and per-file details for multi-file runs."""
    if total_files <= 1 or batch_start_time is None:
        return

    total_elapsed = time.time() - batch_start_time
    total_media_text = format_elapsed_time(total_media_seconds) if total_media_seconds > 0 else "N/A"
    batch_speed = format_total_processing_speed(total_media_seconds, total_elapsed)
    batch_elapsed_text = format_elapsed_time(total_elapsed)
    log(
        f"  [Batch Summary] Files: {total_files} | "
        f"Succeeded: {counters['succeeded']} | "
        f"No speech: {counters['no_speech']} | "
        f"Failed: {counters['failed']} | "
        f"Media duration: {total_media_text} | Elapsed: {batch_elapsed_text} | Total processing speed: {batch_speed}",
        "INFO",
    )
    log("  [Batch Files]", "INFO")
    for item in file_stats:
        log(
            f"    - {item['file_name']} | "
            f"Status: {item['status']} | "
            f"Media: {item['media_text']} | "
            f"Elapsed: {item['elapsed_text']} | "
            f"Speed: {item['speed_summary']}",
            "INFO",
        )


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
        cmd = [FFPROBE_CMD, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        return float(subprocess.check_output(cmd).decode().strip())
    except Exception as exc:
        if isinstance(exc, (OSError, ValueError)):
            return 0.0

        called_process_error = getattr(subprocess, "CalledProcessError", None)
        if (
            isinstance(called_process_error, type)
            and issubclass(called_process_error, BaseException)
            and isinstance(exc, called_process_error)
        ):
            return 0.0

        raise


def format_timestamp(seconds):
    """Converts seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{math.floor(seconds):02d},{milliseconds:03d}"


def parse_timestamp(ts_str):
    """Converts SRT timestamp (HH:MM:SS,mmm) to seconds."""
    try:
        if ":" not in ts_str:
            return 0.0
        h, m, s_ms = ts_str.split(":")
        if "," in s_ms:
            s, ms = s_ms.split(",")
        elif "." in s_ms:
            s, ms = s_ms.split(".")
        else:
            s, ms = s_ms, "0"
        fraction_digits = len(ms)
        if fraction_digits == 0:
            fraction = 0.0
        elif fraction_digits == 1:
            fraction = int(ms) / 10.0
        elif fraction_digits == 2:
            fraction = int(ms) / 100.0
        elif fraction_digits == 3:
            fraction = int(ms) / 1000.0
        else:
            fraction = int(ms) / (10**fraction_digits)
        return int(h) * 3600 + int(m) * 60 + int(s) + fraction
    except (ValueError, TypeError):
        return 0.0


def _process_ffmpeg_line(line, start_time, total_duration, desc):
    """Helper to parse progress line from FFmpeg."""
    if line and "time=" in line:
        try:
            time_str = line.split("time=")[1].split()[0]
            current_seconds = parse_timestamp(time_str.replace(".", ","))
            if total_duration > 0:
                elapsed = time.time() - start_time
                speed = current_seconds / elapsed if elapsed > 0 else 0
                eta = (total_duration - current_seconds) / speed if speed > 0 else 0

                print_progress_bar(current_seconds, total_duration, prefix=desc, elapsed=elapsed, speed=speed, eta=eta)
        except (IndexError, TypeError, ValueError):
            gc.collect()


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
            total_duration, total_duration, prefix=desc, elapsed=elapsed, speed=total_duration / elapsed if elapsed > 0 else 1.0
        )

    if process.returncode != 0:
        error_cls = getattr(subprocess, "CalledProcessError", None)
        if isinstance(error_cls, type) and issubclass(error_cls, BaseException):
            raise error_cls(process.returncode, cmd)

        # Test environments may replace the subprocess module with a mock object.
        raise RuntimeError(f"FFmpeg command failed with return code {process.returncode}: {cmd}")


def run_ffmpeg_progress(cmd, desc, total_duration):
    """Executes FFmpeg command with a real-time progress bar UI."""
    start_time = time.time()
    with subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
        encoding="utf-8",
        errors="replace",
    ) as process:
        register_subprocess(process)
        try:
            _monitor_ffmpeg_process(process, start_time, total_duration, desc)
            _finalize_ffmpeg_progress(process, cmd, start_time, total_duration, desc)
        finally:
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
        except (OSError, ValueError):
            pass

    log("  [Pre-Process] Extracting & Normalizing Audio...", "INFO")

    cmd = [
        FFMPEG_CMD,
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_f32le",
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        temp_wav,
    ]

    try:
        total_dur = get_audio_duration(video_path)
        run_ffmpeg_progress(cmd, "  [Sample] Extracting Audio", total_dur)
    except (OSError, RuntimeError) as e:
        log(f"Audio extraction failed: {e}", "ERROR")
        if os.path.exists(temp_wav):
            for _ in range(3):
                try:
                    os.remove(temp_wav)
                    break
                except OSError:
                    time.sleep(0.5)
        raise

    _validate_clean_audio_file(temp_wav)
    return temp_wav


def _is_temp_file(filename, base_name, video_filename):
    """Checks if a file is a temporary file related to the video."""
    if filename == video_filename:
        return False
    is_temp_name = (
        filename.startswith(base_name)
        or filename.startswith(f".temp_output.{base_name}")
        or filename.startswith(f".temp_input.{base_name}")
    )
    if not is_temp_name:
        return False
    return filename.endswith(".wav") or filename.endswith(".mp3") or filename.endswith(".json")


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
    if sys.platform == "win32" and winreg is not None:
        try:
            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
            processor_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            return processor_name.strip()
        except OSError:
            pass
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
    except OSError:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise


def save_translated_srt(segments, translations, path):
    """Saves translated segments to an SRT file."""
    segment_cls = _get_segment_class()

    final_segments = []
    for i, seg in enumerate(segments):
        text = translations[i] if i < len(translations) else "[Missing]"
        final_segments.append(segment_cls(seg.start, seg.end, text))
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
    is_valid = False
    if not os.path.exists(path):
        return is_valid

    try:
        # Check size (extremely small is suspicious for an SRT)
        if os.path.getsize(path) < 10:
            return is_valid

        # Use utf-8-sig to handle BOM automatically
        with open(path, "r", encoding="utf-8-sig") as f:
            content = f.read(4096)  # Check first 4KB

            stripped = content.strip()
            if not stripped:
                return is_valid

            # Basic SRT Signature:
            # 1. Starts with a number (Index)
            # 2. Contains timestamp separator

            # Check 1: First non-whitespace char is digit
            if not stripped[0].isdigit():
                return is_valid

            # Check 2: Contains " --> "
            if " --> " not in stripped:
                return is_valid

        is_valid = True
    except OSError:
        is_valid = False

    return is_valid


def parse_srt(path):
    """Parses an SRT file back into a list of Segment objects."""
    segment_cls = _get_segment_class()
    # CRITICAL: Validate first
    if not validate_srt(path):
        logging.getLogger("Antigravity").warning(
            "  [Guard] Rejected corrupted SRT: %s",
            os.path.basename(path),
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
                        segments.append(segment_cls(start, end, text))
                    except (ValueError, IndexError):
                        continue  # Skip invalid timestamps

    return segments
