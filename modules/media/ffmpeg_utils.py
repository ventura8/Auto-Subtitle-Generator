"""FFmpeg operations and audio extraction utilities."""

import gc
import os
import shutil
import subprocess
import sys
import time

from ..configuration import config
from ..runtime.logging_utils import log, register_subprocess, unregister_subprocess
from ..runtime.progress import print_progress_bar
from ..safe_io import discard_temp_path, promote_temp_path, reserve_temp_path
from ..subtitles.timestamp_utils import parse_timestamp
from ..workdir import ensure_work_dir
from .input_binding import media_source, rewind_inputs


def _resolve_ffmpeg_pair(bin_dir, ext):
    """Return (ffmpeg, ffprobe) paths from a candidate directory, or None."""
    ffmpeg_path = os.path.join(bin_dir, f"ffmpeg{ext}")
    ffprobe_path = os.path.join(bin_dir, f"ffprobe{ext}")
    if (
        os.path.isfile(ffmpeg_path)
        and os.access(ffmpeg_path, os.X_OK)
        and os.path.isfile(ffprobe_path)
        and os.access(ffprobe_path, os.X_OK)
    ):
        return ffmpeg_path, ffprobe_path
    return None


def _iter_ffmpeg_candidates():
    """Yield (bin_dir, extension) candidates for local FFmpeg discovery."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    extensions = [".exe"] if sys.platform == "win32" else [""]
    bin_dirs = [
        os.path.join(base, ".venv", "ffmpeg", "bin"),
        os.path.join(base, ".venv", "Scripts"),
        os.path.join(base, ".venv", "bin"),
        # The active environment may differ from the repository-root .venv.
        os.path.join(sys.prefix, "ffmpeg", "bin"),
        os.path.join(sys.prefix, "Scripts"),
        os.path.join(sys.prefix, "bin"),
    ]
    for bin_dir in bin_dirs:
        for ext in extensions:
            yield bin_dir, ext


def _resolve_installed_ffmpeg_pair():
    """Return the PATH-resolved ffmpeg/ffprobe pair when both are installed."""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if ffmpeg_path and ffprobe_path:
        return ffmpeg_path, ffprobe_path
    return None


def get_ffmpeg_paths():
    """Returns paths to FFmpeg binaries, preferring an installed system FFmpeg.

    Project rule: an installed dependency always wins over a bundled or
    locally built copy, matching install_dependencies.sh, which also probes
    the system FFmpeg first. The bundled venv copies are only a fallback.
    """
    installed_pair = _resolve_installed_ffmpeg_pair()
    if installed_pair is not None:
        return installed_pair
    for bin_dir, ext in _iter_ffmpeg_candidates():
        pair = _resolve_ffmpeg_pair(bin_dir, ext)
        if pair is not None:
            return pair
    return "ffmpeg", "ffprobe"


FFMPEG_CMD, FFPROBE_CMD = get_ffmpeg_paths()


def get_audio_duration(file_path):
    """Returns duration of audio file in seconds."""
    try:
        source, pass_fds = media_source(file_path)
        cmd = [FFPROBE_CMD, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", source]
        rewind_inputs(pass_fds)
        return float(subprocess.check_output(cmd, timeout=30, pass_fds=pass_fds).decode().strip())
    except Exception as exc:
        if isinstance(exc, (OSError, ValueError)) or _is_called_process_error(exc):
            return 0.0

        raise


def extract_clean_audio(video_path):
    """Extracts audio from video, normalizes volume, and returns WAV path.

    The WAV lives in the per-video work directory (``<base>.asg-temp/``) so it
    can be reused on resume and is removed together with the directory once
    the video is finished.
    """
    base_dir = os.path.dirname(video_path) or "."
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    work_dir = ensure_work_dir(base_dir, base_name)
    temp_wav = os.path.join(work_dir, f"{base_name}_temp.wav")

    if _has_valid_temp_audio(temp_wav):
        log("  [Pre-Process] Found valid existing temp audio.")
        return temp_wav

    log("  [Pre-Process] Extracting & Normalizing Audio...", "INFO")

    # FFmpeg -y follows symlinks, so extract into a private scratch file and promote.
    scratch = reserve_temp_path(temp_wav, scratch_dir=work_dir)
    source, pass_fds = media_source(video_path)
    cmd = [
        FFMPEG_CMD,
        "-y",
        "-i",
        source,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_f32le",
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        # A plain RIFF WAV caps at 4 GB (about 17 h at this format); RF64 lifts that.
        "-rf64",
        "auto",
        scratch.path,
    ]

    promoted = False
    try:
        total_dur = get_audio_duration(video_path)
        run_ffmpeg_progress(cmd, "  [Sample] Extracting Audio", total_dur, pass_fds=pass_fds)
        _validate_clean_audio_file(scratch.path)
        promote_temp_path(scratch, temp_wav)
        promoted = True
    except (OSError, RuntimeError) as e:
        log(f"Audio extraction failed: {e}", "ERROR")
        raise
    finally:
        if not promoted:
            discard_temp_path(scratch)
    return temp_wav


def run_ffmpeg_quiet(cmd):
    """Run an FFmpeg command with no progress UI, raising RuntimeError on a non-zero exit.

    Used for the many short cut/trim/join operations of chunked vocal
    separation, where a progress bar per call would only be noise.
    """
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
            _, stderr_text = process.communicate()
        finally:
            unregister_subprocess(process)
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed ({process.returncode}): {(stderr_text or '').strip()[-400:]}")


def write_audio_window(source_path, target_path, start_seconds, length_seconds, mono_16k=False):
    """Write ``length_seconds`` of ``source_path`` starting at ``start_seconds`` as float WAV.

    ``mono_16k`` downmixes and resamples to the 16 kHz mono format Whisper
    consumes, which also keeps a joined multi-hour stem far below the RIFF
    limit. ``target_path`` must be a private scratch path: FFmpeg ``-y``
    follows symlinks.
    """
    cmd = [FFMPEG_CMD, "-y", "-loglevel", "error", "-ss", f"{start_seconds:.3f}", "-t", f"{length_seconds:.3f}", "-i", source_path]
    if mono_16k:
        cmd += ["-ac", "1", "-ar", "16000"]
    cmd += ["-c:a", "pcm_f32le", "-rf64", "auto", target_path]
    run_ffmpeg_quiet(cmd)


def concat_audio_files(source_paths, target_path, list_path):
    """Join same-format WAV files in order into ``target_path`` (RF64 once it outgrows RIFF).

    ``list_path`` is where the concat manifest is written; it must live inside
    a private scratch directory owned by this process, like ``target_path``.
    """
    # FFmpeg resolves relative manifest entries against the manifest's own
    # directory, not the working directory, so a relative input path would
    # point the join at the wrong place.
    with open(list_path, "w", encoding="utf-8") as handle:
        for path in source_paths:
            handle.write(f"file '{_concat_quote(os.path.abspath(path))}'\n")
    run_ffmpeg_quiet(
        [
            FFMPEG_CMD,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c:a",
            "copy",
            "-rf64",
            "auto",
            target_path,
        ]
    )


def _concat_quote(path):
    """Escape a path for the FFmpeg concat demuxer's single-quoted ``file`` directive."""
    return path.replace("'", "'\\''")


def run_ffmpeg_progress(cmd, desc, total_duration, pass_fds=()):
    """Executes FFmpeg command with a real-time progress bar UI.

    ``pass_fds`` inherits descriptor-bound inputs (see ``input_binding``) into the child.
    """
    start_time = time.time()
    rewind_inputs(pass_fds)
    with subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=(subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0),
        encoding="utf-8",
        errors="replace",
        pass_fds=pass_fds,
    ) as process:
        register_subprocess(process)
        try:
            _monitor_ffmpeg_process(process, start_time, total_duration, desc)
            _finalize_ffmpeg_progress(process, cmd, start_time, total_duration, desc)
        finally:
            unregister_subprocess(process)


def _is_called_process_error(exc):
    """Return True when exception is subprocess.CalledProcessError-compatible."""
    called_process_error = getattr(subprocess, "CalledProcessError", None)
    return (
        isinstance(called_process_error, type) and issubclass(called_process_error, BaseException) and isinstance(exc, called_process_error)
    )


def _has_valid_temp_audio(temp_wav):
    """Return True when a temp WAV exists, is not a symlink, and has measurable duration."""
    if not os.path.exists(temp_wav) or os.path.islink(temp_wav):
        return False
    try:
        return get_audio_duration(temp_wav) > 0
    except (OSError, ValueError):
        return False


def _validate_clean_audio_file(temp_wav):
    """Verifies that the extracted audio file is valid."""
    if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) < 1024:
        raise RuntimeError("Extracted audio is invalid/empty.")


def _process_ffmpeg_line(line, start_time, total_duration, desc):
    """Helper to parse progress line from FFmpeg."""
    if not line or "time=" not in line:
        return
    try:
        time_str = line.split("time=")[1].split()[0]
        current_seconds = parse_timestamp(time_str.replace(".", ","))
        if total_duration <= 0:
            return
        elapsed, speed, eta = _compute_progress_metrics(current_seconds, start_time, total_duration)
        print_progress_bar(current_seconds, total_duration, prefix=desc, elapsed=elapsed, speed=speed, eta=eta)
    except (IndexError, TypeError, ValueError):
        gc.collect()


def _compute_progress_metrics(current_seconds, start_time, total_duration):
    """Compute elapsed, speed, and ETA for FFmpeg progress reporting."""
    elapsed = time.time() - start_time
    speed = current_seconds / elapsed if elapsed > 0 else 0
    eta = (total_duration - current_seconds) / speed if speed > 0 else 0
    return elapsed, speed, eta


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
        _raise_ffmpeg_failure(process.returncode, cmd)


def _raise_ffmpeg_failure(return_code, cmd):
    """Raise subprocess-compatible FFmpeg failure exception."""
    error_cls = getattr(subprocess, "CalledProcessError", None)
    if isinstance(error_cls, type) and issubclass(error_cls, BaseException):
        raise error_cls(return_code, cmd)
    raise RuntimeError(f"FFmpeg command failed with return code {return_code}: {cmd}")


def build_primary_media_metadata_args(src_lang):
    """Tag copied primary media streams with the detected source language."""
    source_language = config.to_mux_language_code(src_lang)
    return [
        "-metadata:s:v:0",
        f"language={source_language}",
        "-metadata:s:a:0",
        f"language={source_language}",
    ]
