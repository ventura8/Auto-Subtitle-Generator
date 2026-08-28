"""
Utility module for Auto Subtitle Generator.
Handles logging, timestamps, progress bars, and FFmpeg operations.

This module provides backward compatibility by re-exporting all utilities
that were previously in this file. The implementation has been split into
focused modules for better maintainability.
"""

import importlib
import os
import time

# Re-export FFmpeg utilities
from .media.ffmpeg_utils import FFMPEG_CMD, FFPROBE_CMD, extract_clean_audio, get_audio_duration, get_ffmpeg_paths, run_ffmpeg_progress

# Re-export file utilities
from .media.file_utils import collect_video_files, resolve_input_path

# Re-export hardware utilities
from .media.hardware_utils import get_cpu_name

# Re-export batch summary utilities
from .runtime.batch_summary import build_file_summary, classify_batch_result, log_batch_summary

# Re-export logging utilities
from .runtime.logging_utils import (
    handle_shutdown,
    init_console,
    log,
    print_banner,
    register_subprocess,
    setup_signal_handlers,
    unregister_subprocess,
)

# Re-export progress bar utilities
from .runtime.progress import print_progress_bar

# Re-export SRT I/O utilities
from .subtitles.srt_io import parse_srt, save_srt, save_translated_srt, validate_srt

# Re-export timestamp utilities
from .subtitles.timestamp_utils import format_elapsed_time, format_timestamp, format_total_processing_speed, parse_timestamp


def _get_segment_class():
    """Return the Segment class without a static import to avoid cycles."""
    return importlib.import_module("modules.models").Segment


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


TEMP_EXTENSIONS = (".wav", ".mp3", ".json", ".tmp")


def _is_anonymous_temp_file(filename):
    """Return True if filename matches standard anonymous temp file patterns."""
    return filename.startswith("tmp") and len(filename) > 3 and "." not in filename


def _is_temp_file(filename, base_name, video_filename):
    """Checks if a file is a temporary file related to the video."""
    if filename == video_filename:
        return False
    if _is_anonymous_temp_file(filename):
        return True
    return _has_temp_name_prefix(filename, base_name) and filename.endswith(TEMP_EXTENSIONS)


def _has_temp_name_prefix(filename, base_name):
    """Return True when filename matches known temp naming prefixes."""
    expected_prefix = f"{base_name}_temp"
    has_expected = filename.startswith(expected_prefix) and (
        len(filename) == len(expected_prefix) or filename[len(expected_prefix)] in {"_", "."}
    )
    prefixes = (f".temp_output.{base_name}.", f".temp_input.{base_name}", f"{base_name}.")
    return has_expected or filename.startswith(prefixes)


def _has_temp_extension(filename):
    """Return True for known temporary media/manifest extensions."""
    return filename.endswith(TEMP_EXTENSIONS)


__all__ = [
    # Logging
    "print_banner",
    "register_subprocess",
    "unregister_subprocess",
    "handle_shutdown",
    "init_console",
    "setup_signal_handlers",
    "log",
    # Progress
    "print_progress_bar",
    # Timestamps
    "format_timestamp",
    "parse_timestamp",
    "format_elapsed_time",
    "format_total_processing_speed",
    # Files
    "collect_video_files",
    "resolve_input_path",
    "cleanup_temp_files",
    # FFmpeg
    "get_ffmpeg_paths",
    "FFMPEG_CMD",
    "FFPROBE_CMD",
    "get_audio_duration",
    "extract_clean_audio",
    "run_ffmpeg_progress",
    # SRT I/O
    "save_srt",
    "save_translated_srt",
    "parse_srt",
    "validate_srt",
    # Batch
    "classify_batch_result",
    "build_file_summary",
    "log_batch_summary",
    # Hardware
    "get_cpu_name",
]
