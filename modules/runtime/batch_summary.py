"""Batch processing summary and statistics utilities."""

import os
import time

from ..media.ffmpeg_utils import get_audio_duration
from ..runtime.logging_utils import log
from ..subtitles.timestamp_utils import format_elapsed_time, format_total_processing_speed


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
    succeeded_count = counters.get("succeeded", 0)
    no_speech_count = counters.get("no_speech", 0)
    failed_count = counters.get("failed", 0)
    log(
        f"  [Batch Summary] Files: {total_files} | "
        f"Succeeded: {succeeded_count} | "
        f"No speech: {no_speech_count} | "
        f"Failed: {failed_count} | "
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
