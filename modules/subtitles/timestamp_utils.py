"""Timestamp and time formatting utilities."""

import math


def format_timestamp(seconds):
    """Converts seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    seconds = max(0.0, float(seconds))
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    whole_seconds = math.floor(seconds)
    milliseconds = round((seconds - whole_seconds) * 1000)
    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0
    if whole_seconds == 60:
        minutes += 1
        whole_seconds = 0
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def parse_timestamp(ts_str, strict=False):
    """Converts SRT timestamp (HH:MM:SS,mmm) to seconds."""
    try:
        h, m, s_ms = _split_timestamp_parts(ts_str, strict)
        hours, minutes, seconds, milliseconds = _parse_timestamp_components(h, m, s_ms)
        _validate_timestamp_components(hours, minutes, seconds, strict)
        fraction = _fraction_from_digits(milliseconds)
        return hours * 3600 + minutes * 60 + seconds + fraction
    except (ValueError, TypeError):
        if strict:
            raise
        return 0.0


def _split_timestamp_parts(ts_str, strict):
    """Split timestamp into hour, minute, and second tokens."""
    if ":" not in ts_str:
        if strict:
            raise ValueError("Timestamp missing ':' separators")
        raise ValueError("Timestamp missing separators")
    return ts_str.split(":")


def _parse_timestamp_components(hours_token, minutes_token, seconds_token):
    """Parse timestamp string components into integer hour/minute/second values."""
    seconds_part, milliseconds = _split_seconds_and_fraction(seconds_token)
    return int(hours_token), int(minutes_token), int(seconds_part), milliseconds


def _validate_timestamp_components(hours, minutes, seconds, strict):
    """Validate strict SRT component ranges."""
    if not strict:
        return
    if _has_negative_timestamp_component(hours, minutes, seconds):
        raise ValueError("Timestamp components must be non-negative")
    if _has_out_of_range_timestamp_component(minutes, seconds):
        raise ValueError("Timestamp minutes and seconds must be below 60")


def _has_negative_timestamp_component(hours, minutes, seconds):
    """Return True when any parsed timestamp component is negative."""
    return hours < 0 or minutes < 0 or seconds < 0


def _has_out_of_range_timestamp_component(minutes, seconds):
    """Return True when strict minute/second ranges exceed SRT bounds."""
    return minutes >= 60 or seconds >= 60


def format_elapsed_time(seconds):
    """Public wrapper for elapsed-time formatting."""
    return _format_elapsed_time(seconds)


def format_total_processing_speed(media_seconds, elapsed_seconds):
    """Public wrapper for processing speed summary formatting."""
    return _format_total_processing_speed(media_seconds, elapsed_seconds)


def _format_elapsed_time(seconds):
    """Formats elapsed seconds as HH:MM:SS."""
    safe_seconds = max(0, int(seconds))
    hours = safe_seconds // 3600
    minutes = (safe_seconds % 3600) // 60
    remaining_seconds = safe_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def _format_total_processing_speed(media_seconds, elapsed_seconds):
    """Builds the total processing speed text for one input file."""
    if elapsed_seconds <= 0 or media_seconds <= 0:
        return "N/A"

    speed = media_seconds / elapsed_seconds
    return f"{speed:.2f}x realtime"


def _split_seconds_and_fraction(seconds_token):
    """Split second token into integer seconds and fractional part string."""
    if "," in seconds_token:
        return seconds_token.split(",", maxsplit=1)
    if "." in seconds_token:
        return seconds_token.split(".", maxsplit=1)
    return seconds_token, "0"


def _fraction_from_digits(fraction_digits):
    """Convert fractional seconds digits to float seconds."""
    if fraction_digits.startswith("-"):
        raise ValueError("Fractional timestamp component must be non-negative")

    digit_count = len(fraction_digits)
    if digit_count == 0:
        return 0.0
    return int(fraction_digits) / (10**digit_count)
