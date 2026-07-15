"""Progress bar display utilities."""

import math
import shutil
import sys


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
    parts = _build_progress_prefix_parts(timestamp_str, elapsed)

    eta_text = _format_eta_text(eta)
    if eta_text:
        parts.append(eta_text)

    speed_text = _format_speed_text(speed, speed_unit)
    if speed_text:
        parts.append(speed_text)

    if suffix:
        parts.append(suffix)
    return parts


def _build_progress_prefix_parts(timestamp_str, elapsed):
    """Build initial progress info parts from timestamp or elapsed value."""
    if timestamp_str:
        return [timestamp_str]
    if elapsed is not None:
        return [_format_time_component(elapsed)]
    return []


def _format_eta_text(eta):
    """Format ETA fragment for progress display when value is valid."""
    if eta is None:
        return None
    try:
        eta_value = float(eta)
    except (TypeError, ValueError):
        return None
    if eta_value <= 0:
        return None
    return f"ETA {_format_time_component(eta_value)}"


def _format_speed_text(speed, speed_unit):
    """Format speed fragment for progress display when value is valid."""
    if speed is None:
        return None
    try:
        return f"{float(speed):.2f}{speed_unit}"
    except (TypeError, ValueError):
        return None


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
    if not math.isfinite(normalized_iteration):
        normalized_iteration = 0.0
    normalized_iteration = min(max(normalized_iteration, 0.0), normalized_total)

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
