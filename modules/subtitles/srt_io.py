"""SRT file I/O and validation utilities."""

import importlib
import logging
import os

from .timestamp_utils import format_timestamp, parse_timestamp


def save_srt(segments, path):
    """Saves a list of Segment objects to an SRT file."""
    _write_srt_segments_atomically(segments, path)


def save_translated_srt(segments, translated_lines, path):
    """Saves translated segments to an SRT file with translations replacing original text."""
    translated_segments = _build_translated_segments(segments, translated_lines)

    _write_srt_segments_atomically(translated_segments, path)


def _write_srt_segments_atomically(segments, path):
    """Write SRT cues to a temp path and atomically promote to destination."""

    temp_path = f"{path}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as file_handle:
            for idx, segment in enumerate(segments, start=1):
                file_handle.write(f"{idx}\n")
                file_handle.write(f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n")
                file_handle.write(f"{segment.text}\n\n")
        os.replace(temp_path, path)
    except OSError:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def _build_translated_segments(segments, translated_lines):
    """Build translated segment objects while enforcing one-to-one alignment."""
    if len(segments) != len(translated_lines):
        raise ValueError("segments and translated_lines must have equal lengths")

    segment_cls = _get_segment_class()
    return [segment_cls(segments[i].start, segments[i].end, translated_lines[i]) for i in range(len(segments))]


def parse_srt(path):
    """Parses an SRT file back into a list of Segment objects."""
    is_valid, parsed_segments, rejection_reason = _validate_and_parse_srt(path)
    if is_valid:
        return parsed_segments

    if rejection_reason == "malformed":
        logging.getLogger("Antigravity").warning(
            "  [Guard] Rejected malformed SRT cue in: %s",
            os.path.basename(path),
        )
    else:
        logging.getLogger("Antigravity").warning(
            "  [Guard] Rejected corrupted SRT: %s",
            os.path.basename(path),
        )
    return []


def _parse_srt_chunks(chunks, segment_cls):
    """Parse all non-empty SRT chunks and return None on malformed cues."""
    parsed_segments = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        parsed_segment = _parse_srt_chunk(chunk, segment_cls)
        if parsed_segment is None:
            return None
        parsed_segments.append(parsed_segment)
    return parsed_segments


def validate_srt(path):
    """Checks for basic SRT markers to filter out obviously undefined files."""
    is_valid, _, _ = _validate_and_parse_srt(path)
    return is_valid


def _is_srt_too_small(path):
    """Return True when file is too small to contain valid SRT cues."""
    return os.path.getsize(path) < 10


def _is_valid_full_srt_content(path):
    """Validate preview markers then parse all cues from the full file."""
    preview = _read_srt_preview(path)
    if not _is_valid_srt_preview(preview):
        return False
    parsed_segments = _read_and_parse_full_srt(path)
    return parsed_segments is not None and len(parsed_segments) > 0


def _validate_and_parse_srt(path):
    """Return (is_valid, parsed_segments, rejection_reason) for one-pass SRT checks."""
    if not os.path.exists(path):
        return False, [], "corrupted"

    try:
        if _is_srt_too_small(path):
            return False, [], "corrupted"

        content = _read_srt_content(path)
        if not _should_parse_srt_content(content):
            return False, [], "corrupted"

        return _parse_validated_srt_content(content)
    except (OSError, UnicodeError):
        return False, [], "corrupted"


def _read_srt_content(path):
    """Read full SRT content with BOM-tolerant UTF-8 decoding."""
    with open(path, "r", encoding="utf-8-sig") as file_handle:
        return file_handle.read()


def _should_parse_srt_content(content):
    """Return True when SRT content passes structural and corruption pre-checks."""
    return _is_valid_srt_preview(content[:4096]) and not _contains_known_srt_corruption(content)


def _parse_validated_srt_content(content):
    """Parse SRT content known to pass preview checks into validation tuple."""
    parsed_result = _parse_srt_chunks(content.split("\n\n"), _get_segment_class())
    if parsed_result is None:
        return False, [], "malformed"
    if parsed_result:
        return True, parsed_result, None
    return False, [], "corrupted"


def _contains_known_srt_corruption(content):
    """Return True when any line pair matches explicit corruption patterns."""
    lines = content.splitlines()
    for idx, line in enumerate(lines):
        next_line = lines[idx + 1] if idx + 1 < len(lines) else None
        if _check_srt_corruption(line.strip(), next_line.strip() if next_line is not None else None):
            return True
    return False


def _read_and_parse_full_srt(path):
    """Read full SRT and parse every cue, returning None on malformed content."""
    with open(path, "r", encoding="utf-8-sig") as file_handle:
        full_content = file_handle.read()
    chunks = full_content.split("\n\n")
    return _parse_srt_chunks(chunks, _get_segment_class())


def _get_segment_class():
    """Return the Segment class without a static import to avoid cycles."""
    return importlib.import_module("modules.models").Segment


def _parse_srt_chunk(chunk, segment_cls):
    """Parse one SRT chunk into a segment instance when structurally valid."""
    lines = chunk.strip().split("\n")
    if len(lines) < 3 or not lines[0].strip().isdigit():
        return None

    parsed_range = _parse_srt_time_range(lines[1])
    if parsed_range is None:
        return None

    start, end = parsed_range
    text = "\n".join(lines[2:])
    return segment_cls(start, end, text)


def _parse_srt_time_range(time_range):
    """Parse SRT timestamp range and return (start, end) or None."""
    if " --> " not in time_range:
        return None
    try:
        start_str, end_str = time_range.split(" --> ", maxsplit=1)
        start = parse_timestamp(start_str, strict=True)
        end = parse_timestamp(end_str, strict=True)
    except (ValueError, IndexError):
        return None
    if end < start:
        return None
    return (start, end)


def _read_srt_preview(path):
    """Read first bytes of SRT for fast structural validation."""
    with open(path, "r", encoding="utf-8-sig") as file_handle:
        return file_handle.read(4096)


def _is_valid_srt_preview(content):
    """Return True when SRT preview contains expected basic structure."""
    stripped = content.strip()
    if not stripped:
        return False
    if not stripped[0].isdigit():
        return False
    return " --> " in stripped


def _check_srt_corruption(line, next_line=None):
    """Helper to check for specific SRT corruption patterns in a line."""
    if _is_invalid_index_followup(line, next_line):
        return True
    return _has_impossible_timestamp_hours(line)


def _is_invalid_index_followup(line, next_line):
    """Return True when an index line is not followed by a timestamp line."""
    return line.isdigit() and bool(next_line) and " --> " not in next_line


def _has_impossible_timestamp_hours(line):
    """Return True when timestamp line contains impossible hour digit width."""
    if " --> " not in line:
        return False
    parts = line.split(" --> ")
    for part in parts:
        if len(part.split(":")[0]) > 3:
            return True
    return False
