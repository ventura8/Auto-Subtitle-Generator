"""Subtitle cue building for the NVIDIA ASR engines.

Parakeet-TDT returns per-token times, so its words are grouped into cues
directly. Canary returns text only, so its cues are timed proportionally to
character counts and snapped to the silences found inside the span.
"""

import re
import textwrap
from collections import namedtuple

Limits = namedtuple("Limits", ["max_chars", "max_seconds", "pause_seconds", "min_seconds"])
DEFAULT_LIMITS = Limits(84, 7.0, 0.5, 0.8)
Cue = tuple[float, float, str]

# A proportional boundary moves to a pause midpoint only when one is this close.
SNAP_SECONDS = 1.0
# Lower bound for the time-driven wrap width, so sparse speech is not cut word by word.
MIN_PIECE_CHARS = 12

_SENTENCE_END = (".", "?", "!", "…")
_CLOSERS = "\"'”’»)]"
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!…])\s+|(?<=[.?!…][\"'”’»)\]])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])\s+")


def words_from_stamps(stamps: list[dict], offset: float) -> list[Cue]:
    """Merge Parakeet token chunks into words with absolute start and end times."""
    words: list[Cue] = []
    for stamp in stamps:
        _absorb_stamp(words, stamp)
    return [(offset + start, offset + end, text.strip()) for start, end, text in words if text.strip()]


def _absorb_stamp(words: list[Cue], stamp: dict) -> None:
    """Start a new word on a leading space, otherwise extend the last word."""
    piece = stamp.get("token")
    if not piece:
        return
    end = float(stamp["end"])
    if words and not piece.startswith(" "):
        last_start, last_end, last_text = words[-1]
        words[-1] = (last_start, max(last_end, end), last_text + piece)
        return
    words.append((float(stamp["start"]), end, piece.strip()))


def cues_from_words(words: list[Cue], limits: Limits = DEFAULT_LIMITS) -> list[Cue]:
    """Group timed words into cues at sentence ends, pauses and size limits."""
    cues: list[Cue] = []
    group: list[Cue] = []
    for index, word in enumerate(words):
        if _overflows(group, word, limits):
            group = _flush(cues, group)
        group.append(word)
        if _is_boundary(word, _next_word(words, index), limits):
            group = _flush(cues, group)
    _flush(cues, group)
    return cues


def _flush(cues: list[Cue], group: list[Cue]) -> list[Cue]:
    """Append the grouped words as one cue and return a fresh group."""
    if group:
        cues.append((group[0][0], group[-1][1], " ".join(word[2] for word in group)))
    return []


def _overflows(group: list[Cue], word: Cue, limits: Limits) -> bool:
    """Return True when adding the word would exceed the character or duration cap."""
    if not group:
        return False
    chars = sum(len(member[2]) + 1 for member in group) + len(word[2])
    return chars > limits.max_chars or word[1] - group[0][0] > limits.max_seconds


def _is_boundary(word: Cue, following: Cue | None, limits: Limits) -> bool:
    """Return True when a cue should end after this word."""
    if word[2].rstrip(_CLOSERS).endswith(_SENTENCE_END):
        return True
    return following is not None and following[0] - word[1] >= limits.pause_seconds


def _next_word(words: list[Cue], index: int) -> Cue | None:
    """Return the word after ``index`` or None at the end."""
    return words[index + 1] if index + 1 < len(words) else None


def cues_from_text(
    text: str,
    span: tuple[float, float],
    pauses: list[tuple[float, float]],
    limits: Limits = DEFAULT_LIMITS,
) -> list[Cue]:
    """Split untimed Canary text into cues timed across the span and snapped to pauses."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    pieces = _split_text(cleaned, _piece_width(cleaned, span, limits))
    edges = _boundaries(pieces, span, pauses, limits)
    return list(zip(edges, edges[1:], pieces))


def _piece_width(text: str, span: tuple[float, float], limits: Limits) -> int:
    """Return the character width that keeps each piece near ``max_seconds`` of speech."""
    duration = span[1] - span[0]
    if duration <= limits.max_seconds:
        return limits.max_chars
    by_time = int(len(text) * limits.max_seconds / duration)
    return min(limits.max_chars, max(MIN_PIECE_CHARS, by_time))


def _split_text(text: str, width: int) -> list[str]:
    """Split text at sentence ends, then clauses, then word wrap, into pieces of ``width``."""
    pieces: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        pieces.extend(_split_sentence(sentence, width))
    return pieces


def _split_sentence(sentence: str, width: int) -> list[str]:
    """Pack clauses into pieces of at most ``width`` and wrap any clause still too long."""
    if len(sentence) <= width:
        return [sentence]
    pieces: list[str] = []
    for packed in _pack(_CLAUSE_SPLIT.split(sentence), width):
        pieces.extend(_wrap(packed, width))
    return pieces


def _pack(parts: list[str], width: int) -> list[str]:
    """Join consecutive parts greedily while the result fits ``width``."""
    packed: list[str] = []
    for part in parts:
        if packed and len(packed[-1]) + 1 + len(part) <= width:
            packed[-1] = f"{packed[-1]} {part}"
        else:
            packed.append(part)
    return packed


def _wrap(piece: str, width: int) -> list[str]:
    """Word-wrap a piece longer than ``width``; a single long word stays whole."""
    if len(piece) <= width:
        return [piece]
    return textwrap.wrap(piece, width, break_long_words=False, break_on_hyphens=False)


def _boundaries(pieces: list[str], span: tuple[float, float], pauses: list[tuple[float, float]], limits: Limits) -> list[float]:
    """Return monotonic cue edges proportional to piece length, snapped to pauses."""
    start = span[0]
    end = max(span[0], span[1])
    duration = end - start
    total = sum(len(piece) for piece in pieces)
    min_len = min(limits.min_seconds, duration / len(pieces))
    edges = [start]
    consumed = 0
    for index, piece in enumerate(pieces[:-1], start=1):
        consumed += len(piece)
        estimate = _snap(start + duration * consumed / total, pauses)
        ceiling = end - (len(pieces) - index) * min_len
        edges.append(min(max(estimate, edges[-1] + min_len), ceiling))
    edges.append(end)
    return edges


def _snap(estimate: float, pauses: list[tuple[float, float]]) -> float:
    """Move an estimated boundary to the nearest pause midpoint within ``SNAP_SECONDS``."""
    midpoints = [(pause_start + pause_end) / 2 for pause_start, pause_end in pauses]
    if not midpoints:
        return estimate
    nearest = min(midpoints, key=lambda midpoint: abs(midpoint - estimate))
    return nearest if abs(nearest - estimate) <= SNAP_SECONDS else estimate


def finalize_cues(cues: list[Cue]) -> list[Cue]:
    """Strip and drop empty cues, sort them, and keep them non-overlapping with end > start in ms."""
    final: list[Cue] = []
    floor_ms = 0
    for start, end, text in sorted(cues):
        cleaned = text.strip()
        if not cleaned:
            continue
        start_ms = max(round(start * 1000), floor_ms)
        end_ms = max(round(end * 1000), start_ms + 1)
        final.append((start_ms / 1000, end_ms / 1000, cleaned))
        floor_ms = end_ms
    return final
