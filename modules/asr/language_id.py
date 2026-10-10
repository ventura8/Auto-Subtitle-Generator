"""Spread language-ID vote over the speech of a whole file.

faster-whisper's own ``detect_language`` locks onto the first confident window,
so an English intro wins for a Romanian film. Here Whisper scores several
windows spread over the cumulative speech time and the probabilities are
summed; the routing decision rests on the winner of that vote.
"""

import importlib
import math
from collections import namedtuple
from operator import itemgetter
from typing import Any

from modules.runtime.logging_utils import log

LID_WINDOWS = 8
LID_WINDOW_SECONDS = 30.0
LID_MIXED_SHARE = 0.6
_MIN_PIECE_SECONDS = 1e-6
_MIXED_REPORT_LANGUAGES = 3

Span = tuple[float, float]
LanguageVote = namedtuple("LanguageVote", ["language", "probability", "share", "distribution"])


def _numpy() -> Any:
    """Return the lazily imported numpy module."""
    return importlib.import_module("numpy")


def _anchors(total: float, count: int, length: float) -> list[float]:
    """Return window starts in speech time, spread evenly; never more windows than the speech fills."""
    windows = max(1, min(count, math.ceil(total / length)))
    if windows == 1:
        return [0.0]
    step = (total - length) / (windows - 1)
    return [index * step for index in range(windows)]


def _clip_piece(span: Span, cursor: float, window: Span) -> Span | None:
    """Return the part of ``span`` (starting at speech time ``cursor``) inside a speech-time window."""
    low = max(cursor, window[0])
    high = min(cursor + span[1] - span[0], window[1])
    if high - low <= _MIN_PIECE_SECONDS:
        return None
    return span[0] + low - cursor, span[0] + high - cursor


def _window_pieces(spans: list[Span], anchor: float, length: float) -> list[Span]:
    """Return the consecutive span pieces covering ``length`` s of speech from ``anchor``."""
    pieces = []
    cursor = 0.0
    for span in spans:
        piece = _clip_piece(span, cursor, (anchor, anchor + length))
        cursor += span[1] - span[0]
        if piece is not None:
            pieces.append(piece)
    return pieces


def pick_lid_windows(spans: list[Span], count: int = LID_WINDOWS, length: float = LID_WINDOW_SECONDS) -> list[list[Span]]:
    """Return up to ``count`` windows of at most ``length`` s of speech, anchored evenly across the speech."""
    total = sum(end - start for start, end in spans)
    if total <= 0:
        return []
    return [_window_pieces(spans, anchor, length) for anchor in _anchors(total, count, length)]


def _sum_probabilities(windows: list[list[tuple[str, float]]]) -> dict[str, float]:
    """Sum each language's probability over the windows."""
    totals: dict[str, float] = {}
    for probabilities in windows:
        for code, probability in probabilities:
            totals[code] = totals.get(code, 0.0) + probability
    return totals


def _top_language(probabilities: list[tuple[str, float]]) -> str:
    """Return the most probable language of one window."""
    return max(probabilities, key=itemgetter(1))[0]


def _distribution(windows: list[list[tuple[str, float]]]) -> dict[str, float]:
    """Return each language's mean probability over the windows, most probable first."""
    ranked = sorted(_sum_probabilities(windows).items(), key=itemgetter(1), reverse=True)
    return {code: total / len(windows) for code, total in ranked}


def _share(windows: list[list[tuple[str, float]]], winner: str) -> float:
    """Return the fraction of windows whose most probable language is ``winner``."""
    return [_top_language(probabilities) for probabilities in windows].count(winner) / len(windows)


def vote_language(per_window: list[list[tuple[str, float]]]) -> LanguageVote | None:
    """Sum the per-window probabilities; ``share`` is the fraction of windows whose top language won."""
    windows = [probabilities for probabilities in per_window if probabilities]
    if not windows:
        return None
    distribution = _distribution(windows)
    winner = next(iter(distribution))
    return LanguageVote(winner, distribution[winner], _share(windows, winner), distribution)


def _window_probabilities(whisper_model: Any, source: Any, pieces: list[Span]) -> list[tuple[str, float]]:
    """Score one window with Whisper; a window it cannot score counts as no vote."""
    audio = _numpy().concatenate([source.read(start, end) for start, end in pieces])
    try:
        _language, _probability, probabilities = whisper_model.detect_language(audio=audio)
    except ValueError as error:
        log(f"[ASR] Language-ID window skipped: {error}", "DEBUG")
        return []
    return list(probabilities)


def _warn_mixed(vote: LanguageVote) -> None:
    """Log that no language dominates the vote."""
    leaders = list(vote.distribution.items())[:_MIXED_REPORT_LANGUAGES]
    summary = ", ".join(f"{code} {probability:.0%}" for code, probability in leaders)
    log(
        f"[ASR] Mixed-language audio ({summary}): '{vote.language}' leads only {vote.share:.0%} of the "
        f"language-ID windows; routing on '{vote.language}'.",
        "WARNING",
    )


def detect_language(whisper_model: Any, source: Any, spans: list[Span]) -> LanguageVote | None:
    """Vote the file's language over windows spread across its speech; ``None`` when nothing could be scored."""
    windows = pick_lid_windows(spans)
    vote = vote_language([_window_probabilities(whisper_model, source, pieces) for pieces in windows])
    if vote is not None and vote.share < LID_MIXED_SHARE:
        _warn_mixed(vote)
    return vote
