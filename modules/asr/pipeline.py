"""Batched NVIDIA ASR over VAD speech spans: decoding, output filters and cue building.

Spans are decoded longest first so a batch pads as little as possible; the cues
are put back in time order at the end. Engine differences stay in the wrappers:
a result carrying token ``stamps`` (Parakeet) is grouped word by word, one
without them (Canary) is split at punctuation and snapped to the pauses a fine
VAD pass finds inside the span. A ``logprob`` (Canary only) enables the
confidence filter.
"""

import functools
import time
from collections import namedtuple
from typing import Any

from modules.asr import common
from modules.asr.audio_source import internal_pauses
from modules.asr.cues import cues_from_text, cues_from_words, finalize_cues, words_from_stamps
from modules.asr.languages import fold_romanian_diacritics
from modules.runtime.progress import print_progress_bar

AsrRequest = namedtuple("AsrRequest", ["language", "batch_size", "limits"])
# dropped maps a filter name to the number of spans it removed.
AsrResult = namedtuple("AsrResult", ["cues", "speech_seconds", "decoded_seconds", "dropped"])

# Mean token log-probability floors (Canary); a text of a few words needs more confidence to stand.
LOGPROB_FLOOR = -1.5
SHORT_TEXT_LOGPROB_FLOOR = -1.0
SHORT_TEXT_WORDS = 3

# With at least this much detected speech, decoding under this share of it to text means the
# engine failed (wrong language token, NaNs), not that the audio is silent.
MIN_GUARDED_SPEECH_SECONDS = 30.0
MIN_DECODED_SPEECH_RATIO = 0.2


class AsrEngineFailed(RuntimeError):
    """The NVIDIA engine decoded too little of the detected speech to be trusted."""


def transcribe(model, source, spans, request) -> AsrResult:
    """Decode every speech span with ``model`` in length-sorted batches; cues come back in time order."""
    state: dict[str, Any] = {"cues": [], "dropped": {}, "decoded": 0.0, "done": 0.0, "started": time.monotonic()}
    speech_seconds = sum(end - start for start, end in spans)
    for batch in _length_batches(spans, request.batch_size):
        _decode_batch(model, source, [spans[index] for index in batch], request, state)
        _show_progress(state, speech_seconds)
    return AsrResult(finalize_cues(state["cues"]), speech_seconds, state["decoded"], state["dropped"])


def check_decoded_speech(result: AsrResult) -> None:
    """Raise ``AsrEngineFailed`` when plenty of speech was found but little of it decoded to text."""
    if result.speech_seconds < MIN_GUARDED_SPEECH_SECONDS:
        return
    if result.decoded_seconds / result.speech_seconds < MIN_DECODED_SPEECH_RATIO:
        raise AsrEngineFailed(f"only {result.decoded_seconds:.0f} s of {result.speech_seconds:.0f} s of detected speech decoded to text")


def _length_batches(spans, batch_size):
    """Group span indices into batches of ``batch_size``, longest spans first."""
    order = sorted(range(len(spans)), key=lambda index: spans[index][0] - spans[index][1])
    size = max(1, int(batch_size))
    return [order[first : first + size] for first in range(0, len(order), size)]


def _decode_clips(model, language, clips):
    """One ``transcribe_batch`` call; the language goes positionally (Parakeet names it ``_language``)."""
    return model.transcribe_batch(clips, language)


def _decode_batch(model, source, batch_spans, request, state):
    """Decode one batch of spans and add their cues, drop counts and decoded seconds to ``state``."""
    clips = [source.read(start, end) for start, end in batch_spans]
    outputs = common.generate_with_oom_bisection(functools.partial(_decode_clips, model, request.language), clips)
    for span, clip, output in zip(batch_spans, clips, outputs):
        span_cues = _span_cues(output, span, clip, request, state["dropped"])
        state["cues"].extend(span_cues)
        state["decoded"] += (span[1] - span[0]) if span_cues else 0.0
        state["done"] += span[1] - span[0]


def _span_cues(output, span, clip, request, dropped):
    """Filter one decoded span and turn what survives into cues."""
    text, problem = _filtered_text(output, span[1] - span[0])
    if problem:
        dropped[problem] = dropped.get(problem, 0) + 1
        return []
    fold = _folder(request.language)
    # Token stamps only spell the text out while no repeat was collapsed away.
    if output.stamps is not None and text == output.text.strip():
        return cues_from_words(_folded_words(output.stamps, span[0], fold), request.limits)
    return cues_from_text(fold(text), span, internal_pauses(clip, span[0]), request.limits)


def _folder(language):
    """Return the text fold for ``language``: Romanian cedilla to comma-below, else unchanged."""
    return fold_romanian_diacritics if language == "ro" else _unchanged


def _unchanged(text):
    """Identity fold for languages without a spelling fix-up."""
    return text


def _folded_words(stamps, offset, fold):
    """Absolute word timings from token stamps, each word passed through ``fold``."""
    return [(start, end, fold(word)) for start, end, word in words_from_stamps(stamps, offset)]


def _filtered_text(output, seconds):
    """Return ``(text, problem)``: the cleaned text, or the name of the filter that rejected it."""
    text = output.text.strip()
    problem = _text_problem(output, text, seconds)
    if problem:
        return "", problem
    text = common.collapse_repeats(text)
    return text, _confidence_problem(output.logprob, text)


def _text_problem(output, text, seconds):
    """Name the deterministic filter the text fails, or None."""
    if not text:
        return "empty"
    if output.degenerate:
        return "runaway"
    if common.zlib_ratio(text) > common.ZLIB_RATIO_LIMIT:
        return "repetitive"
    return "too fast" if _too_fast(text, seconds) else None


def _too_fast(text, seconds):
    """True for more characters per second than any real speech."""
    return seconds > 0 and len(text) / seconds > common.MAX_CHARS_PER_SECOND


def _confidence_problem(logprob, text):
    """Name the confidence filter the text fails (Canary only), or None."""
    if logprob is None:
        return None
    floor = SHORT_TEXT_LOGPROB_FLOOR if len(text.split()) <= SHORT_TEXT_WORDS else LOGPROB_FLOOR
    return "low confidence" if logprob < floor else None


def _show_progress(state, speech_seconds):
    """Redraw the ASR progress bar over seconds of detected speech."""
    elapsed = time.monotonic() - state["started"]
    print_progress_bar(
        state["done"],
        speech_seconds,
        prefix="  [ASR] Transcribing",
        elapsed=elapsed,
        speed=state["done"] / elapsed if elapsed > 0 else 0,
    )
