"""Timing and scoring for ``tests.tools.asr_benchmark``: batched decoding, RTF, WER/CER and long-form timing.

Accuracy uses ``tests.tools.asr_metrics``: corpus WER/CER with diacritics kept
(cedilla folded to comma-below) and Open-ASR-normalised, the Romanian diacritic
error rate, and a paired cluster bootstrap against Whisper on the same corpus.
"""

import statistics
import time
from typing import Any

from tests.tools import asr_metrics
from tests.tools.asr_bench_data import SAMPLE_RATE

Timing = tuple[float, float]


def batches(utterances: list, size: int) -> list[list[int]]:
    """Indices in length-sorted batches (longest first), so padding stays small as in production."""
    order = sorted(range(len(utterances)), key=lambda index: -len(utterances[index].audio))
    return [order[start : start + size] for start in range(0, len(order), size)]


def _store(texts: list[str], indices: list[int], outputs: list[str]) -> None:
    """Put a batch's outputs back at their corpus positions."""
    for index, text in zip(indices, outputs):
        texts[index] = text


def decode_corpus(decoder: Any, utterances: list, size: int) -> tuple[list[str], list[Timing]]:
    """Decode every utterance; also return ``(audio seconds, wall seconds)`` per decoder call."""
    texts = [""] * len(utterances)
    timings = []
    for indices in batches(utterances, size):
        clips = [utterances[index].audio for index in indices]
        started = time.perf_counter()
        outputs = decoder(clips)
        timings.append((sum(len(clip) for clip in clips) / SAMPLE_RATE, time.perf_counter() - started))
        _store(texts, indices, outputs)
    return texts, timings


def audio_seconds(timings: list[Timing]) -> float:
    """Audio decoded across all calls."""
    return round(sum(timing[0] for timing in timings), 2)


def rtf(timings: list[Timing]) -> float | None:
    """Real-time factor without the first call, which carries one-off warm-up costs."""
    timed = timings[1:] or timings
    audio = sum(timing[0] for timing in timed)
    return sum(timing[1] for timing in timed) / audio if audio else None


def pairs_of(corpus: Any, texts: list[str]) -> list[tuple[str, str]]:
    """``(reference, hypothesis)`` per utterance; references are the raw transcriptions."""
    return [(utterance.text, text) for utterance, text in zip(corpus.utterances, texts)]


def scores(pairs: list[tuple[str, str]]) -> dict:
    """Corpus WER/CER under both normalisers plus the Romanian diacritic error rate."""
    keep, open_asr = asr_metrics.normalise_keep_diacritics, asr_metrics.normalise_open_asr
    return {
        "wer": asr_metrics.corpus_rate(pairs, keep, "word"),
        "wer_open_asr": asr_metrics.corpus_rate(pairs, open_asr, "word"),
        "cer": asr_metrics.corpus_rate(pairs, keep, "char"),
        "cer_open_asr": asr_metrics.corpus_rate(pairs, open_asr, "char"),
        "diacritic_error_rate": asr_metrics.diacritic_error_rate(pairs),
    }


def word_errors(pairs: list[tuple[str, str]]) -> list[tuple[int, int]]:
    """Per-utterance ``(word errors, reference words)`` with diacritics kept, for the paired bootstrap."""
    keep = asr_metrics.normalise_keep_diacritics
    return [asr_metrics.error_counts(keep(ref), keep(hyp), "word") for ref, hyp in pairs]


def _delta_ci(errors: list[tuple[int, int]], reference: list[tuple[int, int]], clusters: list) -> list[float]:
    """95 % CI of WER(cell) - WER(reference), resampling whole clusters."""
    err_a = [error[0] for error in errors]
    err_b = [error[0] for error in reference]
    words = [error[1] for error in reference]
    return list(asr_metrics.paired_cluster_bootstrap(err_a, err_b, words, clusters))


def _attach_delta(cell: dict, baseline: dict | None) -> None:
    """WER difference to Whisper on the same corpus with its CI; negative means better than Whisper."""
    if baseline is None or baseline is cell:
        return
    cell["delta_wer_vs_whisper"] = cell["wer"] - baseline["wer"]
    cell["delta_wer_vs_whisper_ci95"] = _delta_ci(cell["_errors"], baseline["_errors"], cell["_clusters"])


def add_deltas(cells: list[dict]) -> None:
    """Compare every NVIDIA cell with Whisper's cell on the same corpus, then drop the per-utterance vectors."""
    baselines = {cell["corpus"]: cell for cell in cells if cell["engine"] == "whisper"}
    for cell in cells:
        _attach_delta(cell, baselines.get(cell["corpus"]))
    for cell in cells:
        del cell["_errors"], cell["_clusters"]


def _onset_error(span: tuple, cues: list) -> float | None:
    """Distance from an utterance's start to the earliest cue overlapping it; None when no cue does (dropped)."""
    starts = [cue[0] for cue in cues if cue[0] < span[1] and cue[1] > span[0]]
    return abs(min(starts) - span[0]) if starts else None


def _median(values: list[float]) -> float | None:
    """Median, or None for no values."""
    return statistics.median(values) if values else None


def _matched_onsets(truth: list, cues: list) -> list[float]:
    """Onset errors of the utterances some cue overlaps."""
    return [onset for onset in (_onset_error(span, cues) for span in truth) if onset is not None]


def longform_metrics(truth: list, cues: list) -> dict:
    """Long-form WER, utterances with no overlapping cue, and the median cue-onset error."""
    pairs = [(" ".join(span[2] for span in truth), " ".join(cue[2] for cue in cues))]
    matched = _matched_onsets(truth, cues)
    return {
        **scores(pairs),
        "utterances": len(truth),
        "cues": len(cues),
        "dropped_spans": len(truth) - len(matched),
        "median_onset_error_s": _median(matched),
    }
