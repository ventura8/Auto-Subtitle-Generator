"""Streaming 16 kHz mono PCM access and block-wise Silero VAD for the NVIDIA ASR engines.

The NVIDIA engines decode one speech span at a time, so nothing here holds a
whole multi-hour input in RAM (AGENTS contract 7): 16 kHz mono WAVs are read by
seeking, short inputs are decoded in memory, and anything else is transcoded
once into the work directory and then read by seeking.
"""

import importlib
import itertools
import math
import os
from collections import namedtuple
from typing import Any

from modules.media import ffmpeg_utils
from modules.runtime.logging_utils import log
from modules.safe_io import discard_temp_path, promote_temp_path, reserve_temp_path

SAMPLE_RATE = 16000
BLOCK_SECONDS = 600.0
WHOLE_DECODE_MAX_SECONDS = 1800.0
TRANSCODE_TOLERANCE_SECONDS = 0.5
MERGE_GAP_SECONDS = 0.3
MIN_SPAN_SECONDS = 0.3
# Re-read this much audio before a block edge so a word cut short by the edge
# (too short for Silero on its own) is detected whole in the next block.
EDGE_LOOKBACK_SECONDS = 1.0
EDGE_TOLERANCE_SECONDS = 0.05
PAUSE_MIN_SILENCE_MS = 150
PAUSE_PAD_MS = 30

Span = tuple[float, float]
VadSettings = namedtuple("VadSettings", ["threshold", "min_speech_ms", "min_silence_ms", "speech_pad_ms", "max_speech_s"])


def asr_vad_settings(max_speech_s: float, min_silence_ms: int) -> VadSettings:
    """Return the VAD settings the NVIDIA engines segment with."""
    return VadSettings(threshold=0.5, min_speech_ms=250, min_silence_ms=min_silence_ms, speech_pad_ms=300, max_speech_s=max_speech_s)


def _soundfile() -> Any:
    """Return the lazily imported soundfile module."""
    return importlib.import_module("soundfile")


def _vad_module() -> Any:
    """Return the lazily imported faster-whisper Silero VAD module."""
    return importlib.import_module("faster_whisper.vad")


def _decode_audio(path: str) -> Any:
    """Decode ``path`` to a 16 kHz mono float32 array with faster-whisper."""
    return importlib.import_module("faster_whisper").decode_audio(path, sampling_rate=SAMPLE_RATE)


class PcmSource:
    """Random access by time to 16 kHz mono float32 PCM."""

    def __init__(self, total_frames: int):
        self._total_frames = int(total_frames)
        self.duration = self._total_frames / SAMPLE_RATE

    def read(self, start_s: float, end_s: float) -> Any:
        """Return the samples between two times, clamped to the source."""
        first = min(max(0, round(start_s * SAMPLE_RATE)), self._total_frames)
        last = min(max(first, round(end_s * SAMPLE_RATE)), self._total_frames)
        return self._frames(first, last)

    def close(self) -> None:
        """Release whatever the source holds."""

    def _frames(self, first: int, last: int) -> Any:
        """Return samples ``first`` to ``last``; implemented by each reader."""
        raise NotImplementedError


class _SoundFileSource(PcmSource):
    """Seek-and-read access to a 16 kHz mono file; RAM stays O(one read)."""

    def __init__(self, handle: Any):
        super().__init__(handle.frames)
        self._handle = handle

    def _frames(self, first: int, last: int) -> Any:
        """Seek to ``first`` and read up to ``last``."""
        self._handle.seek(first)
        return self._handle.read(last - first, dtype="float32", always_2d=False)

    def close(self) -> None:
        """Close the file handle."""
        self._handle.close()


class _ArraySource(PcmSource):
    """An input decoded whole into memory (only used up to ``WHOLE_DECODE_MAX_SECONDS``)."""

    def __init__(self, samples: Any):
        super().__init__(len(samples))
        self._samples = samples

    def _frames(self, first: int, last: int) -> Any:
        """Slice the decoded array."""
        return self._samples[first:last]

    def close(self) -> None:
        """Drop the decoded array."""
        self._samples = self._samples[:0]


def _probe(path: str) -> tuple[bool, float]:
    """Return ``(is 16 kHz mono, duration)``; FFprobe supplies the duration if libsndfile cannot read it."""
    try:
        info = _soundfile().info(path)
    except RuntimeError:
        return False, ffmpeg_utils.get_audio_duration(path)
    return info.samplerate == SAMPLE_RATE and info.channels == 1, float(info.duration)


def _open_soundfile(path: str) -> PcmSource:
    """Open a 16 kHz mono file for seek-and-read access."""
    return _SoundFileSource(_soundfile().SoundFile(path, mode="r"))


def _is_reusable_transcode(transcode_path: str, duration: float) -> bool:
    """True when an earlier run left a regular 16 kHz mono transcode of the right length."""
    if os.path.islink(transcode_path) or not os.path.isfile(transcode_path):
        return False
    is_mono_16k, transcoded = _probe(transcode_path)
    return is_mono_16k and abs(transcoded - duration) <= TRANSCODE_TOLERANCE_SECONDS


def _transcode_once(path: str, transcode_path: str, duration: float) -> str:
    """Transcode ``path`` to 16 kHz mono at ``transcode_path`` atomically, reusing a finished one."""
    if _is_reusable_transcode(transcode_path, duration):
        return transcode_path
    log(
        f"[ASR] {os.path.basename(path)} runs {duration / 60:.0f} min and is not 16 kHz mono; "
        "transcoding it once so it can be streamed instead of decoded whole.",
        "WARNING",
    )
    reservation = reserve_temp_path(transcode_path, scratch_dir=os.path.dirname(os.path.abspath(transcode_path)))
    try:
        ffmpeg_utils.write_audio_window(path, reservation.path, 0, duration, mono_16k=True)
        promote_temp_path(reservation, transcode_path)
    except BaseException:
        discard_temp_path(reservation)
        raise
    return transcode_path


def open_pcm_source(path: str, transcode_path: str) -> PcmSource:
    """Open ``path`` as a PCM source, picking the cheapest reader that bounds RAM."""
    is_mono_16k, duration = _probe(path)
    if is_mono_16k:
        return _open_soundfile(path)
    if duration <= WHOLE_DECODE_MAX_SECONDS:
        return _ArraySource(_decode_audio(path))
    return _open_soundfile(_transcode_once(path, transcode_path, duration))


def _vad_options(module: Any, vad: VadSettings) -> Any:
    """Build faster-whisper ``VadOptions`` from ``vad``."""
    return module.VadOptions(
        threshold=vad.threshold,
        min_speech_duration_ms=vad.min_speech_ms,
        min_silence_duration_ms=vad.min_silence_ms,
        speech_pad_ms=vad.speech_pad_ms,
        max_speech_duration_s=vad.max_speech_s,
    )


def _detect(module: Any, options: Any, audio: Any, offset: float) -> list[Span]:
    """Run Silero over ``audio`` and return its spans in absolute seconds."""
    stamps = module.get_speech_timestamps(audio, options)
    return [(offset + stamp["start"] / SAMPLE_RATE, offset + stamp["end"] / SAMPLE_RATE) for stamp in stamps]


def _carries(spans: list[Span], window: Span, block_seconds: float) -> bool:
    """True when the last span runs into the block edge and is short enough to re-read whole."""
    read_end = window[1]
    return bool(spans) and spans[-1][1] >= read_end - EDGE_TOLERANCE_SECONDS and read_end - spans[-1][0] < block_seconds


def _next_read_start(spans: list[Span], read_start: float, block_end: float) -> float:
    """Start the next block just after the last kept span, looking back over the edge."""
    last_end = spans[-1][1] if spans else read_start
    return max(last_end, block_end - EDGE_LOOKBACK_SECONDS)


def _block_spans(source: PcmSource, detector: tuple[Any, Any], window: Span, block_seconds: float) -> tuple[list[Span], float]:
    """Detect one block's spans; return the kept spans and where the next block starts reading."""
    module, options = detector
    read_start, block_end = window
    spans = _detect(module, options, source.read(read_start, block_end), read_start)
    if block_end >= source.duration:
        return spans, block_end
    if _carries(spans, window, block_seconds):
        return spans[:-1], spans[-1][0]
    return spans, _next_read_start(spans, read_start, block_end)


def _raw_speech_spans(source: PcmSource, vad: VadSettings, block_seconds: float) -> list[Span]:
    """Run Silero block by block, carrying a span that touches a block edge into the next block."""
    module = _vad_module()
    detector = (module, _vad_options(module, vad))
    spans: list[Span] = []
    read_start = 0.0
    block_end = 0.0
    while block_end < source.duration:
        block_end = min(block_end + block_seconds, source.duration)
        kept, read_start = _block_spans(source, detector, (read_start, block_end), block_seconds)
        spans.extend(kept)
    return spans


def _can_merge(previous: Span, span: Span, max_seconds: float) -> bool:
    """True when two spans are separated by a short gap and fit the cap together."""
    return span[0] - previous[1] < MERGE_GAP_SECONDS and max(previous[1], span[1]) - previous[0] <= max_seconds


def _merge_close(spans: list[Span], max_seconds: float) -> list[Span]:
    """Merge spans separated by less than ``MERGE_GAP_SECONDS`` while they stay within the cap."""
    merged: list[Span] = []
    for span in spans:
        if merged and _can_merge(merged[-1], span, max_seconds):
            merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
        else:
            merged.append(span)
    return merged


def _split_long(span: Span, max_seconds: float) -> list[Span]:
    """Split a span longer than the cap into equal pieces."""
    start, end = span
    pieces = max(1, math.ceil((end - start) / max_seconds))
    step = (end - start) / pieces
    edges = [start + index * step for index in range(pieces)] + [end]
    return list(itertools.pairwise(edges))


def _tidy_spans(spans: list[Span], max_seconds: float) -> list[Span]:
    """Merge close spans, drop slivers and split anything over the cap."""
    merged = _merge_close(sorted(spans), max_seconds)
    kept = [span for span in merged if span[1] - span[0] >= MIN_SPAN_SECONDS]
    return [piece for span in kept for piece in _split_long(span, max_seconds)]


def iter_speech_spans(source: PcmSource, vad: VadSettings, block_seconds: float = BLOCK_SECONDS) -> list[Span]:
    """Return the speech spans of ``source`` in absolute seconds, sorted, each within ``vad.max_speech_s``."""
    return _tidy_spans(_raw_speech_spans(source, vad, block_seconds), vad.max_speech_s)


def internal_pauses(audio: Any, offset: float) -> list[Span]:
    """Return the absolute silence gaps a fine VAD pass finds inside one span's audio."""
    module = _vad_module()
    options = module.VadOptions(threshold=0.5, min_silence_duration_ms=PAUSE_MIN_SILENCE_MS, speech_pad_ms=PAUSE_PAD_MS)
    parts = _detect(module, options, audio, offset)
    return [(left[1], right[0]) for left, right in itertools.pairwise(parts) if right[0] > left[1]]
