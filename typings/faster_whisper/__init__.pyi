from typing import Any, Iterable

__all__: list[str]

class Word:
    start: float
    end: float
    word: str
    probability: float

class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: list[Word] | None
    temperature: float

class TranscriptionInfo:
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: list[tuple[str, float]] | None
    transcription_options: Any
    vad_options: Any

class WhisperModel:
    def __init__(self, model_size_or_path: str, *args: Any, **kwargs: Any) -> None: ...
    def transcribe(self, *args: Any, **kwargs: Any) -> tuple[Iterable[Segment], TranscriptionInfo]: ...
