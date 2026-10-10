"""Engine calls and GPU measurement for ``tests.tools.asr_benchmark``, all through the production code.

Utterance-level decoding calls each engine directly on short clips: faster-whisper
through ``modules.models.WhisperModel`` (VAD off, production prompt and options)
and the NVIDIA wrappers' ``transcribe_batch`` with the production OOM bisection.
Whole-file decoding goes end to end: Whisper through the production transcribe
call (VAD, prompt, beam); Canary/Parakeet through the streaming PCM source,
block VAD, ``modules.asr.pipeline`` and its decoded-speech guard; then the shared
known-phrase hallucination filter for every engine. Private ``transcription``
helpers are used on purpose, so the benchmark cannot drift from production.
"""

import contextlib
import functools
import os
import shutil
import subprocess
import threading
import time
from collections import namedtuple
from collections.abc import Callable
from typing import Any

from modules.asr import audio_source, common
from modules.asr import pipeline as asr_pipeline
from modules.asr.cues import DEFAULT_LIMITS
from modules.asr.languages import NVIDIA_EU25
from modules.configuration import config
from modules.models import Segment, apply_dynamic_asr_batch
from modules.pipeline import transcription
from modules.runtime.optional_imports import is_cuda_usable, load_optional_torch
from tests.tools.asr_bench_data import SAMPLE_RATE, noise, transcode_path

# Typed Any (not Any | None): every CUDA access below is guarded by is_cuda_usable().
torch: Any = load_optional_torch()

ENGINES = ("whisper", "canary", "parakeet")
GIB = 1024**3
# Utterance-level Whisper decoding; whole-file runs use the production beam from the optimizer profile.
WHISPER_UTTERANCE_BEAM = 5
WHISPER_OPTIONS = {"condition_on_previous_text": True, "no_speech_threshold": 0.6}
WARMUP_SECONDS = 1.0
SMI_INTERVAL_SECONDS = 0.5
SMI_TIMEOUT_SECONDS = 10

# One whole-file run: ``batch_size`` clips per NVIDIA generate call, ``max_segment`` the VAD span cap.
Job = namedtuple("Job", ["engine", "language", "max_segment", "batch_size"])


def _output_lines(result: subprocess.CompletedProcess) -> list[str]:
    """Non-empty stdout lines of a successful command."""
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _smi_lines(smi: str | None, *arguments: str) -> list[str]:
    """Output lines of one nvidia-smi query, or [] when it is missing or fails."""
    if not smi:
        return []
    try:
        return _output_lines(subprocess.run([smi, *arguments], capture_output=True, text=True, timeout=SMI_TIMEOUT_SECONDS, check=False))
    except (OSError, subprocess.TimeoutExpired):
        return []


def _to_mib(lines: list[str]) -> int | None:
    """First numeric value of a ``memory.used`` query."""
    try:
        return int(float(lines[0]))
    except (IndexError, ValueError):
        return None


def _row_mib(row: str) -> int:
    """``used_memory`` of one compute-app row (``pid, name, 3954 MiB``); 0 when the driver reports N/A."""
    try:
        return int(row.rsplit(",", 1)[1].split()[0])
    except (IndexError, ValueError):
        return 0


def _max(current: int | None, value: int | None) -> int | None:
    """Running maximum that ignores missing readings."""
    if value is None:
        return current
    return max(value, current or 0)


class GpuSampler:
    """Poll nvidia-smi in a background thread for this process's and the device's peak GPU memory.

    CTranslate2 (faster-whisper) allocates outside torch, so ``max_memory_allocated``
    misses Whisper. The per-process ``--query-compute-apps`` reading covers every
    engine (torch's cache included) and ignores other processes on the card; the
    device-wide reading is kept for drivers that report per-process memory as N/A.
    """

    def __init__(self, evidence: bool):
        self.evidence = evidence
        self.readings: dict[str, int | None] = {"baseline": None, "device_peak": None, "process_peak": None}
        self.apps: list[str] = []
        self._smi = shutil.which("nvidia-smi")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuSampler":
        if self._smi:
            self.readings["baseline"] = self._used_mib()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._sample()

    def fields(self) -> dict:
        """Memory fields of one result row (MiB from nvidia-smi, GiB from torch)."""
        return {
            "torch_peak_gib": torch_peak_gib(),
            "process_peak_mib": self.readings["process_peak"],
            "gpu_peak_mib": self.readings["device_peak"],
            "gpu_baseline_mib": self.readings["baseline"],
            "compute_apps": self.apps,
        }

    def _poll(self) -> None:
        """Sample until the context exits."""
        while not self._stop.wait(SMI_INTERVAL_SECONDS):
            self._sample()

    def _sample(self) -> None:
        """Record one device and one per-process reading; keep this process's rows as device evidence when asked."""
        rows = self._own_apps()
        self.readings["device_peak"] = _max(self.readings["device_peak"], self._used_mib())
        self.readings["process_peak"] = _max(self.readings["process_peak"], sum(_row_mib(row) for row in rows) if rows else None)
        if self.evidence:
            self.apps = rows or self.apps

    def _used_mib(self) -> int | None:
        """Device 0 memory in use, all processes included."""
        return _to_mib(_smi_lines(self._smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"))

    def _own_apps(self) -> list[str]:
        """``pid, name, used_memory`` rows of this process: proof the engine ran on the GPU."""
        lines = _smi_lines(self._smi, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader")
        return [line for line in lines if line.split(",")[0].strip() == str(os.getpid())]


def reset_torch_peak() -> None:
    """Start a fresh ``max_memory_allocated`` window."""
    if is_cuda_usable(torch):
        torch.cuda.reset_peak_memory_stats()


def torch_peak_gib() -> float | None:
    """Peak torch allocation since the last reset (Canary/Parakeet only; CTranslate2 is invisible here)."""
    if not is_cuda_usable(torch):
        return None
    return round(torch.cuda.max_memory_allocated() / GIB, 3)


def _inference_mode() -> Any:
    """``torch.inference_mode()`` when torch is importable, else a no-op context."""
    return contextlib.nullcontext() if torch is None else torch.inference_mode()


def _whisper_utterance(model: Any, clip: Any, language: str) -> str:
    """One short clip through faster-whisper with the production prompt and options, VAD off (the clip is one utterance)."""
    segments, _info = model.transcribe(
        clip,
        language=language,
        beam_size=WHISPER_UTTERANCE_BEAM,
        initial_prompt=transcription._transcription_options(language, None)["prompt"],
        vad_filter=False,
        **WHISPER_OPTIONS,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def _whisper_batch(manager: Any, language: str, clips: list) -> list[str]:
    """Whisper has no batch API in the pipeline; clips are decoded one by one."""
    model = manager.get_whisper()
    return [_whisper_utterance(model, clip, language) for clip in clips]


def _decode_batch(model: Any, language: str, clips: list) -> list:
    """Positional ``transcribe_batch`` call for ``functools.partial`` (no lambda captures a loop variable)."""
    return model.transcribe_batch(clips, language)


def _nvidia_batch(engine: str, manager: Any, language: str, clips: list) -> list[str]:
    """Canary/Parakeet wrapper output for one batch, with the production OOM bisection."""
    model = manager.get_asr(engine)
    decoded = common.generate_with_oom_bisection(functools.partial(_decode_batch, model, language), list(clips))
    return [clip.text for clip in decoded]


UTTERANCE_DECODERS: dict[str, Callable[..., list[str]]] = {
    "whisper": _whisper_batch,
    "canary": functools.partial(_nvidia_batch, "canary"),
    "parakeet": functools.partial(_nvidia_batch, "parakeet"),
}


def _canary_generate(model: Any, beams: int, language: str, clips: list) -> list[str]:
    """Canary with ``num_beams``; the wrapper is greedy-only, so this drives its loaded processor and network."""
    processor, network, choice = (getattr(model, name) for name in ("_processor", "_model", "_choice"))
    inputs = processor.apply_transcription_request(audio=list(clips), source_language=language, punctuation=True)
    prompt_len = int(inputs["decoder_input_ids"].shape[1])
    positions = int(network.config.decoder_config.max_position_embeddings)
    budget = common.canary_token_budget(max(len(clip) for clip in clips) / SAMPLE_RATE, prompt_len, positions)
    inputs = inputs.to(choice.device, dtype=choice.dtype)
    with _inference_mode():
        output = network.generate(**inputs, max_new_tokens=budget, num_beams=beams, return_dict_in_generate=True)
    generated = [row[prompt_len:] for row in output.sequences.cpu().tolist()]
    return [text.strip() for text in processor.batch_decode(generated, skip_special_tokens=True)]


def canary_beam_batch(beams: int, manager: Any, language: str, clips: list) -> list[str]:
    """Beam-search Canary for the beam sweep, with the production OOM bisection."""
    model = manager.get_asr("canary")
    return common.generate_with_oom_bisection(functools.partial(_canary_generate, model, beams, language), list(clips))


def release_models(manager: Any) -> None:
    """Free every ASR model between engines, as the pipeline does between stages."""
    manager.offload_whisper()
    manager.offload_asr()


def prepare_engine(manager: Any, engine: str) -> None:
    """Load the engine and run one throw-away decode, so timings exclude the load and the first-call warm-up."""
    UTTERANCE_DECODERS[engine](manager, "en", [noise(WARMUP_SECONDS)])


def supports(engine: str, language: str) -> bool:
    """The NVIDIA engines only know the 25 European languages."""
    return engine == "whisper" or language in NVIDIA_EU25


def batch_size(manager: Any, engine: str, max_segment: float, pinned: int | None) -> int:
    """``--batch-size`` when given, else the production batch sized from the VRAM left after loading; Whisper runs one clip."""
    if engine == "whisper":
        return 1
    return pinned or apply_dynamic_asr_batch(manager.get_asr(engine), max_segment)


def _whisper_file(manager: Any, job: Job, path: str) -> tuple[list, dict]:
    """Whisper on a whole file through the production call: VAD, prompt, beam and thresholds."""
    options = transcription._transcription_options(job.language, None)
    segments, _info = transcription._run_whisper_transcribe_call(manager.get_whisper(), path, transcription._whisper_vad_params(), options)
    return [(segment.start, segment.end, segment.text.strip()) for segment in segments], {"dropped": {}}


def _engine_failure(result: Any) -> str | None:
    """The decoded-speech guard's verdict; production would redo such a file with Whisper."""
    try:
        asr_pipeline.check_decoded_speech(result)
    except asr_pipeline.AsrEngineFailed as error:
        return str(error)
    return None


def _nvidia_file(manager: Any, job: Job, path: str) -> tuple[list, dict]:
    """Canary/Parakeet on a whole file: streaming source, block VAD, then the ASR pipeline and its filters."""
    model = manager.get_asr(job.engine)
    source = audio_source.open_pcm_source(path, transcode_path(path))
    try:
        spans = audio_source.iter_speech_spans(source, audio_source.asr_vad_settings(job.max_segment, config.VAD_MIN_SILENCE_MS))
        request = asr_pipeline.AsrRequest(job.language, job.batch_size, DEFAULT_LIMITS)
        result = asr_pipeline.transcribe(model, source, spans, request)
    finally:
        source.close()
    info = {"dropped": dict(result.dropped), "speech_seconds": result.speech_seconds, "decoded_seconds": result.decoded_seconds}
    return list(result.cues), {**info, "engine_failed": _engine_failure(result)}


FILE_RUNNERS: dict[str, Callable[..., tuple[list, dict]]] = {"whisper": _whisper_file, "canary": _nvidia_file, "parakeet": _nvidia_file}


def transcribe_file(manager: Any, job: Job, path: str) -> tuple[list, dict, float]:
    """Cues after the shared hallucination-phrase filter, run details (filter drops etc.) and wall seconds."""
    started = time.perf_counter()
    cues, info = FILE_RUNNERS[job.engine](manager, job, path)
    kept, phrase_drops = transcription._filter_hallucinations([Segment(*cue) for cue in cues], config.HALLUCINATION_PHRASES)
    elapsed = time.perf_counter() - started
    info["dropped"]["known_phrase"] = phrase_drops
    return [(segment.start, segment.end, segment.text) for segment in kept], info, elapsed
