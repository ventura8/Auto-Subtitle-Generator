"""Model wrappers and runtime model lifecycle helpers."""

import gc
import importlib
import logging
import os
from collections import namedtuple
from typing import Any

from .configuration import config
from .runtime.optional_imports import load_optional_torch
from .translators import nllb as nllb_backend
from .translators import translategemma as translategemma_backend

torch: Any | None = load_optional_torch()
LOGGER = logging.getLogger(__name__)

Segment = namedtuple("Segment", ["start", "end", "text"])


def _import_module(module_name):
    """Import module lazily by name to keep optional dependencies optional."""
    return importlib.import_module(module_name)


class SystemOptimizer:
    """Simple runtime optimizer with hardware facts and tunable config."""

    def __init__(self):
        self.gpu_name = "CPU"
        self.vram_gb = 0
        self.cpu_cores = os.cpu_count() or 1
        self.profile = "CPU_ONLY"
        self._default_config = {
            "whisper_beam": 5,
            "nllb_batch": 8,
            "translategemma_batch": 6,
            "translategemma_max_new_tokens": 160,
            "ffmpeg_threads": max(1, min(self.cpu_cores, 8)),
            "whisper_workers": 1,
            "whisper_beam_overridden": False,
        }
        self.config = dict(self._default_config)

    def reset(self):
        """Reset optimizer tunables to default values."""
        self.config = dict(self._default_config)

    def detect_hardware(self, verbose=False):
        """Populate hardware fields and keep profile-derived defaults stable."""
        _ = verbose
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            self.gpu_name = getattr(props, "name", "CUDA GPU")
            self.vram_gb = int(getattr(props, "total_memory", 0) / (1024**3))
            self.profile = _resolve_hardware_profile(self.vram_gb)
        else:
            self.gpu_name = "CPU"
            self.vram_gb = 0
            self.profile = "CPU_ONLY"

        self._apply_profile_tuning()

    def _apply_profile_tuning(self):
        """Apply profile-aware defaults for throughput-critical model settings."""
        if self.profile == "ULTRA":
            self.config["nllb_batch"] = 8
            self.config["translategemma_batch"] = 24
            self.config["translategemma_max_new_tokens"] = 192
            return

        if self.profile == "HIGH":
            self.config["nllb_batch"] = 8
            self.config["translategemma_batch"] = 8
            self.config["translategemma_max_new_tokens"] = 192
            return

        if self.profile == "MID":
            self.config["nllb_batch"] = 6
            self.config["translategemma_batch"] = 4
            self.config["translategemma_max_new_tokens"] = 160
            return

        # CPU-only: keep memory pressure controlled while preserving quality.
        self.config["nllb_batch"] = 2
        self.config["translategemma_batch"] = 1
        self.config["translategemma_max_new_tokens"] = 144

    def snapshot(self):
        """Return a lightweight dict snapshot of detected hardware state."""
        return {
            "gpu_name": self.gpu_name,
            "vram_gb": self.vram_gb,
            "cpu_cores": self.cpu_cores,
            "profile": self.profile,
        }


def _resolve_hardware_profile(vram_gb):
    """Resolve runtime hardware profile from detected GPU memory."""
    if vram_gb >= 24:
        return "ULTRA"
    if vram_gb >= 10:
        return "HIGH"
    return "MID"


class WhisperModel:
    """Thin Faster-Whisper wrapper used by transcription orchestration."""

    def __init__(self):
        faster_whisper = _import_module("faster_whisper")
        self._faster_whisper_model = faster_whisper.WhisperModel
        self._model, self._using_cpu = _load_whisper_model(self._faster_whisper_model)

    def transcribe(self, *args, **kwargs):
        """Delegate transcription to underlying Faster-Whisper model."""
        try:
            return self._model.transcribe(*args, **kwargs)
        except (RuntimeError, OSError, ValueError) as error:
            if self._using_cpu or not _is_cuda_runtime_missing_error(error):
                raise
            LOGGER.warning(
                "Faster-Whisper CUDA runtime failed during transcription (%s). Retrying on CPU int8.",
                error,
            )
            self._model = self._faster_whisper_model(
                config.WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
            self._using_cpu = True
            return self._model.transcribe(*args, **kwargs)

    def release(self):
        """Release wrapped model reference."""
        self._model = None


class SeparatorModel:
    """Audio separator wrapper used by transcription pipeline."""

    def __init__(self, output_dir=None):
        separator_module = _import_module("audio_separator.separator")
        separator = separator_module.Separator
        self._output_dir = output_dir
        # Ask audio-separator to generate only vocals to avoid creating instrumental stems.
        self._separator = separator(output_dir=output_dir, output_single_stem="Vocals")
        self._separator.load_model(model_filename=config.AUDIO_SEPARATOR_MODEL_ID)

    def separate(self, audio_input_path):
        """Run separator and return output stem paths."""
        return self._separator.separate(audio_input_path)

    def release(self):
        """Release wrapped separator reference."""
        self._separator = None


class ModelManager:
    """Lazy model manager preserving heavy model lifetimes between files."""

    def __init__(self):
        self._whisper = None
        self._nllb = None
        self._translategemma = None
        self._separator = None
        self._separator_output_dir = None

    def get_whisper(self):
        """Return lazily initialized Whisper wrapper."""
        if self._whisper is None:
            self._whisper = WhisperModel()
        return self._whisper

    def offload_whisper(self):
        """Release Whisper model and reclaim memory."""
        if self._whisper is not None:
            self._whisper.release()
        self._whisper = None
        _cleanup_torch_cache()

    def get_nllb(self):
        """Return lazily initialized NLLB translator wrapper."""
        if self._nllb is None:
            self._nllb = nllb_backend.NLLBTranslator()
        return self._nllb

    def offload_nllb(self):
        """Release NLLB model and reclaim memory."""
        if self._nllb is not None:
            self._nllb.release()
        self._nllb = None
        _cleanup_torch_cache()

    def get_translategemma(self):
        """Return TranslateGemma backend for worker dispatch."""
        if self._translategemma is None:
            runtime_settings = {
                "max_new_tokens": self._read_optimizer_int("translategemma_max_new_tokens", 160),
            }
            self._translategemma = translategemma_backend.TranslateGemmaTranslator(runtime_settings=runtime_settings)
        return self._translategemma

    def offload_translategemma(self):
        """Release TranslateGemma model and reclaim memory."""
        if self._translategemma is not None:
            self._translategemma.release()
        self._translategemma = None
        _cleanup_torch_cache()

    def _read_optimizer_int(self, key, default):
        """Read integer optimizer value with safe fallback."""
        value = OPTIMIZER.config.get(key, default)
        try:
            parsed = int(value)
            if parsed <= 0:
                return default
            return parsed
        except (TypeError, ValueError):
            return default

    def get_separator(self, output_dir=None):
        """Return lazily initialized separator wrapper."""
        if self._separator is None or self._separator_output_dir != output_dir:
            if self._separator is not None:
                self.offload_separator()
            self._separator = SeparatorModel(output_dir=output_dir)
            self._separator_output_dir = output_dir
        return self._separator

    def offload_separator(self):
        """Release separator model and reclaim memory."""
        if self._separator is not None:
            self._separator.release()
        self._separator = None
        self._separator_output_dir = None
        _cleanup_torch_cache()


OPTIMIZER = SystemOptimizer()


def _cleanup_torch_cache():
    """Run conservative garbage collection and optional CUDA cache cleanup."""
    gc.collect()
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_runtime_missing_error(error: Exception) -> bool:
    """Detect known CUDA runtime/DLL load failures from CTranslate2/Faster-Whisper."""
    message = str(error).lower()
    known_tokens = (
        "cublas64_12.dll",
        "cudnn64_9.dll",
        "cudart64_12",
        "cannot be loaded",
    )
    return any(token in message for token in known_tokens)


def _build_cpu_whisper_model(faster_whisper_model):
    """Build a CPU-backed Whisper model with deterministic int8 compute."""
    return faster_whisper_model(
        config.WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
    )


def _build_cuda_whisper_model(faster_whisper_model):
    """Build a CUDA-backed Whisper model with float16 compute."""
    return faster_whisper_model(
        config.WHISPER_MODEL_SIZE,
        device="cuda",
        compute_type="float16",
    )


def _should_try_cuda_whisper() -> bool:
    """Return True when torch CUDA runtime appears available."""
    return bool(torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available())


def _load_whisper_model(faster_whisper_model):
    """Load Faster-Whisper with CUDA first and deterministic CPU fallback."""
    if not _should_try_cuda_whisper():
        return _build_cpu_whisper_model(faster_whisper_model), True

    try:
        return _build_cuda_whisper_model(faster_whisper_model), False
    except (RuntimeError, OSError, ValueError) as error:
        if not _is_cuda_runtime_missing_error(error):
            raise
        LOGGER.warning(
            "Faster-Whisper CUDA runtime is unavailable (%s). Falling back to CPU int8. "
            "Run install_dependencies.ps1 to install required CUDA runtime DLLs.",
            error,
        )
        return _build_cpu_whisper_model(faster_whisper_model), True
