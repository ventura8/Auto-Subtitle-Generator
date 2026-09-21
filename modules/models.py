"""Model wrappers and runtime model lifecycle helpers."""

import gc
import importlib
import logging
import os
import sys
from collections import namedtuple
from typing import Any

from .configuration import config
from .runtime import vram_tuning
from .runtime.model_cache import is_corrupt_model_error as _is_corrupt_checkpoint_error
from .runtime.model_cache import purge_separator_checkpoint as _purge_cached_separator_checkpoint
from .runtime.model_cache import purge_whisper_model_cache as _purge_whisper_model_cache
from .runtime.optional_imports import is_cuda_usable, is_mps_available, load_optional_torch
from .translators import nllb as nllb_backend
from .translators import translategemma as translategemma_backend

torch: Any | None = load_optional_torch()
LOGGER = logging.getLogger(__name__)

Segment = namedtuple("Segment", ["start", "end", "text"])

# Apple Silicon unified memory is shared with the OS; treat only half as GPU memory.
UNIFIED_MEMORY_USABLE_FRACTION = 0.5


def _import_module(module_name):
    """Import module lazily by name to keep optional dependencies optional."""
    return importlib.import_module(module_name)


def _detect_system_memory_gb() -> int:
    """Detect total system memory in gigabytes using POSIX sysconf when available."""
    if not hasattr(os, "sysconf"):
        return 0
    try:
        return int(getattr(os, "sysconf")("SC_PAGE_SIZE") * getattr(os, "sysconf")("SC_PHYS_PAGES") / (1024**3))
    except (AttributeError, ValueError, OSError):
        return 0


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
            "nllb_batch_overridden": False,
            "max_vram_usage_gb": 0,
        }
        self.config = dict(self._default_config)

    def reset(self):
        """Reset optimizer tunables to default values."""
        self.config = dict(self._default_config)

    def _detect_mps_or_cpu(self):
        """Fallback detection for Apple Silicon MPS or generic CPU."""
        if is_mps_available(torch):
            usable_gb = _usable_unified_memory_gb(_detect_system_memory_gb())
            profile = _resolve_mps_profile(usable_gb) if usable_gb > 0 else "CPU_ONLY"
            return "Apple Silicon (MPS)", usable_gb, profile
        return "CPU", 0, "CPU_ONLY"

    def _detect_gpu_props(self):
        """Detect GPU properties from available acceleration backends."""
        if is_cuda_usable(torch):
            props = torch.cuda.get_device_properties(0)
            name = getattr(props, "name", "CUDA GPU")
            vram = int(getattr(props, "total_memory", 0) / (1024**3))
            return name, vram, _resolve_hardware_profile(vram)
        return self._detect_mps_or_cpu()

    def detect_hardware(self, verbose=False):
        """Populate hardware fields and keep profile-derived defaults stable."""
        _ = verbose
        self.gpu_name, self.vram_gb, self.profile = self._detect_gpu_props()
        self._reprofile_for_cap()
        self._apply_profile_tuning()

    def apply_vram_cap(self, cap_gb):
        """Plan against at most ``cap_gb`` of VRAM: re-derive the tier and its caps for a GPU."""
        self.config["max_vram_usage_gb"] = float(cap_gb)
        self._reprofile_for_cap()
        self._apply_profile_tuning()

    def _reprofile_for_cap(self):
        """Lower a GPU tier to match ``max_vram_usage_gb`` when a cap is set."""
        if self.profile == "CPU_ONLY" or self.effective_vram_gb() == self.vram_gb:
            return
        capped = _resolve_hardware_profile(self.effective_vram_gb())
        self.profile = "HIGH" if self.gpu_name.startswith("Apple") and capped == "ULTRA" else capped

    def _apply_profile_tuning(self):
        """Apply profile caps for throughput-critical settings.

        ``nllb_batch`` here is the profile's *maximum*; the translation worker
        refines it downward from the VRAM actually free after the model loads
        (``apply_dynamic_translation_batch``) unless the user pinned it.
        """
        nllb_batch, gemma_batch, gemma_tokens = _PROFILE_TUNING.get(self.profile, _PROFILE_TUNING["CPU_ONLY"])
        # The worker loads config (which may pin nllb_batch) before detecting hardware.
        if not self.config.get("nllb_batch_overridden"):
            self.config["nllb_batch"] = nllb_batch
        self.config["translategemma_batch"] = gemma_batch
        self.config["translategemma_max_new_tokens"] = gemma_tokens

    def effective_vram_gb(self):
        """VRAM the pipeline may plan against: detected, capped by ``performance.max_vram_usage_gb``."""
        cap = float(self.config.get("max_vram_usage_gb") or 0)
        return min(self.vram_gb, cap) if cap > 0 else self.vram_gb

    def snapshot(self):
        """Return a lightweight dict snapshot of detected hardware state."""
        return {
            "gpu_name": self.gpu_name,
            "vram_gb": self.vram_gb,
            "cpu_cores": self.cpu_cores,
            "profile": self.profile,
        }


# (nllb_batch cap, translategemma_batch, translategemma_max_new_tokens) per profile.
# NLLB throughput rose from 7.3 to 12.4 lines/s going from batch 8 to 16 on calibration. Pinned
# at 32 on a GPU shared with another process, the same translation took 23 min against 3.5 min
# at 8 or 16 (allocator retry churn), so 16 is the ceiling everywhere the memory allows and
# dynamic sizing trims it to the VRAM actually free.
_PROFILE_TUNING = {
    "ULTRA": (16, 24, 192),
    "HIGH": (16, 8, 192),
    "MID": (16, 4, 160),
    "LOW": (8, 1, 144),
    "CPU_ONLY": (2, 1, 144),
}


def _resolve_hardware_profile(vram_gb):
    """Resolve runtime hardware profile from detected GPU memory.

    Thresholds follow what fits: NLLB-200-3.3B fp16 (6.7 GB) needs a 12 GB
    card to leave headroom, the distilled 1.3B fits from 6 GB, and below that
    Faster-Whisper drops to int8 weights and NLLB to the distilled 600M.
    """
    if vram_gb >= 24:
        return "ULTRA"
    if vram_gb >= 12:
        return "HIGH"
    if vram_gb >= 6:
        return "MID"
    return "LOW"


def _usable_unified_memory_gb(total_memory_gb):
    """Return the share of Apple Silicon unified memory usable by the GPU.

    Unified memory is shared with the OS and every other process, so only a
    conservative fraction may be treated as GPU memory.
    """
    return int(total_memory_gb * UNIFIED_MEMORY_USABLE_FRACTION)


def _resolve_mps_profile(usable_memory_gb):
    """Resolve an MPS profile, capped below the dedicated-VRAM ULTRA tier."""
    profile = _resolve_hardware_profile(usable_memory_gb)
    return "HIGH" if profile == "ULTRA" else profile


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
        self._load_separator_model()

    def _load_separator_model(self):
        """Load separator model checkpoint with automatic corrupt cache cleanup."""
        try:
            self._separator.load_model(model_filename=config.AUDIO_SEPARATOR_MODEL_ID)
        except (RuntimeError, OSError, ValueError, KeyError) as error:
            if not _is_corrupt_checkpoint_error(error):
                raise
            LOGGER.warning("Audio separator checkpoint file appears corrupt (%s). Purging and retrying download...", error)
            _purge_cached_separator_checkpoint(config.AUDIO_SEPARATOR_MODEL_ID, getattr(self._separator, "model_file_dir", None))
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
            self._nllb = nllb_backend.NLLBTranslator(model_id=resolve_nllb_model_id())
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
    if is_cuda_usable(torch):
        torch.cuda.empty_cache()


def resolve_nllb_model_id():
    """Return the NLLB model to load: the configured id, or the largest that fits the card on ``auto``."""
    model_id, note = vram_tuning.select_nllb_model(config.NLLB_MODEL_ID, OPTIMIZER.effective_vram_gb())
    if note:
        LOGGER.log(logging.WARNING if config.NLLB_MODEL_ID != vram_tuning.AUTO else logging.INFO, "NLLB model %s (%s)", model_id, note)
    return model_id


def apply_dynamic_translation_batch(model_id, logger_func=None):
    """Size ``nllb_batch`` from the VRAM free after the translator loaded; returns the batch.

    Honours an explicit ``performance.nllb_batch`` and does nothing without a
    CUDA device, where the profile cap already applies.
    """
    cap = int(OPTIMIZER.config["nllb_batch"])
    if OPTIMIZER.config.get("nllb_batch_overridden") or not _should_try_cuda_whisper():
        return cap
    free_gb = _cuda_free_gb()
    if free_gb is None:
        return cap
    free_gb = min(free_gb, _budget_headroom_gb(model_id))
    batch = vram_tuning.dynamic_batch_size(free_gb, model_id, config.NLLB_NUM_BEAMS, cap)
    OPTIMIZER.config["nllb_batch"] = batch
    message = vram_tuning.describe_budget(OPTIMIZER.effective_vram_gb(), free_gb, model_id, batch, cap)
    (logger_func or LOGGER.info)(message)
    return batch


def _budget_headroom_gb(model_id):
    """VRAM left for activations inside ``max_vram_usage_gb`` once the model's weights are counted.

    Unbounded when no cap is set, so the real free memory decides.
    """
    if float(OPTIMIZER.config.get("max_vram_usage_gb") or 0) <= 0:
        return float("inf")
    weights = vram_tuning.NLLB_WEIGHT_GB.get(model_id, 0.0)
    return max(0.0, OPTIMIZER.effective_vram_gb() - weights)


def _cuda_free_gb():
    """Free VRAM on the active CUDA device in GB, or None when it cannot be read."""
    try:
        free_bytes, _total = torch.cuda.mem_get_info()
    except (RuntimeError, AttributeError, ValueError):
        return None
    return free_bytes / (1024**3)


def _is_cuda_runtime_missing_error(error: Exception) -> bool:
    """Detect known CUDA runtime/DLL/so load failures from CTranslate2/Faster-Whisper."""
    message = str(error).lower()
    known_tokens = (
        "cublas64_12",
        "cudnn",
        "cudart64_12",
        # CUDA 13.x Windows runtime names (substring match covers cublas64_13.dll,
        # cublas64_130_0.dll, cudart64_130_0, etc.)
        "cublas64_13",
        "cudart64_13",
        "libcudart.so",
        "libcublas",
        "libcudnn",
        "cuda driver version is insufficient",
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
    """Build a CUDA-backed Whisper model; fp16 with room to spare, int8 weights on small cards."""
    compute_type = vram_tuning.whisper_compute_type(OPTIMIZER.effective_vram_gb())
    LOGGER.info("Faster-Whisper CUDA compute type: %s (%s GB VRAM)", compute_type, OPTIMIZER.effective_vram_gb())
    return faster_whisper_model(
        config.WHISPER_MODEL_SIZE,
        device="cuda",
        compute_type=compute_type,
    )


def _should_try_cuda_whisper() -> bool:
    """Return True when torch CUDA runtime appears available."""
    return is_cuda_usable(torch)


def _cuda_runtime_installation_guidance():
    """Return platform-appropriate CUDA runtime installation guidance."""
    if sys.platform == "win32":
        return "Run install_dependencies.ps1 to install required CUDA runtime DLLs."
    return "Run ./install_dependencies.sh to install the required CUDA runtime libraries."


def _load_whisper_model(faster_whisper_model):
    """Load Faster-Whisper with CUDA first and deterministic CPU fallback, auto-recovering from corrupted downloads."""
    try:
        return _load_whisper_model_internal(faster_whisper_model)
    except (RuntimeError, OSError, ValueError) as error:
        if _is_corrupt_checkpoint_error(error):
            LOGGER.warning("Faster-Whisper model cache appears corrupt (%s). Purging cache and retrying download...", error)
            _purge_whisper_model_cache(config.WHISPER_MODEL_SIZE)
            return _load_whisper_model_internal(faster_whisper_model)
        raise


def _load_whisper_model_internal(faster_whisper_model):
    """Load Faster-Whisper model with GPU or CPU strategy."""
    if not _should_try_cuda_whisper():
        return _build_cpu_whisper_model(faster_whisper_model), True

    try:
        return _build_cuda_whisper_model(faster_whisper_model), False
    except (RuntimeError, OSError, ValueError) as error:
        if not _is_cuda_runtime_missing_error(error):
            raise
        LOGGER.warning(
            "Faster-Whisper CUDA runtime is unavailable (%s). Falling back to CPU int8. %s",
            error,
            _cuda_runtime_installation_guidance(),
        )
        return _build_cpu_whisper_model(faster_whisper_model), True
