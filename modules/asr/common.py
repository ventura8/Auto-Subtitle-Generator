"""Shared device, loading, OOM and degeneracy helpers for the NVIDIA ASR engines."""

import gc
import math
import re
import types
import zlib
from collections import namedtuple
from typing import Any

from modules.runtime.logging_utils import log
from modules.runtime.optional_imports import is_cuda_usable, load_optional_torch
from modules.translators.common import load_with_cache_recovery

torch: Any | None = load_optional_torch()

DeviceChoice = namedtuple("DeviceChoice", ["device", "dtype"])
DecodedClip = namedtuple("DecodedClip", ["text", "stamps", "logprob", "degenerate"])

# Generated-token budget for Canary: a placeholder until calibrated on FLEURS (2x the 99.9th percentile).
CANARY_TOKENS_PER_SECOND = 12.0
TOKEN_MARGIN = 16

# Whisper's compression-ratio threshold, and a speaking rate no real speech reaches.
ZLIB_RATIO_LIMIT = 2.4
MAX_CHARS_PER_SECOND = 25.0

_REPEAT_PATTERN = re.compile(r"(\b.+?\b)(?:\s*\1){3,}")


def _warn_cache(message, *args):
    """Route ``load_with_cache_recovery``'s %-style corruption warning to the pipeline log."""
    log(message % args, "WARNING")


# load_with_cache_recovery expects a logging.Logger; LOGGER output never reaches the log file, so adapt it.
_CACHE_LOGGER = types.SimpleNamespace(warning=_warn_cache)


def _cpu_choice(torch_module) -> DeviceChoice:
    """CPU placement in float32 (``None`` dtype when torch is missing)."""
    return DeviceChoice("cpu", getattr(torch_module, "float32", None))


def select_device(torch_module) -> DeviceChoice:
    """Pick ``cuda:0`` in bf16 (native only) or float32, else the CPU in float32; MPS is not used."""
    if not is_cuda_usable(torch_module):
        return _cpu_choice(torch_module)
    # Emulated bf16 on pre-Ampere cards is slow, so only native support counts.
    native_bf16 = torch_module.cuda.is_bf16_supported(including_emulation=False)
    return DeviceChoice("cuda:0", torch_module.bfloat16 if native_bf16 else torch_module.float32)


def _load_on(loader, model_id: str, revision: str | None, choice: DeviceChoice, label: str):
    """Load one checkpoint on an explicit device and dtype with corrupt-cache recovery."""
    kwargs = {"revision": revision, "dtype": choice.dtype, "device_map": choice.device}
    return load_with_cache_recovery(loader, model_id, kwargs, _CACHE_LOGGER, label)


def load_pretrained(loader, model_id: str, revision: str | None, label: str) -> tuple[object, DeviceChoice]:
    """Load a model on the best device; a CUDA OOM falls back to the CPU in float32 with a WARNING."""
    choice = select_device(torch)
    try:
        return _load_on(loader, model_id, revision, choice, label), choice
    except RuntimeError as error:
        if choice.device == "cpu" or not is_cuda_oom(error):
            raise
        log(f"  [ASR] {label} does not fit on {choice.device} ({error}); loading it on the CPU in float32.", "WARNING")
    # Retry outside the except block so the traceback no longer pins the failed allocation.
    release_cuda_cache()
    cpu = _cpu_choice(torch)
    return _load_on(loader, model_id, revision, cpu, label), cpu


def flatten_lstm_weights(model) -> None:
    """Compact recurrent weights (Parakeet's LSTM) into one chunk after loading or moving the model."""
    for module in model.modules():
        flatten = getattr(module, "flatten_parameters", None)
        if callable(flatten):
            flatten()


def is_cuda_oom(error: BaseException) -> bool:
    """Return True when the error text reports an out-of-memory failure."""
    return "out of memory" in str(error).lower()


def release_cuda_cache() -> None:
    """Collect garbage and return torch's cached CUDA blocks to the driver."""
    gc.collect()
    if torch is None or not is_cuda_usable(torch):
        return
    try:
        torch.cuda.empty_cache()
    except (RuntimeError, OSError, ValueError) as cleanup_err:
        log(f"  [ASR] CUDA cache cleanup warning: {cleanup_err}", "WARNING")


def generate_with_oom_bisection(generate, clips: list) -> list:
    """Run ``generate(clips)``, halving the batch recursively on CUDA OOM; output order matches ``clips``."""
    try:
        return list(generate(clips))
    except RuntimeError as error:
        if len(clips) <= 1 or not is_cuda_oom(error):
            raise
    release_cuda_cache()
    middle = len(clips) // 2
    log(f"    [ASR OOM] Reducing batch {len(clips)} -> {middle} + {len(clips) - middle} and retrying.", "WARNING")
    return generate_with_oom_bisection(generate, clips[:middle]) + generate_with_oom_bisection(generate, clips[middle:])


def canary_token_budget(seconds: float, prompt_len: int, max_positions: int = 1024) -> int:
    """``max_new_tokens`` for Canary: scaled with clip length, never past the decoder's positional table."""
    scaled = math.ceil(seconds * CANARY_TOKENS_PER_SECOND) + TOKEN_MARGIN
    return max(1, min(scaled, max_positions - prompt_len))


def parakeet_step_budget(valid_frames: int, max_symbols: int) -> int:
    """``max_new_tokens`` for Parakeet: steps include blanks, so allow ``max_symbols`` per encoder frame."""
    return max_symbols * valid_frames + 1


def zlib_ratio(text: str) -> float:
    """Compression ratio of the UTF-8 text; repetitive output compresses far better than speech."""
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


def collapse_repeats(text: str) -> str:
    """Collapse a phrase repeated four or more times in a row to a single copy."""
    return _REPEAT_PATTERN.sub(r"\1", text)


def is_degenerate(text: str, seconds: float) -> bool:
    """Return True for empty, highly repetitive or impossibly fast output."""
    stripped = text.strip()
    if not stripped:
        return True
    too_fast = seconds > 0 and len(stripped) / seconds > MAX_CHARS_PER_SECOND
    return too_fast or zlib_ratio(stripped) > ZLIB_RATIO_LIMIT
