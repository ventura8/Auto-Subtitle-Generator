"""NLLB translation backend implementation."""

import logging
from typing import Any

from modules.configuration import config
from modules.runtime.optional_imports import is_mps_available, load_optional_torch
from modules.translators import common
from modules.translators.common import add_device_load_kwargs, import_transformers_module, resolve_device_map

torch: Any | None = load_optional_torch()
LOGGER = logging.getLogger(__name__)


class NLLBTranslator:
    """NLLB translation wrapper with local-cache fallback on load errors."""

    def __init__(self):
        transformers = _import_transformers_module()
        auto_model_for_seq2seq_lm = transformers.AutoModelForSeq2SeqLM
        nllb_tokenizer = transformers.NllbTokenizer
        self._tokenizer = _load_nllb_tokenizer(nllb_tokenizer)
        self._model = _load_nllb_model(auto_model_for_seq2seq_lm)
        self._device = _resolve_nllb_execution_device(self._model)

    def translate(self, texts, src_code, tgt_code):
        """Translate a list of texts from source code to target code."""
        if not texts:
            return []

        self._tokenizer.src_lang = src_code
        encoded = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        encoded = _move_nllb_encoded_to_device(encoded, getattr(self, "_device", None))

        forced_id = self._tokenizer.convert_tokens_to_ids(tgt_code)
        generated = self._model.generate(
            **encoded,
            forced_bos_token_id=forced_id,
            num_beams=config.NLLB_NUM_BEAMS,
            length_penalty=config.NLLB_LENGTH_PENALTY,
            repetition_penalty=config.NLLB_REPETITION_PENALTY,
            no_repeat_ngram_size=config.NLLB_NO_REPEAT_NGRAM_SIZE,
        )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)

    def release(self):
        """Release wrapped model/tokenizer references."""
        self._model = None
        self._tokenizer = None
        self._device = None


def _move_nllb_encoded_to_device(encoded, device):
    """Move NLLB inputs to the configured device or an available CUDA device."""
    target_device = _resolve_nllb_input_device(device)
    if target_device is None:
        return encoded
    return {key: value.to(target_device) for key, value in encoded.items()}


def _resolve_nllb_input_device(device):
    """Resolve the preferred NLLB input device."""
    if device is not None:
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if is_mps_available(torch):
        return "mps"
    return None


def _import_transformers_module():
    """Import transformers lazily to preserve optional dependency behavior."""
    return import_transformers_module()


def _resolve_device_map():
    """Resolve explicit device mapping for transformer loading."""
    return resolve_device_map()


def _build_nllb_model_kwargs():
    """Build model load kwargs for NLLB with explicit device map and float16 precision."""
    return add_device_load_kwargs({}, _resolve_device_map(), torch)


def _resolve_nllb_execution_device(model):
    """Resolve the execution device for NLLB inputs."""
    if model is None:
        return None
    device_attr = getattr(model, "device", None)
    if device_attr is not None:
        return device_attr
    return _resolve_nllb_accelerator_device()


def _resolve_nllb_accelerator_device():
    """Resolve an available accelerator for NLLB model execution."""
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if is_mps_available(torch):
        return "mps"
    return None


def _load_nllb_from_pretrained_fallback(auto_model_for_seq2seq_lm, model_kwargs):
    """Load model with local-files fallback on network error, and auto-purge on corrupted cache."""
    return common.load_with_cache_recovery(
        auto_model_for_seq2seq_lm.from_pretrained,
        config.NLLB_MODEL_ID,
        model_kwargs,
        logger=LOGGER,
        model_label="NLLB model",
    )


def _load_nllb_model(auto_model_for_seq2seq_lm):
    """Load NLLB model with GPU fallback to CPU when VRAM is insufficient."""
    model_kwargs = _build_nllb_model_kwargs()
    try:
        return _load_nllb_from_pretrained_fallback(auto_model_for_seq2seq_lm, model_kwargs)
    except (MemoryError, RuntimeError) as exc:
        if not _can_retry_nllb_on_cpu(exc, model_kwargs):
            raise
    _clear_cuda_cache()
    return _load_nllb_from_pretrained_fallback(auto_model_for_seq2seq_lm, {"device_map": "cpu"})


def _can_retry_nllb_on_cpu(exc, model_kwargs):
    """Return whether a CUDA out-of-memory error permits CPU retry."""
    return _is_cuda_oom_error(exc) and model_kwargs.get("device_map") is not None


def _is_cuda_oom_error(exc):
    """Recognize CUDA out-of-memory errors from supported torch versions."""
    if "out of memory" in str(exc).lower():
        return True
    return torch is not None and hasattr(torch, "OutOfMemoryError") and isinstance(exc, torch.OutOfMemoryError)


def _clear_cuda_cache():
    """Release cached CUDA allocations before retrying model load on CPU."""
    if torch is not None and hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()


def _load_nllb_tokenizer(nllb_tokenizer):
    """Load NLLB tokenizer with local-files fallback and corrupt cache auto-purge."""
    return common.load_with_cache_recovery(
        nllb_tokenizer.from_pretrained,
        config.NLLB_MODEL_ID,
        logger=LOGGER,
        model_label="NLLB tokenizer",
    )
