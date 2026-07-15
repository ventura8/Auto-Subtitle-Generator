"""TranslateGemma translation backend implementation."""

import contextlib
import os
from typing import Any

from modules.configuration import config
from modules.runtime.optional_imports import load_optional_torch
from modules.translators.common import import_transformers_module, resolve_device_map

torch: Any | None = load_optional_torch()


class TranslateGemmaTranslator:
    """TranslateGemma translation wrapper using causal generation."""

    def __init__(self, runtime_settings=None):
        transformers = _import_transformers_module()
        auto_model_for_causal_lm = transformers.AutoModelForCausalLM
        auto_tokenizer = transformers.AutoTokenizer
        self._tokenizer = _load_translategemma_tokenizer(auto_tokenizer)
        self._model = _load_translategemma_model(auto_model_for_causal_lm)
        self._device = _resolve_generation_device(self._model)
        self._runtime_settings = _resolve_runtime_settings(runtime_settings)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        if hasattr(self._model, "eval"):
            self._model.eval()

    def translate(self, texts, src_code, tgt_code):
        """Translate a list of texts from source code to target code."""
        if not texts:
            return []

        src_iso = config.nllb_to_iso(src_code)
        tgt_iso = config.nllb_to_iso(tgt_code)
        request = {
            "device": self._device,
            "texts": texts,
            "src_iso": src_iso,
            "tgt_iso": tgt_iso,
            "runtime_settings": self._runtime_settings,
        }
        return _translate_batch_translategemma_texts(
            self._model,
            self._tokenizer,
            request,
        )

    def release(self):
        """Release wrapped model/tokenizer references."""
        self._model = None
        self._tokenizer = None
        self._device = None


def _import_transformers_module():
    """Import transformers lazily to preserve optional dependency behavior."""
    return import_transformers_module()


def _resolve_hf_token():
    """Resolve Hugging Face token from environment for gated model access."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    return None


def _resolve_device_map():
    """Resolve explicit device mapping for transformer loading."""
    return resolve_device_map()


def _load_translategemma_tokenizer(auto_tokenizer):
    """Load TranslateGemma tokenizer with local-files fallback on network errors."""
    token = _resolve_hf_token()
    load_args = {"token": token} if token else {}
    try:
        return auto_tokenizer.from_pretrained(config.TRANSLATEGEMMA_MODEL_ID, **load_args)
    except OSError:
        return auto_tokenizer.from_pretrained(config.TRANSLATEGEMMA_MODEL_ID, local_files_only=True, **load_args)


def _build_translategemma_model_kwargs():
    """Build model load kwargs for TranslateGemma without auto device offload."""
    kwargs = {}
    token = _resolve_hf_token()
    if token:
        kwargs["token"] = token

    device_map = _resolve_device_map()
    if device_map is not None:
        kwargs["device_map"] = device_map
        if torch is not None and hasattr(torch, "float16"):
            kwargs["dtype"] = torch.float16
    return kwargs


def _load_translategemma_model(auto_model_for_causal_lm):
    """Load TranslateGemma model with explicit device mapping and local fallback."""
    model_kwargs = _build_translategemma_model_kwargs()
    try:
        return auto_model_for_causal_lm.from_pretrained(config.TRANSLATEGEMMA_MODEL_ID, **model_kwargs)
    except OSError:
        return auto_model_for_causal_lm.from_pretrained(
            config.TRANSLATEGEMMA_MODEL_ID,
            local_files_only=True,
            **model_kwargs,
        )


def _resolve_generation_device(model):
    """Resolve the execution device for generation inputs."""
    if model is None:
        return None
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


def _build_translategemma_prompt(text, src_iso, tgt_iso):
    """Build deterministic subtitle translation prompt for causal generation."""
    return f"Translate this subtitle line from {src_iso} to {tgt_iso}. Return only the translated line.\nText: {text}\nTranslation:"


def _build_translation_prompts(texts, src_iso, tgt_iso):
    """Build deterministic prompts for a translation batch."""
    return [_build_translategemma_prompt(text, src_iso, tgt_iso) for text in texts]


def _move_encoded_to_device(encoded, device):
    """Move encoded batch tensors to target generation device when available."""
    if device is None:
        return encoded
    return {key: value.to(device) for key, value in encoded.items()}


def _decode_generated_batch(tokenizer, generated, texts, prompt_lengths):
    """Decode generated token IDs back into subtitle lines."""
    translated_texts = []
    for idx, source_text in enumerate(texts):
        generated_ids = generated[idx]
        translated_ids = generated_ids[prompt_lengths[idx] :]
        translated_text = tokenizer.decode(translated_ids, skip_special_tokens=True).strip()
        translated_texts.append(translated_text or source_text)
    return translated_texts


def _translate_batch_translategemma_texts(model, tokenizer, request):
    """Translate a batch of subtitle segments using one generation call."""
    prompts = _build_translation_prompts(request["texts"], request["src_iso"], request["tgt_iso"])
    encoded = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True)
    encoded = _move_encoded_to_device(encoded, request["device"])

    generation_kwargs = _build_generation_kwargs(encoded, tokenizer, request["runtime_settings"])
    with _inference_context():
        generated = model.generate(**generation_kwargs)
    prompt_lengths = _resolve_prompt_lengths(encoded)
    return _decode_generated_batch(tokenizer, generated, request["texts"], prompt_lengths)


def _resolve_runtime_settings(runtime_settings):
    """Normalize runtime settings with safe defaults."""
    raw = runtime_settings if isinstance(runtime_settings, dict) else {}
    max_new_tokens = _safe_positive_int(raw.get("max_new_tokens"), 160)
    return {
        "max_new_tokens": max_new_tokens,
    }


def _safe_positive_int(value, default):
    """Parse positive integer with fallback to default."""
    try:
        parsed = int(value)
        if parsed > 0:
            return parsed
    except (TypeError, ValueError):
        pass
    return default


def _build_generation_kwargs(encoded, tokenizer, runtime_settings):
    """Build stable generation kwargs for high-throughput deterministic translation."""
    return {
        **encoded,
        "max_new_tokens": runtime_settings["max_new_tokens"],
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
    }


def _inference_context():
    """Return inference context manager when torch provides one."""
    if torch is not None and hasattr(torch, "inference_mode"):
        return torch.inference_mode()
    return contextlib.nullcontext()


def _resolve_prompt_lengths(encoded_batch):
    """Resolve per-row prompt lengths from tokenizer batch output."""
    attention_mask = encoded_batch.get("attention_mask")
    if attention_mask is not None:
        return [int(length) for length in attention_mask.sum(dim=1)]

    input_ids = encoded_batch["input_ids"]
    return [len(row) for row in input_ids]
