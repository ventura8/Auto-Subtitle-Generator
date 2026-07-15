"""NLLB translation backend implementation."""

from typing import Any

from modules.configuration import config
from modules.runtime.optional_imports import load_optional_torch
from modules.translators.common import import_transformers_module, resolve_device_map

torch: Any | None = load_optional_torch()


class NLLBTranslator:
    """NLLB translation wrapper with local-cache fallback on load errors."""

    def __init__(self):
        transformers = _import_transformers_module()
        auto_model_for_seq2seq_lm = transformers.AutoModelForSeq2SeqLM
        nllb_tokenizer = transformers.NllbTokenizer
        self._tokenizer = _load_nllb_tokenizer(nllb_tokenizer)
        self._model = _load_nllb_model(auto_model_for_seq2seq_lm)

    def translate(self, texts, src_code, tgt_code):
        """Translate a list of texts from source code to target code."""
        if not texts:
            return []

        self._tokenizer.src_lang = src_code
        encoded = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        if torch is not None and torch.cuda.is_available():
            encoded = {key: value.to("cuda") for key, value in encoded.items()}

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


def _import_transformers_module():
    """Import transformers lazily to preserve optional dependency behavior."""
    return import_transformers_module()


def _resolve_device_map():
    """Resolve explicit device mapping for transformer loading."""
    return resolve_device_map()


def _load_nllb_model(auto_model_for_seq2seq_lm):
    """Load NLLB model with local-files fallback when remote fetch fails."""
    device_map = _resolve_device_map()
    try:
        return auto_model_for_seq2seq_lm.from_pretrained(
            config.NLLB_MODEL_ID,
            device_map=device_map,
        )
    except OSError:
        return auto_model_for_seq2seq_lm.from_pretrained(
            config.NLLB_MODEL_ID,
            local_files_only=True,
            device_map=device_map,
        )


def _load_nllb_tokenizer(nllb_tokenizer):
    """Load NLLB tokenizer with local-files fallback when remote fetch fails."""
    try:
        return nllb_tokenizer.from_pretrained(config.NLLB_MODEL_ID)
    except OSError:
        return nllb_tokenizer.from_pretrained(config.NLLB_MODEL_ID, local_files_only=True)
