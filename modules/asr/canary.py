"""NVIDIA Canary-1B-v2 speech recognition wrapper (transformers ``CanaryForConditionalGeneration``)."""

import contextlib
import math
from typing import Any, cast

from modules.asr.common import DecodedClip, DeviceChoice, canary_token_budget, load_pretrained
from modules.runtime.optional_imports import load_optional_torch
from modules.translators.common import import_transformers_module, load_with_cache_recovery

torch: Any | None = load_optional_torch()

SAMPLE_RATE = 16000


class CanaryModel:
    """Batched Canary transcription with a per-clip confidence and runaway flag."""

    kind = "canary"

    def __init__(self, model_id: str, revision: str | None):
        transformers = import_transformers_module()
        self.model_id = model_id
        self._processor = load_with_cache_recovery(
            transformers.AutoProcessor.from_pretrained, model_id, {"revision": revision}, model_label="Canary processor"
        )
        # load_pretrained returns the model as ``object``; its transformers API is untyped here.
        self._model, self._choice = cast(
            tuple[Any, DeviceChoice],
            load_pretrained(transformers.CanaryForConditionalGeneration.from_pretrained, model_id, revision, "Canary"),
        )
        self._model.eval()

    @property
    def device(self) -> str:
        """Device the model was loaded on (``cuda:0`` or ``cpu``)."""
        return self._choice.device

    def transcribe_batch(self, clips: list, language: str) -> list[DecodedClip]:
        """Transcribe 16 kHz mono float32 clips spoken in ``language`` in one ``generate`` call."""
        inputs = self._processor.apply_transcription_request(audio=list(clips), source_language=language, punctuation=True)
        prompt_len = _uniform_prompt_length(inputs["decoder_input_ids"].tolist(), self._processor.tokenizer.pad_token_id)
        inputs = inputs.to(self._choice.device, dtype=self._choice.dtype)
        budget = canary_token_budget(max(len(clip) for clip in clips) / SAMPLE_RATE, prompt_len, _decoder_positions(self._model))
        with _inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=budget, return_dict_in_generate=True, output_scores=True)
            scores = self._model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)
        generated = [row[prompt_len:] for row in output.sequences.cpu().tolist()]
        texts = self._processor.batch_decode(generated, skip_special_tokens=True)
        eos_ids = _eos_ids(self._model.generation_config.eos_token_id)
        rows = zip(texts, generated, scores.float().cpu().tolist())
        return [_decoded_clip(text, tokens, row_scores, eos_ids) for text, tokens, row_scores in rows]

    def release(self) -> None:
        """Drop the model and processor references so their memory can be reclaimed."""
        self._model = None
        self._processor = None


def _inference_mode():
    """``torch.inference_mode()`` when torch is importable, else a no-op context."""
    if torch is None:
        return contextlib.nullcontext()
    return torch.inference_mode()


def _uniform_prompt_length(prompts: list[list[int]], pad_id: int | None) -> int:
    """Shared decoder prompt length; padded prompts are refused because Canary gets no decoder mask."""
    if any(pad_id in prompt for prompt in prompts):
        raise ValueError("Canary prompts in one batch must have the same length (no decoder attention mask is passed).")
    return len(prompts[0])


def _decoder_positions(model) -> int:
    """Size of the decoder positional table, the hard cap on prompt plus generated tokens."""
    return int(model.config.decoder_config.max_position_embeddings)


def _eos_ids(eos_token_id) -> frozenset[int]:
    """Normalise the generation config's EOS id (an int or a list) to a set."""
    if isinstance(eos_token_id, int):
        return frozenset((eos_token_id,))
    return frozenset(eos_token_id or ())


def _decoded_clip(text: str, tokens: list[int], scores: list[float], eos_ids: frozenset[int]) -> DecodedClip:
    """Build one result; a row that never emitted EOS ran into the token budget and is degenerate."""
    end = next((index for index, token in enumerate(tokens) if token in eos_ids), None)
    if end is not None:
        scores = scores[: end + 1]
    return DecodedClip(text.strip(), None, _mean_logprob(scores), end is None)


def _mean_logprob(scores: list[float]) -> float:
    """Mean of the finite token log-probabilities; positions after EOS must already be cut off."""
    finite = [score for score in scores if math.isfinite(score)]
    if not finite:
        return float("-inf")
    return sum(finite) / len(finite)
