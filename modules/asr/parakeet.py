"""NVIDIA Parakeet-TDT-0.6B-v3 speech recognition wrapper (transformers ``ParakeetForTDT``)."""

import contextlib
from typing import Any, cast

from modules.asr.common import DecodedClip, DeviceChoice, flatten_lstm_weights, load_pretrained, parakeet_step_budget
from modules.asr.tdt_guard import MaskDurationLogits, install_symbol_cap, loop_detected, mask_frames_past_end
from modules.runtime.optional_imports import load_optional_torch
from modules.translators.common import import_transformers_module, load_with_cache_recovery

torch: Any | None = load_optional_torch()

SAMPLE_RATE = 16000


class ParakeetModel:
    """Batched Parakeet TDT transcription with token timestamps and a decode-loop flag."""

    kind = "parakeet"

    def __init__(self, model_id: str, revision: str | None):
        transformers = import_transformers_module()
        self.model_id = model_id
        self._processor = load_with_cache_recovery(
            transformers.AutoProcessor.from_pretrained, model_id, {"revision": revision}, model_label="Parakeet processor"
        )
        # load_pretrained returns the model as ``object``; its transformers API is untyped here.
        self._model, self._choice = cast(
            tuple[Any, DeviceChoice], load_pretrained(transformers.AutoModelForTDT.from_pretrained, model_id, revision, "Parakeet")
        )
        self._model.eval()
        flatten_lstm_weights(self._model)
        _check_tdt_pair(self._processor, self._model)
        self.symbol_cap_installed = install_symbol_cap(self._model)
        self._logits_processors = transformers.LogitsProcessorList([MaskDurationLogits(self._model.config.vocab_size)])

    @property
    def device(self) -> str:
        """Device the model was loaded on (``cuda:0`` or ``cpu``)."""
        return self._choice.device

    def transcribe_batch(self, clips: list, _language: str) -> list[DecodedClip]:
        """Transcribe 16 kHz mono float32 clips; Parakeet detects the language itself."""
        inputs = self._processor(list(clips), sampling_rate=SAMPLE_RATE).to(self._choice.device, dtype=self._choice.dtype)
        valid_frames = _valid_encoder_frames(self._model, inputs["attention_mask"])
        budget = parakeet_step_budget(int(valid_frames.max()), self._model.max_symbols_per_step)
        with _inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=budget, logits_processor=self._logits_processors)
        blank_id = self._processor.blank_token_id
        durations = output.durations.cpu()
        # Batched rows keep decoding the padding after their own audio ends; blank those steps out before decoding.
        sequences = mask_frames_past_end(output.sequences.cpu(), durations, valid_frames, blank_id)
        texts, stamps = self._processor.decode(sequences, durations=durations, skip_special_tokens=True)
        rows = zip(texts, stamps, sequences, durations)
        return [
            DecodedClip(text.strip(), row_stamps, None, loop_detected(row_durs, row_seq, blank_id))
            for text, row_stamps, row_seq, row_durs in rows
        ]

    def release(self) -> None:
        """Drop the model and processor references so their memory can be reclaimed."""
        self._model = None
        self._processor = None
        self._logits_processors = None


def _inference_mode():
    """``torch.inference_mode()`` when torch is importable, else a no-op context."""
    if torch is None:
        return contextlib.nullcontext()
    return torch.inference_mode()


def _check_tdt_pair(processor, model) -> None:
    """Refuse a checkpoint whose processor is not a TDT decoder or disagrees with the model on the blank id."""
    decoder_type = getattr(processor, "_decoder_type")
    if decoder_type != "tdt" or processor.blank_token_id != model.config.blank_token_id:
        raise ValueError(
            f"Parakeet processor/model mismatch: decoder type {decoder_type!r}, "
            f"blank ids {processor.blank_token_id} vs {model.config.blank_token_id}."
        )


def _valid_encoder_frames(model, attention_mask):
    """Encoder frames each row really covers (CPU tensor); the rest of the padded batch is padding."""
    lengths = getattr(model, "_get_subsampling_output_length")(attention_mask.sum(-1))
    return lengths.cpu()
