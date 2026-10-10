"""Decoding guards for the transformers Parakeet-TDT greedy decoder.

transformers' TDT generation mixin overrides the per-step update and skips the RNN-T
``max_symbols_per_step`` cap (transformers #49388): a joint that keeps predicting a
non-blank token with duration 0 never advances the encoder and emits the same token
until ``max_new_tokens`` runs out. ``install_symbol_cap`` restores NeMo's cap on the
model instance. Batched rows also keep decoding padding frames after their own audio
ends (the TDT config has no EOS); ``mask_frames_past_end`` blanks those steps.

Nothing here imports torch: every operation is a tensor method, so the guards follow
the tensors' device and stay importable without the ML dependencies.
"""

import functools
import types

from modules.runtime.logging_utils import log

LOOP_RUN = 10

_TDT_MIXIN_MODULE_SUFFIX = "generation_parakeet"
_TDT_UPDATE_QUALNAME = "ParakeetTDTGenerationMixin._update_model_kwargs_for_generation"
_UPDATE_ATTR = "_update_model_kwargs_for_generation"
# Per-generate state the transformers mixin resets in ``generate`` and reads in its stopping criteria.
_SYMBOLS_ATTR = "_symbols_at_frame"
_STEP_DURATIONS_ATTR = "_step_durations"
_FINISHED_ATTR = "_encoder_finished"
_MISSING = object()


class MaskDurationLogits:
    """Logits processor that rules out the duration head so the emitted id is always a vocabulary token."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, _input_ids, scores):
        """Set every duration logit (columns from ``vocab_size`` on) to ``-inf``."""
        return self.mask(scores)

    def mask(self, scores):
        """Mask the duration columns of ``scores`` in place and return it."""
        scores[:, self.vocab_size :] = float("-inf")
        return scores


def _base_update(model_type):
    """``GenerationMixin``'s own update hook from the model's MRO, or None."""
    for klass in model_type.__mro__:
        if klass.__name__ == "GenerationMixin" and _UPDATE_ATTR in vars(klass):
            return vars(klass)[_UPDATE_ATTR]
    return None


def _is_tdt_update(method) -> bool:
    """Return True when ``method`` is the TDT mixin update this guard was written against."""
    module = getattr(method, "__module__", None) or ""
    return module.endswith(_TDT_MIXIN_MODULE_SUFFIX) and getattr(method, "__qualname__", "") == _TDT_UPDATE_QUALNAME


def _has_cap_fields(model) -> bool:
    """Return True when the model exposes the fields the symbol cap needs."""
    config = getattr(model, "config", None)
    fields = (getattr(model, "max_symbols_per_step", _MISSING), getattr(config, "vocab_size", _MISSING))
    return _MISSING not in fields and getattr(config, "blank_token_id", _MISSING) is not _MISSING


def _patchable_base(model):
    """Base update to wrap when ``model`` runs the known TDT update, else None."""
    model_type = type(model)
    if not _is_tdt_update(getattr(model_type, _UPDATE_ATTR, None)) or not _has_cap_fields(model):
        return None
    return _base_update(model_type)


def cap_symbols(tokens, durations, symbols_at_frame, limits):
    """Return ``(durations, symbols)`` with a forced one-frame advance where the per-frame cap is reached.

    ``limits`` is ``(blank_id, max_symbols)``. A non-blank token with duration 0 counts towards the
    cap; a blank with duration 0 (which would stall) or a count at the cap advances one frame.
    """
    blank_id, max_symbols = limits
    blank = tokens.eq(blank_id)
    stay = durations.eq(0)
    previous = tokens.new_zeros(tokens.shape) if symbols_at_frame is None else symbols_at_frame
    symbols = (previous + 1).where(stay & ~blank, 0)
    force = (stay & blank) | symbols.ge(max_symbols)
    return durations.where(~force, 1), symbols.where(~force, 0)


def _capped_update(base_update, model, outputs, *args, **kwargs):
    """TDT update with NeMo's ``max_symbols_per_step`` cap (mirrors the RNN-T mixin's guard)."""
    model_kwargs = base_update(model, outputs, *args, **kwargs)
    config = model.config
    logits = outputs.logits[:, -1, :]
    tokens = logits[:, : config.vocab_size].argmax(-1)
    # Duration index == duration value for the shipped (0, 1, 2, 3, 4) head, as in the upstream update.
    durations = logits[:, config.vocab_size :].argmax(-1)
    limits = (config.blank_token_id, model.max_symbols_per_step)
    durations, symbols = cap_symbols(tokens, durations, getattr(model, _SYMBOLS_ATTR, None), limits)
    setattr(model, _SYMBOLS_ATTR, symbols)
    model_kwargs["encoder_frame_idxs"] = model_kwargs["encoder_frame_idxs"] + durations
    getattr(model, _STEP_DURATIONS_ATTR).append(durations)
    setattr(model, _FINISHED_ATTR, model_kwargs["encoder_frame_idxs"].ge(model_kwargs["encoder_valid_lengths"]))
    return model_kwargs


def install_symbol_cap(model) -> bool:
    """Patch the per-frame symbol cap onto this TDT model instance; False (with a WARNING) when unrecognised."""
    base_update = _patchable_base(model)
    if base_update is None:
        log(
            f"  [Parakeet] Unrecognised TDT generation code in {type(model).__name__}; the per-frame symbol cap was not "
            "installed. Decode loops are only caught after the fact.",
            "WARNING",
        )
        return False
    setattr(model, _UPDATE_ATTR, types.MethodType(functools.partial(_capped_update, base_update), model))
    return True


def mask_frames_past_end(sequences, durations, valid_frames, blank_id):
    """Replace tokens decoded at or after each row's last valid encoder frame with ``blank_id``."""
    durations = durations[:, : sequences.shape[1]]
    starts = durations.cumsum(-1) - durations
    past_end = starts.ge(valid_frames.to(starts.device).reshape(-1, 1))
    return sequences.masked_fill(past_end, blank_id)


def _as_list(row) -> list:
    """Plain Python list from a tensor row or any iterable."""
    return row.tolist() if hasattr(row, "tolist") else list(row)


def loop_detected(durations_row, sequences_row, blank_id, run: int = LOOP_RUN) -> bool:
    """Return True when ``run`` consecutive steps emitted a non-blank token without advancing a frame."""
    streak = 0
    for duration, token in zip(_as_list(durations_row), _as_list(sequences_row)):
        streak = streak + 1 if duration == 0 and token != blank_id else 0
        if streak >= run:
            return True
    return False
