"""Tests for the Parakeet-TDT decoding guards in ``modules.asr.tdt_guard``.

Torch is a MagicMock under pytest, so the cap arithmetic runs on a small list-backed
stand-in for 1-D tensors and the rest is checked for control flow. The real-torch
behaviour (stock loop vs patched walk, padded-batch masking) is exercised on a tiny
random-weight ``ParakeetForTDT`` outside the mocked test process.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules.asr import tdt_guard

BLANK = 4
TDT_MODULE = "transformers.models.parakeet.generation_parakeet"


class _Vec:
    """List-backed stand-in for a 1-D tensor with only the methods ``cap_symbols`` uses."""

    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)

    def eq(self, other):
        """Element-wise equality with a scalar."""
        return _Vec(value == other for value in self.values)

    def ge(self, other):
        """Element-wise ``>=`` with a scalar."""
        return _Vec(value >= other for value in self.values)

    def new_zeros(self, shape):
        """Zeros of the given 1-D shape."""
        return _Vec([0] * shape[0])

    def where(self, condition, other):
        """Keep values where ``condition`` holds, else ``other``."""
        return _Vec(value if keep else other for value, keep in zip(self.values, condition.values))

    def tolist(self):
        """Plain list copy."""
        return list(self.values)

    def __add__(self, other):
        return _Vec(value + other for value in self.values)

    def __and__(self, other):
        return _Vec(left and right for left, right in zip(self.values, other.values))

    def __or__(self, other):
        return _Vec(left or right for left, right in zip(self.values, other.values))

    def __invert__(self):
        return _Vec(not value for value in self.values)


class GenerationMixin:
    """Stand-in for transformers' base mixin; its update records the call."""

    def _update_model_kwargs_for_generation(self, outputs, *args, **kwargs):
        return {"base_args": (self, outputs, args, kwargs), "encoder_frame_idxs": MagicMock(), "encoder_valid_lengths": "valid"}


def _tdt_update(self, outputs, *args, **kwargs):
    """Stand-in for the TDT mixin update the guard replaces."""
    return {"tdt_args": (self, outputs, args, kwargs)}


_tdt_update.__module__ = TDT_MODULE
_tdt_update.__qualname__ = "ParakeetTDTGenerationMixin._update_model_kwargs_for_generation"


class FakeTDT(GenerationMixin):
    _update_model_kwargs_for_generation = _tdt_update


class FakeRNNT(GenerationMixin):
    pass


class NoBaseTDT:
    _update_model_kwargs_for_generation = _tdt_update


def _tdt_model(model_type=FakeTDT, **fields):
    """Instance with the fields the symbol cap needs, overridable per test."""
    model = model_type()
    model.config = fields.pop("config", SimpleNamespace(vocab_size=5, blank_token_id=BLANK))
    model.max_symbols_per_step = fields.pop("max_symbols_per_step", 3)
    model._symbols_at_frame = None
    model._step_durations = []
    model._encoder_finished = None
    return model


class TestMaskDurationLogits(unittest.TestCase):
    def test_masks_duration_columns_in_place(self):
        scores = MagicMock()
        self.assertIs(tdt_guard.MaskDurationLogits(5)("ids", scores), scores)
        scores.__setitem__.assert_called_once_with((slice(None), slice(5, None)), float("-inf"))

    def test_mask_method_is_the_call_body(self):
        scores = MagicMock()
        self.assertIs(tdt_guard.MaskDurationLogits(7).mask(scores), scores)
        scores.__setitem__.assert_called_once_with((slice(None), slice(7, None)), float("-inf"))


class TestCapSymbols(unittest.TestCase):
    def _cap(self, tokens, durations, previous):
        prior = None if previous is None else _Vec(previous)
        capped, symbols = tdt_guard.cap_symbols(_Vec(tokens), _Vec(durations), prior, (BLANK, 3))
        return capped.tolist(), symbols.tolist()

    def test_first_step_counts_from_zero(self):
        self.assertEqual(self._cap([1, BLANK], [0, 2], None), ([0, 2], [1, 0]))

    def test_reaching_the_cap_forces_one_frame(self):
        self.assertEqual(self._cap([1, 1], [0, 0], [1, 2]), ([0, 1], [2, 0]))

    def test_stalled_blank_advances_and_resets(self):
        self.assertEqual(self._cap([BLANK], [0], [2]), ([1], [0]))

    def test_real_advance_resets_the_count(self):
        self.assertEqual(self._cap([1, BLANK], [3, 1], [2, 2]), ([3, 1], [0, 0]))


class TestInstallSymbolCap(unittest.TestCase):
    def setUp(self):
        log_patch = patch("modules.asr.tdt_guard.log")
        self.log = log_patch.start()
        self.addCleanup(log_patch.stop)

    def test_installs_on_the_instance_only(self):
        model = _tdt_model()
        self.assertTrue(tdt_guard.install_symbol_cap(model))
        self.assertIn("_update_model_kwargs_for_generation", vars(model))
        self.assertIs(vars(FakeTDT)["_update_model_kwargs_for_generation"], _tdt_update)
        self.log.assert_not_called()

    def test_refuses_an_unknown_update_hook(self):
        self.assertFalse(tdt_guard.install_symbol_cap(_tdt_model(FakeRNNT)))
        self.assertEqual(self.log.call_args.args[1], "WARNING")

    def test_refuses_when_cap_fields_are_missing(self):
        model = _tdt_model(config=SimpleNamespace(vocab_size=5))
        self.assertFalse(tdt_guard.install_symbol_cap(model))
        del model.max_symbols_per_step
        model.config.blank_token_id = BLANK
        self.assertFalse(tdt_guard.install_symbol_cap(model))
        self.assertNotIn("_update_model_kwargs_for_generation", vars(model))

    def test_refuses_without_a_generation_mixin_base(self):
        self.assertFalse(tdt_guard.install_symbol_cap(_tdt_model(NoBaseTDT)))

    def test_refuses_a_hook_from_another_module(self):
        def other(self, outputs):
            return (self, outputs)

        other.__qualname__ = _tdt_update.__qualname__
        model_type = type("Other", (GenerationMixin,), {"_update_model_kwargs_for_generation": other})
        self.assertFalse(tdt_guard.install_symbol_cap(_tdt_model(model_type)))


class TestCappedUpdate(unittest.TestCase):
    def test_patched_update_wraps_base_and_advances_by_capped_durations(self):
        model = _tdt_model()
        tdt_guard.install_symbol_cap(model)
        outputs = MagicMock()
        capped, symbols = MagicMock(name="durations"), MagicMock(name="symbols")
        with patch("modules.asr.tdt_guard.cap_symbols", return_value=(capped, symbols)) as cap:
            result = model._update_model_kwargs_for_generation(outputs, "model_kwargs", is_encoder_decoder=True)
        self.assertEqual(result["base_args"], (model, outputs, ("model_kwargs",), {"is_encoder_decoder": True}))
        self.assertEqual(cap.call_args.args[2:], (None, (BLANK, 3)))
        self.assertIs(model._symbols_at_frame, symbols)
        self.assertEqual(model._step_durations, [capped])
        self.assertIs(model._encoder_finished, result["encoder_frame_idxs"].ge.return_value)
        result["encoder_frame_idxs"].ge.assert_called_once_with("valid")

    def test_patched_update_reads_tokens_and_durations_from_last_step_logits(self):
        model = _tdt_model()
        tdt_guard.install_symbol_cap(model)
        outputs = MagicMock()
        with patch("modules.asr.tdt_guard.cap_symbols", return_value=(MagicMock(), MagicMock())) as cap:
            model._update_model_kwargs_for_generation(outputs)
        last = outputs.logits.__getitem__.return_value
        last.__getitem__.assert_any_call((slice(None), slice(None, 5)))
        last.__getitem__.assert_any_call((slice(None), slice(5, None)))
        self.assertIs(cap.call_args.args[0], last.__getitem__.return_value.argmax.return_value)


class TestMaskFramesPastEnd(unittest.TestCase):
    def test_blanks_steps_starting_at_or_after_valid_frames(self):
        sequences, durations, valid = MagicMock(), MagicMock(), MagicMock()
        sequences.shape = (2, 7)
        result = tdt_guard.mask_frames_past_end(sequences, durations, valid, BLANK)
        durations.__getitem__.assert_called_once_with((slice(None), slice(None, 7)))
        trimmed = durations.__getitem__.return_value
        starts = trimmed.cumsum.return_value.__sub__.return_value
        trimmed.cumsum.assert_called_once_with(-1)
        valid.to.assert_called_once_with(starts.device)
        starts.ge.assert_called_once_with(valid.to.return_value.reshape.return_value)
        valid.to.return_value.reshape.assert_called_once_with(-1, 1)
        self.assertIs(result, sequences.masked_fill.return_value)
        sequences.masked_fill.assert_called_once_with(starts.ge.return_value, BLANK)


class TestLoopDetected(unittest.TestCase):
    def test_long_zero_duration_run_is_a_loop(self):
        self.assertTrue(tdt_guard.loop_detected([0] * 12, [7] * 12, BLANK))

    def test_run_shorter_than_threshold_is_not_a_loop(self):
        self.assertFalse(tdt_guard.loop_detected([0] * 9, [7] * 9, BLANK))

    def test_blank_or_advance_breaks_the_run(self):
        durations = [0] * 5 + [1] + [0] * 5 + [0] + [0] * 5
        tokens = [7] * 11 + [BLANK] + [7] * 5
        self.assertFalse(tdt_guard.loop_detected(durations, tokens, BLANK, run=6))

    def test_accepts_tensor_rows_and_custom_run(self):
        self.assertTrue(tdt_guard.loop_detected(_Vec([1, 0, 0, 0]), _Vec([7, 7, 7, 7]), BLANK, run=3))
        self.assertEqual(tdt_guard.LOOP_RUN, 10)
