"""Tests for the Parakeet TDT ASR wrapper (transformers and the TDT guards are mocked; no real tensors)."""

import contextlib
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules.asr import parakeet

_Choice = namedtuple("_Choice", ["device", "dtype"])
_STAMPS = [[{"token": " Bună", "start": 0.0, "end": 0.32}, {"token": " ziua", "start": 0.4, "end": 0.8}], []]


class _Batch(dict):
    """Dict standing in for a transformers ``BatchFeature``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moved_to = None

    def to(self, device, dtype=None):
        """Record the move and return the same batch."""
        self.moved_to = (device, dtype)
        return self


def _tdt_mocks():
    """Processor and model mocks that pass the TDT sanity check."""
    processor, model = MagicMock(), MagicMock()
    processor._decoder_type = "tdt"
    processor.blank_token_id = 8192
    model.config.blank_token_id = 8192
    model.config.vocab_size = 8193
    model.max_symbols_per_step = 10
    return processor, model


def _build(processor, model):
    """Construct a ParakeetModel with every loader and guard patched; returns the wrapper and the mocks behind it."""
    transformers = MagicMock()
    with (
        patch.object(parakeet, "import_transformers_module", return_value=transformers),
        patch.object(parakeet, "load_with_cache_recovery", return_value=processor) as load_processor,
        patch.object(parakeet, "load_pretrained", return_value=(model, _Choice("cuda:0", "bf16"))) as load_model,
        patch.object(parakeet, "flatten_lstm_weights") as flatten,
        patch.object(parakeet, "install_symbol_cap", return_value=True) as install_cap,
        patch.object(parakeet, "MaskDurationLogits") as mask_logits,
    ):
        wrapper = parakeet.ParakeetModel("nvidia/parakeet-tdt-0.6b-v3", "def456")
    return SimpleNamespace(
        wrapper=wrapper,
        transformers=transformers,
        load_processor=load_processor,
        load_model=load_model,
        flatten=flatten,
        install_cap=install_cap,
        mask_logits=mask_logits,
    )


class ParakeetLoadTests(unittest.TestCase):
    """Loading wires processor, model, guards and device together."""

    def setUp(self):
        self.processor, self.model = _tdt_mocks()
        self.built = _build(self.processor, self.model)

    def test_loads_processor_and_model_at_revision(self):
        """The processor is fetched at the pinned revision and the model goes through load_pretrained."""
        transformers = self.built.transformers
        self.built.load_processor.assert_called_once_with(
            transformers.AutoProcessor.from_pretrained,
            "nvidia/parakeet-tdt-0.6b-v3",
            {"revision": "def456"},
            model_label="Parakeet processor",
        )
        self.built.load_model.assert_called_once_with(
            transformers.AutoModelForTDT.from_pretrained, "nvidia/parakeet-tdt-0.6b-v3", "def456", "Parakeet"
        )
        self.model.eval.assert_called_once_with()

    def test_installs_guards_once_at_load(self):
        """LSTM weights are flattened, the symbol cap is installed and duration logits are masked."""
        self.built.flatten.assert_called_once_with(self.model)
        self.built.install_cap.assert_called_once_with(self.model)
        self.assertTrue(self.built.wrapper.symbol_cap_installed)
        self.built.mask_logits.assert_called_once_with(8193)
        self.built.transformers.LogitsProcessorList.assert_called_once_with([self.built.mask_logits.return_value])

    def test_exposes_kind_device_and_model_id(self):
        """The wrapper reports its engine kind, model id and load device."""
        self.assertEqual(self.built.wrapper.kind, "parakeet")
        self.assertEqual(self.built.wrapper.model_id, "nvidia/parakeet-tdt-0.6b-v3")
        self.assertEqual(self.built.wrapper.device, "cuda:0")

    def test_release_drops_references(self):
        """release() clears the model, processor and logits processors."""
        wrapper = self.built.wrapper
        wrapper.release()
        self.assertIsNone(getattr(wrapper, "_model"))
        self.assertIsNone(getattr(wrapper, "_processor"))
        self.assertIsNone(getattr(wrapper, "_logits_processors"))


class ParakeetSanityTests(unittest.TestCase):
    """A checkpoint that is not a TDT pair is refused."""

    def test_ctc_processor_is_refused(self):
        """A CTC processor cannot decode TDT durations."""
        processor, model = _tdt_mocks()
        processor._decoder_type = "ctc"
        with self.assertRaises(ValueError):
            _build(processor, model)

    def test_blank_id_mismatch_is_refused(self):
        """Processor and model must agree on the blank id used for masking."""
        processor, model = _tdt_mocks()
        model.config.blank_token_id = 1024
        with self.assertRaises(ValueError):
            _build(processor, model)


class ParakeetTranscribeTests(unittest.TestCase):
    """Batched transcription: frame budget, padding mask, decode and loop flag."""

    def setUp(self):
        self.processor, self.model = _tdt_mocks()
        self.wrapper = _build(self.processor, self.model).wrapper
        self.batch = _Batch(input_features="feats", attention_mask=MagicMock())
        self.processor.return_value = self.batch
        self.valid = MagicMock()
        self.valid.max.return_value = 74
        self.model._get_subsampling_output_length.return_value.cpu.return_value = self.valid
        self.model.generate.return_value = SimpleNamespace(sequences=MagicMock(), durations=MagicMock())
        self.model.generate.return_value.durations.cpu.return_value = [["dur0"], ["dur1"]]
        self.processor.decode.return_value = (["  Bună ziua ", ""], _STAMPS)
        self.patches = {
            "parakeet_step_budget": MagicMock(return_value=741),
            "mask_frames_past_end": MagicMock(return_value=[["seq0"], ["seq1"]]),
            "loop_detected": MagicMock(side_effect=[False, True]),
        }
        for name, mock in self.patches.items():
            patcher = patch.object(parakeet, name, mock)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.clips = [[0.0] * 16000, [0.0] * 8000]

    def test_features_move_to_load_device(self):
        """Clips are featurised at 16 kHz and moved to the load device and dtype."""
        self.wrapper.transcribe_batch(self.clips, "ro")
        self.processor.assert_called_once_with(self.clips, sampling_rate=16000)
        self.assertEqual(self.batch.moved_to, ("cuda:0", "bf16"))
        self.model._get_subsampling_output_length.assert_called_once_with(self.batch["attention_mask"].sum.return_value)

    def test_step_budget_covers_longest_valid_row(self):
        """max_new_tokens comes from the longest valid frame count and the per-frame symbol cap."""
        self.wrapper.transcribe_batch(self.clips, "ro")
        self.patches["parakeet_step_budget"].assert_called_once_with(74, 10)
        kwargs = self.model.generate.call_args.kwargs
        self.assertEqual(kwargs["max_new_tokens"], 741)
        self.assertIs(kwargs["logits_processor"], getattr(self.wrapper, "_logits_processors"))
        self.assertEqual(kwargs["input_features"], "feats")

    def test_padding_steps_are_masked_before_decode(self):
        """Steps past each row's own audio are blanked, and decode gets the durations for timestamps."""
        self.wrapper.transcribe_batch(self.clips, "ro")
        output = self.model.generate.return_value
        durations = output.durations.cpu.return_value
        self.patches["mask_frames_past_end"].assert_called_once_with(output.sequences.cpu.return_value, durations, self.valid, 8192)
        self.processor.decode.assert_called_once_with([["seq0"], ["seq1"]], durations=durations, skip_special_tokens=True)

    def test_results_carry_stamps_and_loop_flag(self):
        """Each clip gets its stripped text, its own stamps, no logprob and the loop verdict."""
        first, second = self.wrapper.transcribe_batch(self.clips, "ro")
        self.assertEqual(first, parakeet.DecodedClip("Bună ziua", _STAMPS[0], None, False))
        self.assertEqual(second.text, "")
        self.assertTrue(second.degenerate)
        self.patches["loop_detected"].assert_called_with(["dur1"], ["seq1"], 8192)


class ParakeetHelperTests(unittest.TestCase):
    """Pure helpers."""

    def test_inference_mode_uses_torch_when_available(self):
        """With torch present generation runs under torch.inference_mode()."""
        fake_torch = MagicMock()
        with patch.object(parakeet, "torch", fake_torch):
            self.assertIs(getattr(parakeet, "_inference_mode")(), fake_torch.inference_mode.return_value)

    def test_inference_mode_without_torch(self):
        """Without torch the context is a no-op."""
        with patch.object(parakeet, "torch", None):
            self.assertIsInstance(getattr(parakeet, "_inference_mode")(), contextlib.nullcontext)
