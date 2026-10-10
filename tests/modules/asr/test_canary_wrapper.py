"""Tests for the Canary ASR wrapper (transformers is mocked; no real tensors)."""

import contextlib
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules.asr import canary

_Choice = namedtuple("_Choice", ["device", "dtype"])


class _Batch(dict):
    """Dict standing in for a transformers ``BatchFeature``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moved_to = None

    def to(self, device, dtype=None):
        """Record the move and return the same batch."""
        self.moved_to = (device, dtype)
        return self


def _tolist_mock(value):
    """Mock whose ``tolist()`` returns ``value``."""
    mock = MagicMock()
    mock.tolist.return_value = value
    return mock


def _build():
    """Construct a CanaryModel with every loader patched; returns the wrapper and the mocks behind it."""
    transformers, processor, model = MagicMock(), MagicMock(), MagicMock()
    with (
        patch.object(canary, "import_transformers_module", return_value=transformers),
        patch.object(canary, "load_with_cache_recovery", return_value=processor) as load_processor,
        patch.object(canary, "load_pretrained", return_value=(model, _Choice("cuda:0", "bf16"))) as load_model,
    ):
        wrapper = canary.CanaryModel("nvidia/canary-1b-v2", "abc123")
    return SimpleNamespace(
        wrapper=wrapper, processor=processor, model=model, transformers=transformers, load_model=load_model, load_processor=load_processor
    )


def _prepare_generation(processor, model, prompts):
    """Wire processor/model mocks for a two-clip batch; returns the batch handed to generate."""
    batch = _Batch(input_features="feats", attention_mask="mask")
    batch["decoder_input_ids"] = _tolist_mock(prompts)
    processor.apply_transcription_request.return_value = batch
    processor.tokenizer.pad_token_id = 2
    processor.batch_decode.return_value = ["  Bună ziua.  ", "da da da"]
    sequences = MagicMock()
    sequences.cpu.return_value.tolist.return_value = [[7, 8, 9, 10, 11, 3, 2], [7, 8, 9, 12, 13, 14, 15]]
    model.generate.return_value = SimpleNamespace(sequences=sequences, scores=("step",))
    scores = model.compute_transition_scores.return_value
    scores.float.return_value.cpu.return_value.tolist.return_value = [[-0.1, -0.3, -0.2, -9.0], [-0.5, -0.5, float("-inf"), -0.5]]
    model.generation_config.eos_token_id = [3]
    model.config.decoder_config.max_position_embeddings = 1024
    return batch


class CanaryLoadTests(unittest.TestCase):
    """Loading wires processor, model and device together."""

    def test_loads_processor_and_model_at_revision(self):
        """The processor is fetched at the pinned revision and the model goes through load_pretrained."""
        built = _build()
        built.load_processor.assert_called_once_with(
            built.transformers.AutoProcessor.from_pretrained, "nvidia/canary-1b-v2", {"revision": "abc123"}, model_label="Canary processor"
        )
        built.load_model.assert_called_once_with(
            built.transformers.CanaryForConditionalGeneration.from_pretrained, "nvidia/canary-1b-v2", "abc123", "Canary"
        )
        built.model.eval.assert_called_once_with()

    def test_exposes_kind_device_and_model_id(self):
        """The wrapper reports its engine kind, model id and load device."""
        wrapper = _build().wrapper
        self.assertEqual(wrapper.kind, "canary")
        self.assertEqual(wrapper.model_id, "nvidia/canary-1b-v2")
        self.assertEqual(wrapper.device, "cuda:0")

    def test_release_drops_references(self):
        """release() clears the model and processor."""
        wrapper = _build().wrapper
        wrapper.release()
        self.assertIsNone(getattr(wrapper, "_model"))
        self.assertIsNone(getattr(wrapper, "_processor"))


class CanaryTranscribeTests(unittest.TestCase):
    """Batched transcription: prompt handling, budget, logprob and degenerate flag."""

    def setUp(self):
        built = _build()
        self.wrapper, self.processor, self.model = built.wrapper, built.processor, built.model
        self.batch = _prepare_generation(self.processor, self.model, [[7, 8, 9], [7, 8, 9]])
        budget = patch.object(canary, "canary_token_budget", return_value=40)
        self.budget = budget.start()
        self.addCleanup(budget.stop)
        self.clips = [[0.0] * 16000, [0.0] * 32000]

    def test_requests_transcription_and_moves_batch(self):
        """The processor gets the language with punctuation and the batch moves to the load device and dtype."""
        self.wrapper.transcribe_batch(self.clips, "ro")
        self.processor.apply_transcription_request.assert_called_once_with(audio=self.clips, source_language="ro", punctuation=True)
        self.assertEqual(self.batch.moved_to, ("cuda:0", "bf16"))

    def test_budget_uses_longest_clip_and_prompt_length(self):
        """The token budget is sized from the longest clip, the prompt length and the positional table."""
        self.wrapper.transcribe_batch(self.clips, "ro")
        self.budget.assert_called_once_with(2.0, 3, 1024)
        kwargs = self.model.generate.call_args.kwargs
        self.assertEqual(kwargs["max_new_tokens"], 40)
        self.assertTrue(kwargs["return_dict_in_generate"])
        self.assertTrue(kwargs["output_scores"])
        self.assertEqual(kwargs["input_features"], "feats")

    def test_prompt_is_sliced_before_decoding(self):
        """Only generated tokens reach batch_decode."""
        self.wrapper.transcribe_batch(self.clips, "ro")
        self.processor.batch_decode.assert_called_once_with([[10, 11, 3, 2], [12, 13, 14, 15]], skip_special_tokens=True)
        self.model.compute_transition_scores.assert_called_once_with(
            self.model.generate.return_value.sequences, ("step",), normalize_logits=True
        )

    def test_logprob_stops_at_first_eos_and_flags_runaway_rows(self):
        """The mean log-probability ignores post-EOS padding; a row without EOS is degenerate."""
        first, second = self.wrapper.transcribe_batch(self.clips, "ro")
        self.assertEqual(first.text, "Bună ziua.")
        self.assertIsNone(first.stamps)
        self.assertAlmostEqual(first.logprob, -0.2)
        self.assertFalse(first.degenerate)
        self.assertEqual(second.text, "da da da")
        self.assertAlmostEqual(second.logprob, -0.5)
        self.assertTrue(second.degenerate)

    def test_padded_prompts_are_refused(self):
        """Prompts of different lengths cannot be batched because no decoder mask is passed."""
        self.batch["decoder_input_ids"] = _tolist_mock([[7, 8, 9], [7, 8, 2]])
        with self.assertRaises(ValueError):
            self.wrapper.transcribe_batch(self.clips, "ro")
        self.model.generate.assert_not_called()


class CanaryHelperTests(unittest.TestCase):
    """Pure helpers."""

    def test_eos_ids_accepts_int_list_and_none(self):
        """The generation config may carry one id, several ids or none."""
        self.assertEqual(getattr(canary, "_eos_ids")(3), frozenset({3}))
        self.assertEqual(getattr(canary, "_eos_ids")([3, 4]), frozenset({3, 4}))
        self.assertEqual(getattr(canary, "_eos_ids")(None), frozenset())

    def test_mean_logprob_without_finite_scores(self):
        """No finite score means no confidence at all."""
        self.assertEqual(getattr(canary, "_mean_logprob")([float("-inf")]), float("-inf"))
        self.assertEqual(getattr(canary, "_mean_logprob")([]), float("-inf"))

    def test_inference_mode_uses_torch_when_available(self):
        """With torch present generation runs under torch.inference_mode()."""
        fake_torch = MagicMock()
        with patch.object(canary, "torch", fake_torch):
            self.assertIs(getattr(canary, "_inference_mode")(), fake_torch.inference_mode.return_value)

    def test_inference_mode_without_torch(self):
        """Without torch the context is a no-op."""
        with patch.object(canary, "torch", None):
            self.assertIsInstance(getattr(canary, "_inference_mode")(), contextlib.nullcontext)
