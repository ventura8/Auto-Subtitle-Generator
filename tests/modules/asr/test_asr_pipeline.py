"""Tests for the batched NVIDIA ASR span pipeline in modules.asr.pipeline."""

import unittest
from unittest.mock import MagicMock, patch

from modules.asr import pipeline
from modules.asr.common import DecodedClip
from modules.asr.cues import DEFAULT_LIMITS


class _FakeSource:
    """PCM source whose clips are the span tuples themselves, so calls are easy to trace."""

    def __init__(self):
        self.reads = []

    def read(self, start, end):
        """Record and return the requested span."""
        self.reads.append((start, end))
        return [start, end]

    def close(self):
        """Nothing to release."""


class _FakeModel:
    """ASR wrapper stand-in that answers each clip from a dict keyed by span start."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def transcribe_batch(self, clips, language):
        """Return the scripted result of every clip."""
        self.calls.append(([clip[0] for clip in clips], language))
        return [self.answers[clip[0]] for clip in clips]

    def release(self):
        """Nothing to release."""


def _canary(text, logprob=-0.1, degenerate=False):
    """Canary-style result: no stamps, a mean log-probability."""
    return DecodedClip(text, None, logprob, degenerate)


def _parakeet(stamps, text):
    """Parakeet-style result: token stamps, no log-probability."""
    return DecodedClip(text, stamps, None, False)


def _request(language="ro", batch_size=2):
    """Build a request with the default cue limits."""
    return pipeline.AsrRequest(language, batch_size, DEFAULT_LIMITS)


class TestTranscribe(unittest.TestCase):
    def setUp(self):
        patcher = patch("modules.asr.pipeline.print_progress_bar")
        self.progress = patcher.start()
        self.addCleanup(patcher.stop)
        pauses = patch("modules.asr.pipeline.internal_pauses", return_value=[])
        self.pauses = pauses.start()
        self.addCleanup(pauses.stop)

    def test_no_spans_never_calls_the_model(self):
        model = _FakeModel({})
        result = pipeline.transcribe(model, _FakeSource(), [], _request())
        self.assertEqual(result, pipeline.AsrResult([], 0, 0.0, {}))
        self.assertEqual(model.calls, [])
        self.progress.assert_not_called()

    def test_batches_longest_first_and_cues_come_back_in_time_order(self):
        spans = [(0.0, 2.0), (3.0, 9.0), (10.0, 14.0)]
        model = _FakeModel({0.0: _canary("Unu."), 3.0: _canary("Doi."), 10.0: _canary("Trei.")})
        result = pipeline.transcribe(model, _FakeSource(), spans, _request(batch_size=2))
        self.assertEqual(model.calls, [([3.0, 10.0], "ro"), ([0.0], "ro")])
        self.assertEqual([cue[2] for cue in result.cues], ["Unu.", "Doi.", "Trei."])
        self.assertEqual(result.cues[1][:2], (3.0, 9.0))
        self.assertAlmostEqual(result.speech_seconds, 12.0)
        self.assertAlmostEqual(result.decoded_seconds, 12.0)
        self.assertEqual(self.progress.call_count, 2)
        self.assertEqual(self.progress.call_args.kwargs["prefix"], "  [ASR] Transcribing")

    def test_canary_cues_snap_to_internal_pauses(self):
        spans = [(5.0, 15.0)]
        model = _FakeModel({5.0: _canary("Prima propoziție aici. A doua propoziție urmează.")})
        self.pauses.return_value = [(9.9, 10.1)]
        result = pipeline.transcribe(model, _FakeSource(), spans, _request())
        self.pauses.assert_called_once_with([5.0, 15.0], 5.0)
        self.assertEqual(len(result.cues), 2)
        self.assertAlmostEqual(result.cues[0][1], 10.0)

    def test_parakeet_stamps_become_word_timed_cues_offset_by_span(self):
        stamps = [{"token": " Bună", "start": 0.0, "end": 0.4}, {"token": " ziua.", "start": 0.5, "end": 0.9}]
        model = _FakeModel({20.0: _parakeet(stamps, "Bună ziua.")})
        result = pipeline.transcribe(model, _FakeSource(), [(20.0, 21.0)], _request(language="en"))
        self.assertEqual(result.cues, [(20.0, 20.9, "Bună ziua.")])
        self.pauses.assert_not_called()

    def test_romanian_cedilla_is_folded_on_both_cue_paths(self):
        stamps = [{"token": " Aşa", "start": 0.0, "end": 0.4}, {"token": " ţară.", "start": 0.5, "end": 0.9}]
        model = _FakeModel({0.0: _parakeet(stamps, "Aşa ţară."), 2.0: _canary("Şi ţie.")})
        result = pipeline.transcribe(model, _FakeSource(), [(0.0, 1.0), (2.0, 3.0)], _request())
        self.assertEqual([cue[2] for cue in result.cues], ["Așa țară.", "Și ție."])

    def test_collapsed_repeats_fall_back_to_text_cues(self):
        text = "da da da da da"
        stamps = [{"token": " da", "start": index * 0.2, "end": index * 0.2 + 0.1} for index in range(5)]
        model = _FakeModel({0.0: _parakeet(stamps, text)})
        result = pipeline.transcribe(model, _FakeSource(), [(0.0, 4.0)], _request(language="en"))
        self.assertEqual([cue[2] for cue in result.cues], ["da"])
        self.pauses.assert_called_once()

    def test_filters_are_counted_and_not_decoded(self):
        answers = {
            0.0: _canary("   "),
            2.0: _canary("Bucla", degenerate=True),
            4.0: _canary("ha " * 40),
            6.0: _canary("Aceasta este o propoziție lungă rostită mult prea repede."),
            8.0: _canary("Nu sunt sigur deloc de asta.", logprob=-2.0),
            10.0: _canary("Poate da.", logprob=-1.2),
            12.0: _canary("Bine.", logprob=-0.9),
        }
        spans = [(start, start + 1.9) for start in sorted(answers)]
        result = pipeline.transcribe(_FakeModel(answers), _FakeSource(), spans, _request(batch_size=8))
        expected = {"empty": 1, "runaway": 1, "repetitive": 1, "too fast": 1, "low confidence": 2}
        self.assertEqual(result.dropped, expected)
        self.assertEqual([cue[2] for cue in result.cues], ["Bine."])
        self.assertAlmostEqual(result.decoded_seconds, 1.9)

    def test_oom_bisection_wraps_the_model_call(self):
        model = MagicMock()
        model.transcribe_batch.return_value = [_canary("Salut.")]
        with patch("modules.asr.pipeline.common.generate_with_oom_bisection", wraps=pipeline.common.generate_with_oom_bisection) as bisect:
            pipeline.transcribe(model, _FakeSource(), [(0.0, 1.0)], _request(language="de"))
        bisect.assert_called_once()
        model.transcribe_batch.assert_called_once_with([[0.0, 1.0]], "de")


class TestBatchesAndGuard(unittest.TestCase):
    def test_length_batches_cover_every_index_once(self):
        spans = [(0.0, 1.0), (1.0, 5.0), (5.0, 7.0)]
        self.assertEqual(pipeline._length_batches(spans, 0), [[1], [2], [0]])
        self.assertEqual(pipeline._length_batches(spans, 10), [[1, 2, 0]])

    def test_guard_ignores_little_speech(self):
        pipeline.check_decoded_speech(pipeline.AsrResult([], 29.0, 0.0, {}))

    def test_guard_accepts_enough_decoded_speech(self):
        pipeline.check_decoded_speech(pipeline.AsrResult([], 100.0, 20.0, {}))

    def test_guard_raises_when_little_speech_decoded(self):
        with self.assertRaisesRegex(pipeline.AsrEngineFailed, "only 19 s of 100 s"):
            pipeline.check_decoded_speech(pipeline.AsrResult([], 100.0, 19.0, {}))

    def test_too_fast_needs_a_positive_duration(self):
        self.assertFalse(pipeline._too_fast("abc", 0.0))
        self.assertTrue(pipeline._too_fast("x" * 30, 1.0))


if __name__ == "__main__":
    unittest.main()
