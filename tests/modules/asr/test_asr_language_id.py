"""Tests for the spread language-ID vote in modules.asr.language_id."""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules.asr import language_id

# numpy is an ``ml``-group dependency the CI test jobs do not install.
_FAKE_NUMPY = SimpleNamespace(concatenate=lambda parts: list(itertools.chain.from_iterable(parts)))


class _MarkerSource:
    """PCM source whose reads return the requested ``(start, end)`` as a single marker sample."""

    @staticmethod
    def read(start, end):
        """Return a one-sample marker for the read."""
        return [(start, end)]

    @staticmethod
    def close():
        """Nothing to release."""


def _rounded(windows):
    """Round window piece times to milliseconds for comparison."""
    return [[(round(start, 3), round(end, 3)) for start, end in pieces] for pieces in windows]


def _detection(language, probabilities):
    """Build one ``WhisperModel.detect_language`` result."""
    return language, dict(probabilities)[language], probabilities


class TestNumpyAccessor(unittest.TestCase):
    def test_numpy_is_imported_lazily(self):
        with patch("modules.asr.language_id.importlib.import_module", return_value="numpy-module") as load:
            self.assertEqual(language_id._numpy(), "numpy-module")
        load.assert_called_once_with("numpy")


class TestPickLidWindows(unittest.TestCase):
    def test_no_speech_has_no_windows(self):
        self.assertEqual(language_id.pick_lid_windows([]), [])
        self.assertEqual(language_id.pick_lid_windows([(4.0, 4.0)]), [])

    def test_short_speech_is_one_window_of_all_pieces(self):
        self.assertEqual(language_id.pick_lid_windows([(0.0, 4.0), (10.0, 16.0)]), [[(0.0, 4.0), (10.0, 16.0)]])

    def test_windows_never_outnumber_the_speech(self):
        windows = language_id.pick_lid_windows([(0.0, 100.0)])
        self.assertEqual(_rounded(windows), [[(0.0, 30.0)], [(23.333, 53.333)], [(46.667, 76.667)], [(70.0, 100.0)]])

    def test_window_runs_across_consecutive_spans_in_speech_time(self):
        windows = language_id.pick_lid_windows([(0.0, 20.0), (50.0, 80.0)], count=1)
        self.assertEqual(windows, [[(0.0, 20.0), (50.0, 60.0)]])

    def test_windows_spread_to_the_end_of_the_speech(self):
        spans = [(index * 100.0, index * 100.0 + 50.0) for index in range(20)]
        windows = language_id.pick_lid_windows(spans)
        self.assertEqual(len(windows), language_id.LID_WINDOWS)
        self.assertEqual(windows[0], [(0.0, 30.0)])
        self.assertEqual(_rounded(windows[-1:]), [[(1920.0, 1950.0)]])
        for pieces in windows:
            self.assertAlmostEqual(sum(end - start for start, end in pieces), language_id.LID_WINDOW_SECONDS)


class TestVoteLanguage(unittest.TestCase):
    def test_no_scored_windows_is_no_vote(self):
        self.assertIsNone(language_id.vote_language([]))
        self.assertIsNone(language_id.vote_language([[], []]))

    def test_probabilities_are_summed_across_windows(self):
        vote = language_id.vote_language(
            [[("ro", 0.6), ("en", 0.4)], [("en", 0.7), ("ro", 0.3)], [], [("ro", 0.9), ("en", 0.1)]],
        )
        self.assertEqual(vote.language, "ro")
        self.assertAlmostEqual(vote.probability, 0.6)
        self.assertAlmostEqual(vote.share, 2 / 3)
        self.assertEqual(list(vote.distribution), ["ro", "en"])
        self.assertAlmostEqual(vote.distribution["en"], 0.4)

    def test_sum_beats_window_majority(self):
        vote = language_id.vote_language([[("en", 0.51), ("ro", 0.49)], [("en", 0.51), ("ro", 0.49)], [("ro", 0.99), ("en", 0.01)]])
        self.assertEqual(vote.language, "ro")
        self.assertAlmostEqual(vote.share, 1 / 3)


class TestDetectLanguage(unittest.TestCase):
    def setUp(self):
        self.source = _MarkerSource()
        patcher = patch("modules.asr.language_id.log")
        self.log = patcher.start()
        self.addCleanup(patcher.stop)
        numpy_patcher = patch("modules.asr.language_id._numpy", return_value=_FAKE_NUMPY)
        numpy_patcher.start()
        self.addCleanup(numpy_patcher.stop)

    def test_no_spans_is_no_vote(self):
        model = MagicMock()
        self.assertIsNone(language_id.detect_language(model, self.source, []))
        model.detect_language.assert_not_called()

    def test_windows_are_read_and_concatenated(self):
        model = MagicMock()
        model.detect_language.return_value = _detection("ro", [("ro", 0.9), ("en", 0.1)])
        vote = language_id.detect_language(model, self.source, [(0.0, 10.0), (20.0, 30.0)])
        self.assertEqual(vote.language, "ro")
        self.assertEqual(model.detect_language.call_args.kwargs, {"audio": [(0.0, 10.0), (20.0, 30.0)]})
        self.log.assert_not_called()

    def test_value_error_skips_a_window(self):
        model = MagicMock()
        model.detect_language.side_effect = [ValueError("max() arg is an empty sequence"), _detection("en", [("en", 0.8), ("ro", 0.2)])]
        vote = language_id.detect_language(model, self.source, [(0.0, 60.0)])
        self.assertEqual(vote.language, "en")
        self.assertEqual(vote.share, 1.0)
        self.assertEqual(self.log.call_args.args[1], "DEBUG")

    def test_every_window_failing_is_no_vote(self):
        model = MagicMock()
        model.detect_language.side_effect = ValueError("no speech")
        self.assertIsNone(language_id.detect_language(model, self.source, [(0.0, 60.0)]))

    def test_mixed_language_audio_warns(self):
        model = MagicMock()
        model.detect_language.side_effect = [
            _detection("ro", [("ro", 0.8), ("en", 0.15), ("fr", 0.05)]),
            _detection("en", [("en", 0.65), ("ro", 0.3), ("fr", 0.05)]),
        ]
        vote = language_id.detect_language(model, self.source, [(0.0, 60.0)])
        self.assertEqual(vote.language, "ro")
        message, level = self.log.call_args.args
        self.assertEqual(level, "WARNING")
        self.assertIn("Mixed-language audio (ro 55%, en 40%, fr 5%)", message)
        self.assertIn("50%", message)


if __name__ == "__main__":
    unittest.main()
