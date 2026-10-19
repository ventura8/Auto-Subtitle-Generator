"""Tests that translation never reuses a pivot or target SRT made for an older transcription."""

import contextlib
import os
import unittest
from unittest.mock import MagicMock, patch

from modules.configuration import config
from modules.models import Segment
from modules.pipeline import translation

SOURCE = [{"text": "Unu", "start": 0.0, "end": 1.2344}, {"text": "Doi", "start": 2.0, "end": 3.5}]


def _cues(*timings):
    """Parsed SRT cues with the given (start, end) timings."""
    return [Segment(start, end, "text") for start, end in timings]


class TestSameTimings(unittest.TestCase):
    def test_millisecond_equal_timings_match(self):
        self.assertTrue(translation._same_timings(_cues((0.0, 1.234), (2.0, 3.5)), SOURCE))

    def test_count_or_timing_difference_is_stale(self):
        self.assertFalse(translation._same_timings(_cues((0.0, 1.234)), SOURCE))
        self.assertFalse(translation._same_timings(_cues((0.0, 1.234), (2.0, 3.6)), SOURCE))

    def test_unchecked_without_source_data(self):
        with patch("modules.pipeline.translation.utils.parse_srt") as parse:
            self.assertFalse(translation._is_stale_srt("x.srt", None))
        parse.assert_not_called()


class TestStaleTargets(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(config, "TARGET_LANGUAGES", {"es": {"code": "spa_Latn", "label": "Spanish"}})
        patcher.start()
        self.addCleanup(patcher.stop)
        for target, value in (
            ("os.path.exists", True),
            ("modules.pipeline.translation.utils.validate_srt", True),
            ("modules.pipeline.translation.bind_input", contextlib.nullcontext(MagicMock(name="bound"))),
        ):
            started = patch(target, return_value=value)
            started.start()
            self.addCleanup(started.stop)

    def test_matching_target_is_skipped(self):
        with patch("modules.pipeline.translation.utils.parse_srt", return_value=_cues((0.0, 1.234), (2.0, 3.5))):
            missing, skipped = translation._identify_missing_targets("ro", "folder", "base", True, SOURCE)
        self.assertEqual((missing, skipped), ([], 1))

    def test_stale_target_is_redone_with_a_warning(self):
        with (
            patch("modules.pipeline.translation.utils.parse_srt", return_value=_cues((0.0, 1.0))),
            patch("modules.pipeline.translation.log") as mock_log,
        ):
            missing, skipped = translation._identify_missing_targets("ro", "folder", "base", True, SOURCE)
        self.assertEqual((missing, skipped), (["es"], 0))
        mock_log.assert_any_call(
            "  [Translate] es subtitles do not match the current source timings (re-transcribed?). Re-doing.", "WARNING"
        )

    def test_verify_bound_target_srt_when_bind_is_none(self):
        with (
            patch("modules.pipeline.translation.bind_input", return_value=contextlib.nullcontext(None)),
            patch("modules.pipeline.translation.log") as mock_log,
        ):
            missing, skipped = translation._verify_bound_target_srt("target.srt", "es", SOURCE)
        self.assertEqual((missing, skipped), (True, False))
        self.assertIn("Could not bind es SRT", mock_log.call_args[0][0])

    def test_verify_bound_target_srt_when_bind_raises(self):
        with (
            patch("modules.pipeline.translation.bind_input", side_effect=OSError("bind fail")),
            patch("modules.pipeline.translation.log") as mock_log,
        ):
            missing, skipped = translation._verify_bound_target_srt("target.srt", "es", SOURCE)
        self.assertEqual((missing, skipped), (True, False))
        self.assertIn("Could not bind es SRT", mock_log.call_args[0][0])

    def test_classify_target_language_state_when_not_exists(self):
        with patch("os.path.exists", return_value=False):
            missing, skipped = translation._classify_target_language_state("es", "folder", "base", SOURCE)
        self.assertEqual((missing, skipped), (True, False))


class TestStalePivot(unittest.TestCase):
    def setUp(self):
        for target, value in (
            ("os.path.exists", True),
            ("modules.pipeline.translation.utils.validate_srt", True),
            ("modules.pipeline.translation.bind_input", contextlib.nullcontext(MagicMock(name="bound"))),
        ):
            started = patch(target, return_value=value)
            started.start()
            self.addCleanup(started.stop)

    def test_matching_pivot_is_reused(self):
        cues = [Segment(0.0, 1.234, "One"), Segment(2.0, 3.5, "Two")]
        with patch("modules.pipeline.translation.utils.parse_srt", return_value=cues):
            data = translation._load_reusable_pivot_srt_data("folder", "base", SOURCE)
        self.assertEqual(data, [{"text": "One", "start": 0.0, "end": 1.234}, {"text": "Two", "start": 2.0, "end": 3.5}])

    def test_stale_pivot_is_ignored_with_a_warning(self):
        with (
            patch("modules.pipeline.translation.utils.parse_srt", return_value=_cues((0.0, 1.234))),
            patch("modules.pipeline.translation.log") as mock_log,
        ):
            self.assertIsNone(translation._load_reusable_pivot_srt_data("folder", "base", SOURCE))
        mock_log.assert_called_once_with(
            "  [Translate] English pivot SRT does not match the current source timings; translating a new pivot.", "WARNING"
        )

    def test_pivot_config_checks_against_the_source_data(self):
        context = {"src_lang": "ro", "src_code": "ron_Latn", "folder": "folder", "base_name": "base", "missing_langs": ["fr"]}
        with patch("modules.pipeline.translation._load_reusable_pivot_srt_data", return_value=None) as load:
            translation._build_pivot_config({**context, "source_data": SOURCE}, "input.json", [])
        load.assert_called_once_with("folder", "base", SOURCE)

    def test_read_bound_pivot_srt_when_bind_is_none(self):
        with (
            patch("modules.pipeline.translation.bind_input", return_value=contextlib.nullcontext(None)),
            patch("modules.pipeline.translation.log") as mock_log,
        ):
            res = translation._read_bound_pivot_srt("pivot.srt", SOURCE)
        self.assertIsNone(res)
        self.assertIn("Could not bind English pivot SRT", mock_log.call_args[0][0])

    def test_read_bound_pivot_srt_when_bind_raises(self):
        with (
            patch("modules.pipeline.translation.bind_input", side_effect=OSError("bind error")),
            patch("modules.pipeline.translation.log") as mock_log,
        ):
            res = translation._read_bound_pivot_srt("pivot.srt", SOURCE)
        self.assertIsNone(res)
        self.assertIn("Could not bind English pivot SRT", mock_log.call_args[0][0])

    def test_read_bound_pivot_srt_when_invalid_srt(self):
        with patch("modules.pipeline.translation.utils.validate_srt", return_value=False):
            res = translation._read_bound_pivot_srt("pivot.srt", SOURCE)
        self.assertIsNone(res)

    def test_load_reusable_pivot_srt_data_when_not_exists(self):
        with patch("os.path.exists", return_value=False):
            res = translation._load_reusable_pivot_srt_data("folder", "base", SOURCE)
        self.assertIsNone(res)


class TestTranslateSegmentsOrder(unittest.TestCase):
    def test_source_data_reaches_the_target_scan_and_asr_is_offloaded(self):
        segments = [Segment(0.0, 1.0, "Bună"), Segment(1.0, 2.0, "  ")]
        model_mgr = MagicMock()
        with (
            patch("modules.pipeline.translation._identify_missing_targets", return_value=(["en"], 0)) as identify,
            patch("modules.pipeline.translation._execute_translation_workers") as execute,
            patch("modules.pipeline.translation.log"),
        ):
            translation.translate_segments(segments, "ro", model_mgr, {"folder": "folder", "base_name": "base"})
        identify.assert_called_once_with("ro", "folder", "base", True, [{"text": "Bună", "start": 0.0, "end": 1.0}])
        model_mgr.offload_asr.assert_called_once_with()
        self.assertEqual(execute.call_args.args[0]["segments"], [segments[0]])

    def test_empty_source_skips_the_timing_check(self):
        with patch("modules.pipeline.translation._identify_missing_targets", return_value=([], 0)) as identify:
            translation.translate_segments([], "ro", None, {"folder": os.curdir, "base_name": "base"})
        self.assertIsNone(identify.call_args.args[4])


if __name__ == "__main__":
    unittest.main()
