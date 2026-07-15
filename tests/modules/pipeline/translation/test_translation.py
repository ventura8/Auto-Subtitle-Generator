import json
import unittest
from unittest.mock import MagicMock, mock_open, patch


class TestTranslation(unittest.TestCase):
    def setUp(self):
        global translation, config, utils
        from modules import utils
        from modules.configuration import config
        from modules.pipeline import translation

        previous_target_languages = config.TARGET_LANGUAGES
        self.addCleanup(setattr, config, "TARGET_LANGUAGES", previous_target_languages)
        config.TARGET_LANGUAGES = {"es": {"code": "spa_Latn", "label": "Spanish"}}

    @patch("modules.pipeline.translation.utils.validate_srt", return_value=False)
    @patch("os.path.exists", return_value=True)
    def test_identify_missing_targets_invalid_srt(self, mock_exists, mock_validate):
        config.TARGET_LANGUAGES = {"es": {"code": "spa", "label": "Esp"}}
        missing, skipped = translation._identify_missing_targets("en", "folder", "base")
        self.assertEqual(len(missing), 1)
        self.assertEqual(skipped, 0)

    @patch("modules.pipeline.translation.utils.validate_srt", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_identify_missing_targets_skipped(self, mock_exists, mock_validate):
        config.TARGET_LANGUAGES = {"es": {"code": "spa", "label": "Esp"}}
        missing, skipped = translation._identify_missing_targets("en", "folder", "base")
        self.assertEqual(len(missing), 0)
        self.assertEqual(skipped, 1)

    @patch("modules.pipeline.translation.subprocess.Popen")
    @patch("modules.pipeline.translation.utils.save_translated_srt")
    @patch("os.path.exists")
    @patch("os.remove")
    @patch("time.sleep")
    def test_translate_segments_worker_flow(self, mock_sleep, mock_remove, mock_exists, mock_save, mock_popen):
        # Test the orchestrator: translate_segments -> _execute_translation_workers

        # Mock process
        mock_proc = MagicMock()

        # Dynamic side effect for poll: None (running) -> None -> 0 (finished)
        def poll_se():
            poll_se.counter += 1
            if poll_se.counter > 3:
                return 0
            return None

        poll_se.counter = 0
        mock_proc.poll.side_effect = poll_se
        mock_proc.returncode = 0

        class FakeTimeoutExpired(TimeoutError):
            pass

        def wait_se(*args, **kwargs):
            wait_se.counter += 1
            if "timeout" in kwargs and wait_se.counter == 1:
                raise FakeTimeoutExpired(f"worker timeout ({kwargs['timeout']})")
            return 0

        wait_se.counter = 0
        mock_proc.wait.side_effect = wait_se
        mock_popen.return_value = mock_proc
        mock_popen.return_value.__enter__.return_value = mock_proc

        # Mock segments
        seg1 = MagicMock()
        seg1.text = "Hello"
        seg1.start = 0.0
        seg1.end = 1.0
        segments = [seg1]

        # Mock file existence:
        # First calls (checking inputs/manifests) -> True or False
        # Loop calls (checking outputs) -> True eventually
        temp_output_checks = {"count": 0}

        def exists_side_effect(path):
            if "temp_output" in path:
                temp_output_checks["count"] += 1
                return temp_output_checks["count"] >= 3
            # Default to True for other files (like common input) or False if checking target existence check
            return False

        mock_exists.side_effect = exists_side_effect

        # Mock reading the output file
        # We need mock_open to handle read of JSON correctly
        # The file content must be a JSON list of matches segs
        fake_translation = [{"text": "Hola", "start": 0.0, "end": 1.0}]
        fake_json = json.dumps(fake_translation)

        m_open = mock_open(read_data=fake_json)

        with patch("builtins.open", m_open), patch("modules.pipeline.translation.subprocess.TimeoutExpired", FakeTimeoutExpired):
            translation.translate_segments(segments, "en", MagicMock(), "folder", "base")

        # Should have called Popen (worker start)
        mock_popen.assert_called()
        # Should have saved SRT (because output file "appeared")
        mock_save.assert_called()


if __name__ == "__main__":
    unittest.main()
