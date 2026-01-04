import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import sys
import json

# Ensure modules can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

class TestTranslation(unittest.TestCase):
    def setUp(self):
        global translation, config, utils
        from modules import translation, config, utils
        config.TARGET_LANGUAGES = {"es": {"code": "spa_Latn", "label": "Spanish"}}

    @patch("modules.translation.utils.register_subprocess")
    @patch("modules.translation.utils.unregister_subprocess")
    @patch("modules.translation.utils.save_translated_srt")
    @patch("subprocess.Popen")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_handle_pivot_pass_success(self, mock_remove, mock_exists, mock_popen, mock_save_srt, mock_unreg, mock_reg):
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0
        mock_popen.return_value = mock_proc
        mock_exists.return_value = True

        # Ensure TARGET_LANGUAGES has "en" for pivot pass
        config.TARGET_LANGUAGES = {
            "en": {"code": "eng_Latn", "label": "English"},
            "es": {"code": "spa_Latn", "label": "Spanish"}
        }

        source_data = [{"text": "Hello", "start": 0, "end": 1}]
        # Data for json.load
        m_open = mock_open(read_data='["Hello Translated"]')

        with patch("builtins.open", m_open):
            new_data, new_code = translation._handle_pivot_pass(
                source_data, "es", "folder", "base", ["en"], "spa_Latn", []
            )

            self.assertEqual(new_code, "eng_Latn")
            self.assertEqual(new_data[0]["text"], "Hello Translated")
            mock_save_srt.assert_called()

    @patch("modules.translation.utils.validate_srt", return_value=False)
    @patch("os.path.exists", return_value=True)
    def test_identify_missing_targets_invalid_srt(self, mock_exists, mock_validate):
        config.TARGET_LANGUAGES = {"es": {"code": "spa", "label": "Esp"}}
        missing, skipped = translation._identify_missing_targets("en", "folder", "base")
        self.assertEqual(len(missing), 1)
        self.assertEqual(skipped, 0)

    @patch("modules.translation.utils.validate_srt", return_value=True)
    @patch("os.path.exists", return_value=True)
    def test_identify_missing_targets_skipped(self, mock_exists, mock_validate):
        config.TARGET_LANGUAGES = {"es": {"code": "spa", "label": "Esp"}}
        missing, skipped = translation._identify_missing_targets("en", "folder", "base")
        self.assertEqual(len(missing), 0)
        self.assertEqual(skipped, 1)

    @patch("modules.translation.subprocess.Popen")
    @patch("modules.translation.utils.save_translated_srt")
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
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        # Mock segments
        seg1 = MagicMock()
        seg1.text = "Hello"
        seg1.start = 0.0
        seg1.end = 1.0
        segments = [seg1]

        # Mock file existence:
        # First calls (checking inputs/manifests) -> True or False
        # Loop calls (checking outputs) -> True eventually
        def exists_side_effect(path):
            if "temp_output" in path:
                # Simulate file appearing after some polls
                if poll_se.counter >= 2:
                    return True
                return False
            # Default to True for other files (like common input) or False if checking target existence check
            return False

        mock_exists.side_effect = exists_side_effect

        # Mock reading the output file
        # We need mock_open to handle read of JSON correctly
        # The file content must be a JSON list of matches segs
        fake_translation = [{"text": "Hola", "start": 0.0, "end": 1.0}]
        fake_json = json.dumps(fake_translation)

        m_open = mock_open(read_data=fake_json)

        with patch("builtins.open", m_open):
            translation.translate_segments(
                segments, "en", MagicMock(), "folder", "base"
            )

        # Should have called Popen (worker start)
        mock_popen.assert_called()
        # Should have saved SRT (because output file "appeared")
        mock_save.assert_called()


if __name__ == "__main__":
    unittest.main()
