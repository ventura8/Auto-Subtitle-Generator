from modules import translation
import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import sys

# Ensure modules can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


class TestCoverageTranslation(unittest.TestCase):

    @patch("modules.translation.open", new_callable=mock_open)
    @patch("subprocess.Popen")
    @patch("modules.utils.register_subprocess")
    @patch("modules.utils.unregister_subprocess")
    @patch("modules.translation.log")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_handle_pivot_pass_fail(self, mock_remove, mock_exists, mock_log, mock_unreg, mock_reg, mock_popen, mock_file):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        translation._handle_pivot_pass([], "ro", "folder", "base", ["en"], "ron_Latn", [])

        # Check that it was called with "Warning: Pivot pass failed"
        called = False
        for call in mock_log.call_args_list:
            if "Pivot pass failed" in str(call):
                called = True
                break
        self.assertTrue(called)

    @patch("modules.translation.open", new_callable=mock_open, read_data='["hello"]')
    @patch("subprocess.Popen")
    @patch("modules.utils.register_subprocess")
    @patch("modules.utils.unregister_subprocess")
    @patch("modules.utils.save_translated_srt")
    @patch("modules.translation.log")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_handle_pivot_pass_success_with_en_target(self, mock_remove,
                                                      mock_exists, mock_log,
                                                      mock_save, mock_unreg,
                                                      mock_reg, mock_popen,
                                                      mock_file):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0
        mock_popen.return_value = mock_proc

        missing = ["en", "fr"]
        res_data, res_code = translation._handle_pivot_pass(
            [{"text": "salut", "start": 0, "end": 1}], "ro", "folder", "base", missing, "ron_Latn", [MagicMock()])
        self.assertEqual(res_code, "eng_Latn")
        self.assertIn("fr", missing)
        self.assertNotIn("en", missing)
        mock_log.assert_any_call("  [Pivot] English target satisfied via pivot pass.")

    def test_process_completed_output_mismatch(self):
        with patch("modules.translation.open", mock_open(read_data='["line1"]')), \
                patch("modules.translation.log") as mock_log:
            res = translation._process_completed_output("file.json", "en", [MagicMock(), MagicMock()], "folder", "base")
            self.assertFalse(res)
            mock_log.assert_any_call("  [Error] Mismatch for en: 1 vs 2", "ERROR")

    @patch("modules.translation.open", new_callable=mock_open)
    @patch("subprocess.Popen")
    @patch("modules.utils.register_subprocess")
    @patch("modules.utils.unregister_subprocess")
    @patch("modules.translation._poll_translation_results")
    @patch("modules.translation.log")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_execute_translation_workers_orphaned(self, mock_remove, mock_exists,
                                                  mock_log, mock_poll, mock_unreg,
                                                  mock_reg, mock_popen, mock_file):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Still running
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        with patch("modules.config.TARGET_LANGUAGES", {"fr": {"code": "fra_Latn", "label": "French"}}):
            translation._execute_translation_workers(["fr"], [], "eng_Latn", "folder", "base", [])
        mock_log.assert_any_call("!   [Cleanup] Terminating orphaned translation worker...", "WARNING")

    def test_translate_segments_no_missing(self):
        with patch("modules.translation._identify_missing_targets", return_value=([], 0)):
            res = translation.translate_segments([], "en", MagicMock(), "folder", "base")
            self.assertEqual(res, {})

    def test_translate_segments_no_source_data(self):
        with patch("modules.translation._identify_missing_targets", return_value=(["fr"], 0)), \
                patch("modules.translation._prepare_source_data", return_value=[]):
            res = translation.translate_segments([], "en", MagicMock(), "folder", "base")
            self.assertEqual(res, {})


if __name__ == "__main__":
    unittest.main()
