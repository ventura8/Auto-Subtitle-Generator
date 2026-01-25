from modules import isolated_translator
import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import sys

# Ensure modules can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


class TestCoverageIsolated(unittest.TestCase):

    def test_run_translation_worker_batch_size_zero(self):
        with patch("modules.isolated_translator.open", mock_open(read_data='[{"text": "hi", "end": 1}]')), \
                patch("modules.isolated_translator.ModelManager"), \
                patch("modules.isolated_translator.OPTIMIZER") as mock_opt, \
                patch("modules.isolated_translator.log"):
            mock_opt.config = {"nllb_batch": 5}
            # Should not crash and should use 5
            isolated_translator.run_translation_worker("in.json", "out.json", "en", "ro", 0, "Romanian", "prefix")

    def test_run_translation_worker_exception(self):
        with patch("modules.isolated_translator.open", mock_open(read_data='[{"text": "hi", "end": 1}]')), \
                patch("modules.isolated_translator.ModelManager") as mock_mm, \
                patch("modules.isolated_translator.log"):
            mock_mm.return_value.get_nllb.return_value.translate.side_effect = Exception("error")
            with self.assertRaises(Exception):
                isolated_translator.run_translation_worker("in.json", "out.json", "en", "ro", 1, "Romanian", "prefix")

    def test_translate_batch_chunk_error(self):
        translator = MagicMock()
        translator.translate.side_effect = Exception("fail")
        with patch("modules.isolated_translator.log") as mock_log:
            res = isolated_translator._translate_batch_chunk(translator, ["t1"], "en", "ro")
            self.assertEqual(res, ["Translation Error"])
            mock_log.assert_called()

    def test_process_single_job_padding(self):
        translator = MagicMock()
        translator.translate.return_value = ["res1"]  # only 1 result for 2 texts
        job = {"lang": "fr", "tgt_code": "fra_Latn", "input": "in.json", "output": "out.json"}
        with patch("modules.isolated_translator.open", mock_open(read_data='[{"text": "t1", "end": 1}, {"text": "t2", "end": 2}]')), \
                patch("modules.isolated_translator.OPTIMIZER") as mock_opt, \
                patch("modules.isolated_translator.log"), \
                patch("os.path.exists", return_value=False), \
                patch("os.rename"), \
                patch("modules.isolated_translator._translate_batch_chunk", return_value=["res1"]):
            mock_opt.config = {"nllb_batch": 10}
            isolated_translator._process_single_job(job, 0, 1, translator)

    def test_run_batch_translation_worker_no_jobs(self):
        with patch("modules.isolated_translator.open", mock_open(read_data='{"jobs": []}')), \
                patch("modules.isolated_translator.log") as mock_log:
            isolated_translator.run_batch_translation_worker("manifest.json")
            mock_log.assert_any_call("[Isolation] No jobs in manifest. Exiting.")

    def test_main_usage(self):
        with patch("sys.argv", ["script.py"]), \
                patch("sys.exit") as mock_exit, \
                patch("builtins.print") as mock_print:
            isolated_translator.main()
            mock_exit.assert_called_with(1)
            mock_print.assert_called()

    def test_main_batch_mode(self):
        with patch("sys.argv", ["script.py", "--batch", "manifest.json"]), \
                patch("modules.isolated_translator.run_batch_translation_worker") as mock_run, \
                patch("sys.exit", side_effect=SystemExit) as mock_exit:
            with self.assertRaises(SystemExit):
                isolated_translator.main()
            mock_run.assert_called_with("manifest.json")
            mock_exit.assert_called_with(0)

    def test_main_step_args(self):
        with patch("sys.argv", ["script.py", "in.json", "out.json", "en", "ro", "8", "Romanian", "1", "4"]), \
                patch("modules.isolated_translator.run_translation_worker") as mock_run:
            isolated_translator.main()
            mock_run.assert_called()
            # prefix check
            args = mock_run.call_args[0]
            self.assertIn("1/4", args[-1])

    def test_main_fatal_error(self):
        with patch("sys.argv", ["script.py", "in.json", "out.json", "en", "ro", "8", "Romanian"]), \
                patch("modules.isolated_translator.run_translation_worker", side_effect=Exception("fatal")), \
                patch("modules.isolated_translator.log") as mock_log, \
                patch("traceback.print_exc"), \
                patch("sys.exit") as mock_exit:
            isolated_translator.main()
            mock_log.assert_any_call("[Isolation] FATAL ERROR: fatal")
            mock_exit.assert_called_with(1)


if __name__ == "__main__":
    unittest.main()
