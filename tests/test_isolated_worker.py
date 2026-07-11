import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import sys
import importlib

# Ensure modules can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

isolated_translator = importlib.import_module("modules.isolated_translator")


class TestCoverageIsolated(unittest.TestCase):
    def test_cleanup_intermediate_memory_with_cuda(self):
        with patch("modules.isolated_translator.gc.collect") as mock_gc, patch("modules.isolated_translator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            isolated_translator._cleanup_intermediate_memory(True)
            mock_gc.assert_called_once()
            mock_torch.cuda.empty_cache.assert_called_once()

    def test_wait_for_parent_timeout(self):
        time_values = [0.0, 11.0, 11.2]
        with (
            patch("os.path.exists", return_value=True),
            patch("time.time", side_effect=time_values),
            patch("time.sleep"),
            patch("modules.isolated_translator.log") as mock_log,
        ):
            isolated_translator._wait_for_parent_to_consume_output("out.json", "ro")
            mock_log.assert_called()

    def test_run_translation_worker_batch_size_zero(self):
        translator = MagicMock()
        translator.translate.side_effect = lambda batch_texts, *_args: [f"out:{text}" for text in batch_texts]
        test_data = [
            {"text": "t1", "start": 0, "end": 1},
            {"text": "t2", "start": 1, "end": 2},
            {"text": "t3", "start": 2, "end": 3},
            {"text": "t4", "start": 3, "end": 4},
            {"text": "t5", "start": 4, "end": 5},
            {"text": "t6", "start": 5, "end": 6},
            {"text": "t7", "start": 6, "end": 7},
        ]
        with (
            patch("modules.isolated_translator._load_segments", return_value=test_data),
            patch("modules.isolated_translator.ModelManager") as mock_mm,
            patch("modules.isolated_translator._save_worker_output"),
            patch("modules.isolated_translator.OPTIMIZER") as mock_opt,
            patch("modules.isolated_translator.log"),
        ):
            mock_opt.config = {"nllb_batch": 5}
            mock_mm.return_value.get_nllb.return_value = translator
            isolated_translator.run_translation_worker("in.json", "out.json", "en", "ro", 0, "Romanian", "prefix")
            self.assertEqual(translator.translate.call_count, 2)
            self.assertEqual(translator.translate.call_args_list[0][0][0], ["t1", "t2", "t3", "t4", "t5"])
            self.assertEqual(translator.translate.call_args_list[1][0][0], ["t6", "t7"])

    def test_run_translation_worker_exception(self):
        with (
            patch("modules.isolated_translator.open", mock_open(read_data='[{"text": "hi", "start": 0, "end": 1}]')),
            patch("modules.isolated_translator.ModelManager") as mock_mm,
            patch("modules.isolated_translator._save_worker_output") as mock_save,
            patch("modules.isolated_translator.log"),
        ):
            mock_mm.return_value.get_nllb.return_value.translate.side_effect = RuntimeError("error")
            isolated_translator.run_translation_worker("in.json", "out.json", "en", "ro", 1, "Romanian", "prefix")
            mock_save.assert_called_once_with("out.json", ["Translation Error"])

    def test_translate_batch_chunk_error(self):
        translator = MagicMock()
        translator.translate.side_effect = RuntimeError("fail")
        with patch("modules.isolated_translator.log") as mock_log:
            res = isolated_translator._translate_batch_chunk(translator, ["t1"], "en", "ro")
            self.assertEqual(res, ["Translation Error"])
            mock_log.assert_called()

    def test_process_single_job_padding(self):
        translator = MagicMock()
        translator.translate.return_value = ["res1"]  # only 1 result for 2 texts
        job = {"lang": "fr", "tgt_code": "fra_Latn", "input": "in.json", "output": "out.json"}
        with (
            patch("modules.isolated_translator.open", mock_open(read_data='[{"text": "t1", "end": 1}, {"text": "t2", "end": 2}]')),
            patch("modules.isolated_translator.OPTIMIZER") as mock_opt,
            patch("modules.isolated_translator.log"),
            patch("os.path.exists", return_value=False),
            patch("os.replace") as mock_replace,
            patch("modules.isolated_translator.json.dump") as mock_dump,
            patch("modules.isolated_translator._translate_batch_chunk", return_value=["res1"]),
        ):
            mock_opt.config = {"nllb_batch": 10}
            isolated_translator._process_single_job(job, 0, 1, translator)
            saved_translations = mock_dump.call_args[0][0]
            self.assertEqual(saved_translations, ["res1", "Translation Error"])
            mock_replace.assert_called_once()

    def test_run_batch_translation_worker_no_jobs(self):
        with (
            patch("modules.isolated_translator.open", mock_open(read_data='{"jobs": []}')),
            patch("modules.isolated_translator.log") as mock_log,
        ):
            isolated_translator.run_batch_translation_worker("manifest.json")
            mock_log.assert_any_call("[Isolation] No jobs in manifest. Exiting.")

    def test_main_usage(self):
        with patch("sys.argv", ["script.py"]), patch("sys.exit") as mock_exit, patch("builtins.print") as mock_print:
            isolated_translator.main()
            mock_exit.assert_called_with(1)
            mock_print.assert_called()

    def test_main_batch_mode(self):
        with (
            patch("sys.argv", ["script.py", "--batch", "manifest.json"]),
            patch("modules.isolated_translator.run_batch_translation_worker") as mock_run,
            patch("sys.exit", side_effect=SystemExit) as mock_exit,
        ):
            with self.assertRaises(SystemExit):
                isolated_translator.main()
            mock_run.assert_called_with("manifest.json")
            mock_exit.assert_called_with(0)

    def test_main_step_args(self):
        with (
            patch("sys.argv", ["script.py", "in.json", "out.json", "en", "ro", "8", "Romanian", "1", "4"]),
            patch("modules.isolated_translator.run_translation_worker") as mock_run,
        ):
            isolated_translator.main()
            mock_run.assert_called()
            # prefix check
            args = mock_run.call_args[0]
            self.assertIn("1/4", args[-1])

    def test_main_fatal_error(self):
        with (
            patch("sys.argv", ["script.py", "in.json", "out.json", "en", "ro", "8", "Romanian"]),
            patch("modules.isolated_translator.run_translation_worker", side_effect=RuntimeError("fatal")),
            patch("modules.isolated_translator.log") as mock_log,
            patch("traceback.print_exc"),
            patch("sys.exit") as mock_exit,
        ):
            isolated_translator.main()
            mock_log.assert_any_call("[Isolation] FATAL ERROR: fatal")
            mock_exit.assert_called_with(1)

    def test_main_catches_unexpected_exception(self):
        with (
            patch("sys.argv", ["script.py", "in.json", "out.json", "en", "ro", "8", "Romanian"]),
            patch("modules.isolated_translator._run_legacy_mode", side_effect=KeyError("unexpected")),
            patch("modules.isolated_translator.log") as mock_log,
            patch("traceback.print_exc"),
            patch("sys.exit") as mock_exit,
        ):
            isolated_translator.main()
            self.assertTrue(any("FATAL ERROR" in str(call.args[0]) for call in mock_log.call_args_list if call.args))
            mock_exit.assert_called_with(1)

    def test_process_single_job_handles_load_error(self):
        job = {"lang": "fr", "tgt_code": "fra_Latn", "input": "in.json", "output": "out.json"}
        with (
            patch("modules.isolated_translator._load_segments", side_effect=ValueError("bad json")),
            patch("modules.isolated_translator.log") as mock_log,
        ):
            isolated_translator._process_single_job(job, 0, 1, MagicMock())
            mock_log.assert_any_call("[Isolation] Job fr failed: bad json", "ERROR")

    def test_process_single_job_handles_missing_required_field(self):
        job = {"lang": "fr", "input": "in.json", "output": "out.json"}
        with patch("modules.isolated_translator.log") as mock_log:
            isolated_translator._process_single_job(job, 0, 1, MagicMock())
            self.assertTrue(any("Job fr failed" in str(call.args[0]) for call in mock_log.call_args_list if call.args))

    def test_run_pivot_phase_writes_segment_dicts(self):
        pivot_job = {
            "input": "pivot_input.json",
            "output": "pivot_output.json",
            "src_code": "ron_Latn",
            "tgt_code": "eng_Latn",
            "emit_en_output": True,
            "en_output": "en_output.json",
        }
        source_data = [{"text": "salut", "start": 0.0, "end": 1.0}]
        translations = ["hello"]

        with (
            patch("modules.isolated_translator._load_segments", return_value=source_data),
            patch("modules.isolated_translator._run_job_batches", return_value=translations),
            patch("modules.isolated_translator._save_job_translations") as mock_save_en,
            patch("modules.isolated_translator.tempfile.mkstemp", return_value=(99, "pivot_output.tmp")),
            patch("modules.isolated_translator.os.close") as mock_close,
            patch("modules.isolated_translator.os.replace") as mock_replace,
            patch("modules.isolated_translator.open", mock_open()),
            patch("modules.isolated_translator.json.dump") as mock_dump,
            patch("modules.isolated_translator.log"),
        ):
            isolated_translator._run_pivot_phase(pivot_job, MagicMock())

            dumped_payload = mock_dump.call_args[0][0]
            self.assertEqual(dumped_payload, [{"text": "hello", "start": 0.0, "end": 1.0}])
            mock_close.assert_called_once_with(99)
            mock_replace.assert_called_once_with("pivot_output.tmp", "pivot_output.json")
            mock_save_en.assert_called_once_with("en_output.json", ["hello"], source_data)

    def test_run_batch_translation_worker_continues_when_pivot_fails(self):
        manifest = {
            "jobs": [{"lang": "es", "tgt_code": "spa_Latn", "input": "in.json", "output": "out.json"}],
            "pivot": {"input": "pivot_input.json", "output": "pivot_output.json", "src_code": "ron_Latn", "tgt_code": "eng_Latn"},
        }
        with (
            patch("modules.isolated_translator.open", mock_open(read_data="{}")),
            patch("modules.isolated_translator.json.load", return_value=manifest),
            patch("modules.isolated_translator.ModelManager") as mock_mm,
            patch("modules.isolated_translator._load_segments", side_effect=RuntimeError("pivot boom")),
            patch("modules.isolated_translator._process_single_job") as mock_process_job,
            patch("modules.isolated_translator.log"),
        ):
            mock_mm.return_value.get_nllb.return_value = MagicMock()
            isolated_translator.run_batch_translation_worker("manifest.json")
            mock_process_job.assert_called_once()

    def test_run_pivot_phase_logs_failure_without_raising(self):
        pivot_job = {
            "input": "pivot_input.json",
            "output": "pivot_output.json",
            "src_code": "ron_Latn",
            "tgt_code": "eng_Latn",
        }
        with (
            patch("modules.isolated_translator._load_segments", side_effect=RuntimeError("pivot fail")),
            patch("modules.isolated_translator.log") as mock_log,
        ):
            isolated_translator._run_pivot_phase(pivot_job, MagicMock())
            self.assertTrue(any("Pivot phase failed" in str(call.args[0]) for call in mock_log.call_args_list if call.args))


if __name__ == "__main__":
    unittest.main()
