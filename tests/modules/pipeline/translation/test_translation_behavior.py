import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

from modules.pipeline import translation


class TestCoverageTranslation(unittest.TestCase):
    def test_identify_missing_targets_no_targets(self):
        with patch("modules.configuration.config.TARGET_LANGUAGES", {"en": {"code": "eng_Latn", "label": "English"}}):
            missing, skipped = translation._identify_missing_targets("en", "folder", "base")
            self.assertEqual(missing, [])
            self.assertEqual(skipped, 0)

    def test_process_completed_output_mismatch(self):
        with (
            patch("modules.pipeline.translation.open", mock_open(read_data='["line1"]')),
            patch("modules.pipeline.translation.log") as mock_log,
        ):
            res = translation._process_completed_output("file.json", "en", [MagicMock(), MagicMock()], "folder", "base")
            self.assertFalse(res)
            mock_log.assert_any_call("  [Error] Mismatch for en: 1 vs 2", "ERROR")

    def test_process_completed_output_read_failure_after_retries(self):
        with (
            patch("modules.pipeline.translation.open", side_effect=OSError("read failure")),
            patch("modules.pipeline.translation.time.sleep"),
            patch("modules.pipeline.translation.log") as mock_log,
        ):
            res = translation._process_completed_output("file.json", "en", [MagicMock()], "folder", "base")
            self.assertFalse(res)
            mock_log.assert_any_call(
                "  [Error] Failed to process realtime output for en: Could not read file.json",
                "ERROR",
            )

    def test_wait_worker_tick_with_invalid_timeout_expired_type(self):
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = TimeoutError("still running")
        with patch("modules.pipeline.translation.subprocess.TimeoutExpired", "invalid-type"):
            self.assertFalse(translation._wait_worker_tick(mock_proc))

    def test_scan_pending_outputs_keeps_language_when_output_missing(self):
        pending = {"fr"}
        with patch("os.path.exists", return_value=False):
            remaining = translation._scan_pending_outputs(pending, "folder", "base", [])
        self.assertEqual(remaining, {"fr"})

    def test_scan_pending_outputs_removes_language_on_success(self):
        pending = {"fr"}

        def exists_side_effect(path):
            return path.endswith(".temp_output.base.fr.json")

        with (
            patch("os.path.exists", side_effect=exists_side_effect),
            patch("modules.pipeline.translation._process_completed_output", return_value=True),
            patch("os.remove") as mock_remove,
        ):
            remaining = translation._scan_pending_outputs(pending, "folder", "base", [])

        self.assertEqual(remaining, set())
        mock_remove.assert_called_once()

    def test_scan_pending_outputs_removes_language_on_failure(self):
        pending = {"fr"}

        def exists_side_effect(path):
            return path.endswith(".temp_output.base.fr.json")

        with (
            patch("os.path.exists", side_effect=exists_side_effect),
            patch("modules.pipeline.translation._process_completed_output", return_value=False),
            patch("os.remove") as mock_remove,
        ):
            remaining = translation._scan_pending_outputs(pending, "folder", "base", [])

        self.assertEqual(remaining, {"fr"})
        mock_remove.assert_not_called()

    def test_poll_translation_results_final_pass_processes_remaining_file(self):
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, 0]

        with (
            patch("modules.pipeline.translation._wait_worker_tick", return_value=True),
            patch("modules.pipeline.translation._scan_pending_outputs", return_value={"fr"}),
            patch("os.path.exists", return_value=True),
            patch("modules.pipeline.translation._process_completed_output") as mock_process,
        ):
            translation._poll_translation_results(mock_proc, ["fr"], "folder", "base", [])

        mock_process.assert_called_once()

    def test_poll_translation_results_skips_final_processing_when_no_file(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0

        with patch("os.path.exists", return_value=False), patch("modules.pipeline.translation._process_completed_output") as mock_process:
            translation._poll_translation_results(mock_proc, ["fr"], "folder", "base", [])

        mock_process.assert_not_called()

    def test_build_pivot_config_for_english_source(self):
        worker_context = {
            "src_lang": "en",
            "src_code": "eng_Latn",
            "folder": "folder",
            "base_name": "base",
            "missing_langs": ["fr"],
        }
        temp_files = []
        pivot, source_code, input_file = translation._build_pivot_config(worker_context, "input.json", temp_files)
        self.assertIsNone(pivot)
        self.assertEqual(source_code, "eng_Latn")
        self.assertEqual(input_file, "input.json")

    def test_build_pivot_config_emits_en_output_when_needed(self):
        worker_context = {
            "src_lang": "ro",
            "src_code": "ron_Latn",
            "folder": "folder",
            "base_name": "base",
            "missing_langs": ["en", "fr"],
        }
        temp_files = []

        with patch("modules.configuration.config.TARGET_LANGUAGES", {"en": {"code": "eng_Latn", "label": "English"}}):
            pivot, source_code, input_file = translation._build_pivot_config(worker_context, "input.json", temp_files)

        self.assertIsNotNone(pivot)
        assert pivot is not None
        self.assertTrue(pivot["emit_en_output"])
        self.assertIn("en_output", pivot)
        self.assertEqual(source_code, "eng_Latn")
        self.assertEqual(input_file, pivot["output"])

    def test_build_pivot_config_without_en_target(self):
        worker_context = {
            "src_lang": "ro",
            "src_code": "ron_Latn",
            "folder": "folder",
            "base_name": "base",
            "missing_langs": ["fr"],
        }
        temp_files = []

        with patch("modules.configuration.config.TARGET_LANGUAGES", {"en": {"code": "eng_Latn", "label": "English"}}):
            pivot, source_code, input_file = translation._build_pivot_config(worker_context, "input.json", temp_files)

        self.assertIsNotNone(pivot)
        assert pivot is not None
        self.assertFalse(pivot["emit_en_output"])
        self.assertNotIn("en_output", pivot)
        self.assertEqual(source_code, "eng_Latn")
        self.assertEqual(input_file, pivot["output"])

    def test_build_pivot_config_reuses_existing_english_srt(self):
        worker_context = {
            "src_lang": "ro",
            "src_code": "ron_Latn",
            "folder": "folder",
            "base_name": "base",
            "missing_langs": ["fr"],
        }
        temp_files = []
        pivot_segments = [MagicMock(text="Hello", start=0.0, end=1.0)]

        with (
            patch("modules.configuration.config.TARGET_LANGUAGES", {"en": {"code": "eng_Latn", "label": "English"}}),
            patch("os.path.exists", return_value=True),
            patch("modules.pipeline.translation.utils.validate_srt", return_value=True),
            patch("modules.pipeline.translation.utils.parse_srt", return_value=pivot_segments),
            patch("modules.pipeline.translation.open", mock_open()) as mock_file,
            patch("modules.pipeline.translation.json.dump") as mock_dump,
            patch("modules.pipeline.translation.log") as mock_log,
        ):
            pivot, source_code, input_file = translation._build_pivot_config(worker_context, "input.json", temp_files)

        expected_path = os.path.join("folder", "base.pivot_pivoted.json")
        self.assertIsNone(pivot)
        self.assertEqual(source_code, "eng_Latn")
        self.assertEqual(input_file, expected_path)
        mock_file.assert_called_once_with(expected_path, "w", encoding="utf-8")
        mock_dump.assert_called_once_with(
            [{"text": "Hello", "start": 0.0, "end": 1.0}],
            mock_file(),
            ensure_ascii=False,
        )
        mock_log.assert_any_call("  [Translate] Reusing existing English pivot SRT.", "INFO")

    def test_build_manifest_jobs_skips_en_when_pivot_enabled(self):
        worker_context = {
            "missing_langs": ["en", "fr"],
            "folder": "folder",
            "base_name": "base",
        }
        pivot_config = {"output": "pivot.json"}
        temp_files = []

        with (
            patch("modules.configuration.config.nllb_to_iso", return_value="en"),
            patch("modules.configuration.config.TARGET_LANGUAGES", {"fr": {"code": "fra_Latn", "label": "French"}}),
            patch("os.path.exists", return_value=False),
        ):
            jobs = translation._build_manifest_jobs(worker_context, "eng_Latn", "pivot.json", pivot_config, temp_files)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["lang"], "fr")
        self.assertEqual(jobs[0]["input"], "pivot.json")

    def test_cleanup_worker_process_kills_on_wait_timeout(self):
        class FakeTimeoutExpired(Exception):
            pass

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = FakeTimeoutExpired()

        with (
            patch("modules.pipeline.translation.subprocess.TimeoutExpired", FakeTimeoutExpired),
            patch("modules.pipeline.translation.log"),
            patch("modules.utils.unregister_subprocess") as mock_unreg,
        ):
            translation._cleanup_worker_process(mock_proc)

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        mock_unreg.assert_called_once_with(mock_proc)

    def test_cleanup_worker_process_unregisters_when_already_exited(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0

        with patch("modules.utils.unregister_subprocess") as mock_unreg:
            translation._cleanup_worker_process(mock_proc)

        mock_unreg.assert_called_once_with(mock_proc)

    def test_cleanup_temp_files_ignores_remove_oserror(self):
        with patch("os.path.exists", return_value=True), patch("os.remove", side_effect=OSError("locked")):
            translation._cleanup_temp_files(["a.tmp", "b.tmp"])

    def test_safe_remove_ignores_oserror(self):
        with patch("os.path.exists", return_value=True), patch("os.remove", side_effect=OSError("locked")):
            translation._safe_remove("temp.json")

    @patch("modules.pipeline.translation.subprocess.Popen")
    @patch("modules.utils.register_subprocess")
    @patch("modules.pipeline.translation._poll_translation_results")
    @patch("modules.pipeline.translation._cleanup_worker_process")
    @patch("modules.pipeline.translation.log")
    def test_run_worker_process_logs_nonzero_and_raises_runtime_error(
        self,
        mock_log,
        mock_cleanup_worker,
        mock_poll,
        mock_register,
        mock_popen,
    ):
        mock_proc = MagicMock()
        mock_proc.returncode = 2
        mock_popen.return_value.__enter__.return_value = mock_proc

        worker_context = {
            "manifest_path": "manifest.json",
            "missing_langs": ["fr"],
            "folder": "folder",
            "base_name": "base",
            "segments": [],
            "temp_files": ["tmp1.json"],
        }

        with patch("modules.pipeline.translation._cleanup_temp_files") as mock_cleanup_temp:
            with self.assertRaises(RuntimeError):
                translation._run_worker_process(worker_context)

        mock_register.assert_called_once_with(mock_proc)
        mock_poll.assert_called_once()
        mock_log.assert_any_call("!!! Translation worker failed with code 2", "ERROR")
        mock_cleanup_worker.assert_called_once_with(mock_proc)
        mock_cleanup_temp.assert_called_once_with(worker_context["temp_files"])

    @patch("modules.pipeline.translation.subprocess.Popen")
    @patch("modules.utils.register_subprocess")
    @patch("modules.pipeline.translation._run_worker_and_collect_results")
    @patch("modules.pipeline.translation._cleanup_post_worker_memory")
    def test_run_worker_process_ignores_temp_cleanup_oserror(
        self,
        mock_cleanup_memory,
        mock_run_worker,
        mock_register,
        mock_popen,
    ):
        mock_proc = MagicMock()
        mock_popen.return_value.__enter__.return_value = mock_proc

        worker_context = {
            "manifest_path": "manifest.json",
            "missing_langs": ["fr"],
            "folder": "folder",
            "base_name": "base",
            "segments": [],
            "temp_files": ["tmp1.json"],
        }

        with patch("modules.pipeline.translation._cleanup_temp_files", side_effect=OSError("cleanup failed")):
            translation._run_worker_process(worker_context)

        mock_register.assert_called_once_with(mock_proc)
        mock_run_worker.assert_called_once_with(mock_proc, worker_context)
        mock_cleanup_memory.assert_called_once()

    def test_execute_translation_workers_with_positional_arguments(self):
        with (
            patch("modules.configuration.config.nllb_to_iso", return_value="en"),
            patch("modules.pipeline.translation._create_translation_manifest", return_value=("manifest.json", ["temp.json"])),
            patch("modules.pipeline.translation._run_worker_process") as mock_run,
            patch("modules.pipeline.translation.log"),
        ):
            translation._execute_translation_workers(["fr"], [{"text": "x"}], "eng_Latn", "folder", "base", [])

        mock_run.assert_called_once()

    def test_translate_segments_runs_with_no_model_manager(self):
        segment = MagicMock()
        segment.text = "Hello"
        segment.start = 0.0
        segment.end = 1.0

        with (
            patch("modules.pipeline.translation._identify_missing_targets", return_value=(["fr"], 0)),
            patch(
                "modules.pipeline.translation._prepare_source_data",
                return_value=([segment], [{"text": "Hello", "start": 0.0, "end": 1.0}]),
            ),
            patch("modules.configuration.config.get_nllb_code", return_value="eng_Latn"),
            patch("modules.pipeline.translation._execute_translation_workers") as mock_exec,
        ):
            result = translation.translate_segments([segment], "en", None, "folder", "base")

        self.assertEqual(result, {})
        mock_exec.assert_called_once()

    @patch("modules.pipeline.translation.open", new_callable=mock_open)
    @patch("subprocess.Popen")
    @patch("modules.utils.register_subprocess")
    @patch("modules.utils.unregister_subprocess")
    @patch("modules.pipeline.translation._poll_translation_results")
    @patch("modules.pipeline.translation.log")
    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    def test_execute_translation_workers_orphaned(
        self, mock_remove, mock_exists, mock_log, mock_poll, mock_unreg, mock_reg, mock_popen, mock_file
    ):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Still running
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc
        mock_popen.return_value.__enter__.return_value = mock_proc
        mock_poll.return_value = set()

        with patch("modules.configuration.config.TARGET_LANGUAGES", {"fr": {"code": "fra_Latn", "label": "French"}}):
            translation._execute_translation_workers(["fr"], [], "eng_Latn", "folder", "base", [])
        mock_log.assert_any_call("!   [Cleanup] Terminating orphaned translation worker...", "WARNING")
        mock_proc.terminate.assert_called_once()
        mock_unreg.assert_called_once_with(mock_proc)

    def test_translate_segments_no_missing(self):
        with patch("modules.pipeline.translation._identify_missing_targets", return_value=([], 0)):
            res = translation.translate_segments([], "en", MagicMock(), "folder", "base")
            self.assertEqual(res, {})

    def test_translate_segments_no_source_data(self):
        with (
            patch("modules.pipeline.translation._identify_missing_targets", return_value=(["fr"], 0)),
            patch("modules.pipeline.translation._prepare_source_data", return_value=([], [])),
        ):
            res = translation.translate_segments([], "en", MagicMock(), "folder", "base")
            self.assertEqual(res, {})


if __name__ == "__main__":
    unittest.main()
