import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from modules.held_dir import HeldDirectory

# Avoid global sys.modules hacks. conftest.py handles AI libs.
# Delayed import inside tests to ensure mocks are active


class TestIsolatedTranslator(unittest.TestCase):
    def _held_work_dir(self):
        """Create a work directory and hold it the way a validated batch manifest does."""
        from modules.pipeline import isolated_translator

        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        work_dir = os.path.join(os.path.realpath(root), "movie.asg-temp")
        os.mkdir(work_dir)
        isolated_translator._held_work_dir[0] = HeldDirectory(work_dir)
        self.addCleanup(self._release_held_work_dir)
        return work_dir

    @staticmethod
    def _release_held_work_dir():
        from modules.pipeline import isolated_translator

        held = isolated_translator._held_work_dir[0]
        if held is not None:
            held.close()
        isolated_translator._held_work_dir[0] = None

    def test_translate_batch_chunk(self):
        from modules.pipeline import isolated_translator

        # Mock translator
        mock_translator = MagicMock()
        mock_translator.translate.return_value = ["Hola", "Mundo"]

        batch = ["Hello", "World"]
        src_code = "eng_Latn"
        tgt_code = "spa_Latn"

        # CORRECT ORDER: translator, chunk, src, tgt
        res = isolated_translator._translate_batch_chunk(mock_translator, batch, src_code, tgt_code)

        self.assertEqual(res, ["Hola", "Mundo"])
        mock_translator.translate.assert_called()

    @patch("sys.exit")
    def test_main(self, mock_exit):
        from modules.pipeline import isolated_translator

        # Simulate exit stopping execution
        mock_exit.side_effect = SystemExit

        # Use patch.object to ensure we catch the exact reference used by the module
        with (
            patch.object(isolated_translator, "run_batch_translation_worker") as mock_run,
            patch.object(isolated_translator.utils, "init_console"),
            patch.object(isolated_translator, "_load_contained_manifest", return_value={"jobs": []}),
            patch.object(sys, "argv", ["isolated_translator.py", "--batch-stdin", "/work/movie.asg-temp"]),
        ):
            with self.assertRaises(SystemExit):
                isolated_translator.main()

            mock_run.assert_called_with({"jobs": []})
            mock_exit.assert_called_with(0)

    @patch("modules.pipeline.isolated_translator._process_single_job")
    @patch("modules.pipeline.isolated_translator.ModelManager")
    @patch("modules.pipeline.isolated_translator.config.load_config")
    @patch("modules.pipeline.isolated_translator.OPTIMIZER")
    def test_run_batch_translation_worker(self, mock_opt, mock_load, mock_mm, mock_proc):
        from modules.pipeline import isolated_translator

        isolated_translator.run_batch_translation_worker({"jobs": [{"lang": "es"}]})
        mock_proc.assert_called()
        self.assertEqual(mock_proc.call_count, 1)

    @patch("modules.pipeline.isolated_translator.utils.print_progress_bar")
    def test_process_single_job(self, mock_print):
        from modules.pipeline import isolated_translator

        mock_translator = MagicMock()
        mock_translator.translate.return_value = ["Translated Text"]
        work_dir = self._held_work_dir()
        input_file = os.path.join(work_dir, "in.json")
        output_file = os.path.join(work_dir, "out.json")
        with open(input_file, "w", encoding="utf-8") as handle:
            json.dump([{"text": "Source", "start": 0, "end": 1}], handle)
        written = []

        def consume_output(_seconds):
            with open(output_file, "r", encoding="utf-8") as handle:
                written.append(json.load(handle))
            os.remove(output_file)

        job = {"lang": "es", "tgt_code": "spa", "input": input_file, "output": output_file, "src_code": "eng_Latn"}
        with patch("modules.pipeline.isolated_translator.time.sleep", side_effect=consume_output):
            isolated_translator._process_single_job(job, 0, 1, mock_translator)

        mock_translator.translate.assert_called()
        self.assertEqual(written, [["Translated Text"]])
        self.assertEqual(os.listdir(work_dir), ["in.json"])

    @patch("modules.pipeline.isolated_translator.ModelManager")
    @patch("modules.pipeline.isolated_translator.config.load_config")
    @patch("modules.pipeline.isolated_translator.OPTIMIZER")
    @patch("modules.utils.print_progress_bar")
    @patch("modules.pipeline.isolated_translator.log")
    def test_run_translation_worker_direct(self, mock_log, mock_print, mock_opt, mock_load, mock_mm):
        from modules.pipeline import isolated_translator

        mock_translator = MagicMock()
        mock_translator.translate.return_value = ["Translated"]
        mock_mm.return_value.get_nllb.return_value = mock_translator
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        input_file = os.path.join(folder, "in.json")
        output_file = os.path.join(folder, "out.json")
        with open(input_file, "w", encoding="utf-8") as handle:
            json.dump([{"text": "Hello", "start": 0.0, "end": 1.0}], handle)

        isolated_translator.run_translation_worker(input_file, output_file, "eng_Latn", "spa_Latn", 1, "Spanish", " [Prefix]")

        mock_translator.translate.assert_called()
        with open(output_file, "r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), ["Translated"])

    @patch("sys.exit")
    def test_main_legacy_mode(self, mock_exit):
        from modules.pipeline import isolated_translator

        # Simulate legacy CLI args
        # input output src tgt batch label
        test_args = ["isolated_translator.py", "in.json", "out.json", "eng_Latn", "spa_Latn", "4", "Spanish"]

        with (
            patch.object(sys, "argv", test_args),
            patch("modules.pipeline.isolated_translator.run_translation_worker") as mock_worker,
            patch("modules.pipeline.isolated_translator.utils.init_console"),
        ):
            isolated_translator.main()

            mock_worker.assert_called_with("in.json", "out.json", "eng_Latn", "spa_Latn", 4, "Spanish", "  [Translate] Spanish")
            mock_exit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
