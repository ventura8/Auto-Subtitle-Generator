import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import json
import sys

# Ensure modules can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Avoid global sys.modules hacks. conftest.py handles AI libs.
# Delayed import inside tests to ensure mocks are active


class TestIsolatedTranslator(unittest.TestCase):
    def test_translate_batch_chunk(self):
        from modules import isolated_translator
        # Mock translator
        mock_translator = MagicMock()
        mock_translator.translate.return_value = ["Hola", "Mundo"]

        batch = ["Hello", "World"]
        src_code = "eng_Latn"
        tgt_code = "spa_Latn"

        # CORRECT ORDER: translator, chunk, src, tgt
        res = isolated_translator._translate_batch_chunk(
            mock_translator, batch, src_code, tgt_code
        )

        self.assertEqual(res, ["Hola", "Mundo"])
        mock_translator.translate.assert_called()



    @patch("sys.exit")
    def test_main(self, mock_exit):
        from modules import isolated_translator

        # Simulate exit stopping execution
        mock_exit.side_effect = SystemExit

        # Use patch.object to ensure we catch the exact reference used by the module
        with patch.object(isolated_translator, "run_batch_translation_worker") as mock_run, \
                patch.object(isolated_translator.utils, "init_console") as mock_init, \
                patch.object(sys, "argv", ["isolated_translator.py", "--batch", "manifest.json"]):

            with self.assertRaises(SystemExit):
                isolated_translator.main()

            mock_run.assert_called_with("manifest.json")
            mock_exit.assert_called_with(0)

    @patch("modules.isolated_translator._process_single_job")
    @patch("modules.isolated_translator.ModelManager")
    @patch("modules.isolated_translator.config.load_config")
    @patch("modules.isolated_translator.OPTIMIZER")
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data='{"jobs": [{"lang": "es"}]}')
    def test_run_batch_translation_worker(self, mock_open, mock_opt, mock_load, mock_mm, mock_proc):
        from modules import isolated_translator
        isolated_translator.run_batch_translation_worker("manifest.json")
        mock_proc.assert_called()
        self.assertEqual(mock_proc.call_count, 1)

    @patch("modules.isolated_translator.utils.print_progress_bar")
    @patch("modules.isolated_translator.time.sleep")
    @patch("os.rename")
    @patch("os.remove")
    @patch("os.path.exists")
    def test_process_single_job(self, mock_exists, mock_remove, mock_rename, mock_sleep, mock_print):
        from modules import isolated_translator

        # Mock translator
        mock_translator = MagicMock()
        mock_translator.translate.return_value = ["Translated Text"]

        # Mock file I/O
        input_data = [{"text": "Source", "start": 0, "end": 1}]
        m_open = mock_open(read_data=json.dumps(input_data))

        # Files:
        # 1. output_file exists? -> False (for remove check)
        # 2. output_file exists? -> False (for sync wait loop exit)
        mock_exists.return_value = False

        job = {
            "lang": "es",
            "tgt_code": "spa",
            "input": "in.json",
            "output": "out.json",
            "src_code": "eng_Latn"
        }

        with patch("builtins.open", m_open):
            isolated_translator._process_single_job(job, 0, 1, mock_translator)

        mock_translator.translate.assert_called()
        mock_rename.assert_called()
        # Verify file write to satisfy unused variable lint
        m_open().write.assert_called()

    @patch("modules.isolated_translator.ModelManager")
    @patch("modules.isolated_translator.config.load_config")
    @patch("modules.isolated_translator.OPTIMIZER")
    @patch("modules.utils.print_progress_bar")
    @patch("modules.isolated_translator.log")
    def test_run_translation_worker_direct(self, mock_log, mock_print, mock_opt, mock_load, mock_mm):
        from modules import isolated_translator

        # Mocking data
        mock_translator = MagicMock()
        mock_translator.translate.return_value = ["Translated"]
        mock_mm.return_value.get_nllb.return_value = mock_translator

        input_data = [{"text": "Hello", "start": 0.0, "end": 1.0}]
        m_open = mock_open(read_data=json.dumps(input_data))

        with patch("builtins.open", m_open):
            isolated_translator.run_translation_worker(
                "in.json", "out.json", "eng_Latn", "spa_Latn", 1, "Spanish", " [Prefix]"
            )

        mock_translator.translate.assert_called()
        # Verify it validates the write
        m_open().write.assert_called()

    @patch("sys.exit")
    def test_main_legacy_mode(self, mock_exit):
        from modules import isolated_translator

        # Simulate legacy CLI args
        # input output src tgt batch label
        test_args = [
            "isolated_translator.py", "in.json", "out.json",
            "eng_Latn", "spa_Latn", "4", "Spanish"
        ]

        with patch.object(sys, "argv", test_args), \
                patch("modules.isolated_translator.run_translation_worker") as mock_worker, \
                patch("modules.isolated_translator.utils.init_console"):

            isolated_translator.main()

            mock_worker.assert_called_with(
                "in.json", "out.json", "eng_Latn", "spa_Latn", 4, "Spanish", '  [Translate] Spanish'
            )
            # Should NOT exit strict 0 in legacy mode (it falls through or implicit return)
            # But main() doesn't have sys.exit(0) at end of legacy block?
            # Let's check code. It just finishes.


if __name__ == "__main__":
    unittest.main()
