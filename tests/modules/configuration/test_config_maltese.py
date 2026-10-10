import unittest
from unittest.mock import patch

from modules.asr.languages import NVIDIA_EU25
from modules.configuration import config


class TestMalteseLanguageCodes(unittest.TestCase):
    def test_maltese_maps_to_nllb_and_back(self):
        self.assertEqual(config.get_nllb_code("mt"), "mlt_Latn")
        self.assertEqual(config.nllb_to_iso("mlt_Latn"), "mt")
        self.assertEqual(config.NLLB_PREFIX_TO_ISO["mlt"], "mt")

    def test_maltese_mux_code(self):
        self.assertEqual(config.to_mux_language_code("mt"), "mlt")

    def test_every_nvidia_language_has_nllb_and_container_codes(self):
        self.assertFalse(sorted(NVIDIA_EU25 - set(config.ISO_TO_NLLB)))
        self.assertFalse(sorted(NVIDIA_EU25 - set(config.ISO_639_1_TO_639_2)))


class TestMuxLanguageCodeFallbacks(unittest.TestCase):
    def test_static_nllb_prefix_is_used_without_a_639_2_entry(self):
        self.assertEqual(config.to_mux_language_code("sd"), "snd")
        self.assertEqual(config.to_mux_language_code(None), "und")

    def test_target_language_code_prefix_is_used(self):
        with patch.object(config, "TARGET_LANGUAGES", {"xx": {"code": "plt_Latn"}, "yy": {"code": ""}}):
            self.assertEqual(config.to_mux_language_code("xx"), "mlg")
            self.assertEqual(config.to_mux_language_code("yy"), "yy")


class TestNllbCodeFallbackWarning(unittest.TestCase):
    @patch("modules.runtime.logging_utils.log")
    def test_unknown_code_warns_and_falls_back_to_english(self, mock_log):
        self.assertEqual(config.get_nllb_code("haw"), "eng_Latn")
        mock_log.assert_called_once()
        message, level = mock_log.call_args.args
        self.assertEqual(level, "WARNING")
        self.assertIn("'haw'", message)

    @patch("modules.runtime.logging_utils.log")
    def test_known_code_does_not_warn(self, mock_log):
        self.assertEqual(config.get_nllb_code("ro"), "ron_Latn")
        mock_log.assert_not_called()

    @patch("modules.runtime.logging_utils.log")
    def test_target_language_override_does_not_warn(self, mock_log):
        with patch.object(config, "TARGET_LANGUAGES", {"haw": {"code": "haw_Latn"}}):
            self.assertEqual(config.get_nllb_code("haw"), "haw_Latn")
        mock_log.assert_not_called()


if __name__ == "__main__":
    unittest.main()
