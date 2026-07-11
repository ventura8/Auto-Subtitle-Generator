import unittest
from unittest.mock import MagicMock, mock_open, patch
import os
import sys
import importlib

# Ensure modules can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

config = importlib.import_module("modules.config")


class TestCoverageConfig(unittest.TestCase):
    def setUp(self):
        config.TARGET_LANGUAGES = {}
        config.TRANSLATOR_ENGINE = "nllb"
        config.INITIAL_PROMPT = "Transcribe the following audio file."

    def test_load_whisper_config_extra(self):
        w_conf = {"model_size": "base", "language": False, "use_vocal_separation": False, "use_prompt": False}
        config._load_whisper_config(w_conf, MagicMock())
        self.assertEqual(config.WHISPER_MODEL_SIZE, "base")
        self.assertIsNone(config.FORCED_LANGUAGE)
        self.assertFalse(config.USE_VOCAL_SEPARATION)
        self.assertIsNone(config.INITIAL_PROMPT)

    def test_load_whisper_config_custom_prompt_empty(self):
        w_conf = {"use_prompt": True, "custom_prompt": ""}
        config._load_whisper_config(w_conf, MagicMock())
        # Should stay as default or None if not provided
        self.assertIsNotNone(config.INITIAL_PROMPT)

    def test_load_hallucination_config(self):
        h_conf = {"silence_threshold": 0.5, "repetition_threshold": 10, "known_phrases": ["test phrase"]}
        config._load_hallucination_config(h_conf, MagicMock())
        self.assertEqual(config.HALLUCINATION_SILENCE_THRESHOLD, 0.5)
        self.assertEqual(config.HALLUCINATION_REPETITION_THRESHOLD, 10)
        self.assertEqual(config.HALLUCINATION_PHRASES, ["test phrase"])

    def test_load_performance_overrides_empty(self):
        opt = MagicMock()
        config._load_performance_overrides({}, opt, MagicMock())
        # No change

    def test_load_performance_overrides_full(self):
        p_conf = {"whisper_beam": 1, "nllb_batch": 2, "whisper_workers": 3, "ffmpeg_threads": 4}
        opt = MagicMock()
        opt.config = {}
        config._load_performance_overrides(p_conf, opt, MagicMock())
        self.assertEqual(opt.config["whisper_beam"], 1)
        self.assertEqual(opt.config["nllb_batch"], 2)
        self.assertEqual(opt.config["whisper_workers"], 3)
        self.assertEqual(opt.config["ffmpeg_threads"], 4)

    def test_load_nllb_config(self):
        n_conf = {"num_beams": 1, "length_penalty": 1.2, "repetition_penalty": 1.3, "no_repeat_ngram_size": 2}
        config._load_nllb_config(n_conf, MagicMock())
        self.assertEqual(config.NLLB_NUM_BEAMS, 1)
        self.assertEqual(config.NLLB_LENGTH_PENALTY, 1.2)
        self.assertEqual(config.NLLB_REPETITION_PENALTY, 1.3)
        self.assertEqual(config.NLLB_NO_REPEAT_NGRAM_SIZE, 2)

    def test_load_type_and_model_config(self):
        conf = {"file_types": {"extensions": [".mp4"]}, "models": {"nllb": "nllb-model", "audio_separator": "sep-model"}}
        config._load_type_and_model_config(conf, MagicMock())
        self.assertIn(".mp4", config.VIDEO_EXTENSIONS)
        self.assertEqual(config.NLLB_MODEL_ID, "nllb-model")
        self.assertEqual(config.AUDIO_SEPARATOR_MODEL_ID, "sep-model")

    def test_load_config_not_found(self):
        with patch("os.path.exists", return_value=False):
            res = config.load_config(MagicMock(), MagicMock())
            self.assertTrue(res)
            self.assertIn("en", config.TARGET_LANGUAGES)

    def test_load_config_exception(self):
        with patch("os.path.exists", return_value=True), patch("builtins.open", side_effect=OSError("Error")):
            res = config.load_config(MagicMock(), MagicMock())
            self.assertFalse(res)

    def test_nllb_to_iso(self):
        config.TARGET_LANGUAGES = {"custom": {"code": "custom_code"}}
        self.assertEqual(config.nllb_to_iso("custom_code"), "custom")
        self.assertEqual(config.nllb_to_iso("ron_Latn"), "ro")
        self.assertEqual(config.nllb_to_iso("eng_Latn"), "en")
        self.assertEqual(config.nllb_to_iso("spa_Latn"), "es")
        self.assertEqual(config.nllb_to_iso("lvs_Latn"), "lv")
        self.assertEqual(config.nllb_to_iso("nonexistent_Latn"), "no")
        self.assertEqual(config.nllb_to_iso("en"), "en")
        self.assertEqual(config.nllb_to_iso(None), "en")

    def test_load_base_config_translation_engine(self):
        conf = {"translation": {"engine": "Nllb"}}
        config._load_base_config_snippet(conf, MagicMock())
        self.assertEqual(config.TRANSLATOR_ENGINE, "nllb")

    def test_normalize_target_languages_boolean_key(self):
        logger = MagicMock()
        raw = {False: {"code": "nob_Latn", "label": "Norwegian"}}

        normalized = config._normalize_target_languages(raw, logger)

        self.assertIn("no", normalized)
        self.assertEqual(normalized["no"]["code"], "nob_Latn")

    def test_load_base_config_normalizes_boolean_target_key(self):
        logger = MagicMock()
        conf = {"target_languages": {False: {"code": "nob_Latn", "label": "Norwegian"}}}

        config._load_base_config_snippet(conf, logger)

        self.assertIn("no", config.TARGET_LANGUAGES)

    def test_save_token_to_config_replaces_existing_value(self):
        file_data = 'translation:\n  engine: "nllb"\n# keep this comment\nhf_token: "old"\nmodels:\n  nllb: "facebook/nllb-200-3.3B"\n'
        mocked_open = mock_open(read_data=file_data)
        fake_yaml = MagicMock()
        fake_yaml.YAMLError = ValueError

        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mocked_open),
            patch("modules.config._get_yaml_module", return_value=fake_yaml),
            patch("builtins.print"),
        ):
            config._save_token_to_config("new-token")

        written_data = "".join(call.args[0] for call in mocked_open().write.call_args_list)
        self.assertIn('hf_token: "new-token"', written_data)
        self.assertIn("# keep this comment", written_data)
        self.assertIn('translation:\n  engine: "nllb"', written_data)
        self.assertIn('models:\n  nllb: "facebook/nllb-200-3.3B"', written_data)

    def test_save_token_to_config_appends_when_missing(self):
        mocked_open = mock_open(read_data='translation:\n  engine: "nllb"\n')
        fake_yaml = MagicMock()
        fake_yaml.YAMLError = ValueError

        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mocked_open),
            patch("modules.config._get_yaml_module", return_value=fake_yaml),
            patch("builtins.print"),
        ):
            config._save_token_to_config("new-token")

        written_data = "".join(call.args[0] for call in mocked_open().write.call_args_list)
        self.assertIn('translation:\n  engine: "nllb"', written_data)
        self.assertIn('hf_token: "new-token"', written_data)

    def test_handle_hf_token_prompt_full_flow(self):
        logger = MagicMock()
        conf = {
            "translation": {"engine": "translategemma"},
            "models": {"translategemma": "google/translategemma-12b-it"},
        }
        mocked_open = mock_open(read_data='translation:\n  engine: "translategemma"\n')
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mocked_open),
            patch("sys.stdin.isatty", return_value=True),
            patch.dict("os.environ", {}, clear=True),
            patch("builtins.input", return_value="hf_test_token"),
            patch("webbrowser.open") as mock_browser,
            patch("builtins.print"),
            patch(
                "modules.config._get_yaml_module",
                return_value=MagicMock(safe_load=MagicMock(return_value=conf), YAMLError=ValueError),
            ),
        ):
            self.assertTrue(config.load_config(MagicMock(), logger))
            self.assertEqual(os.environ.get("HF_TOKEN"), "hf_test_token")
            self.assertEqual(mock_browser.call_count, 2)

    def test_load_config_nllb_vad_and_performance(self):
        logger = MagicMock()
        optimizer = MagicMock()
        optimizer.config = {}
        conf = {
            "debug_logging": True,
            "translation": {"engine": "nllb"},
            "target_languages": {"it": {"code": "ita_Latn", "label": "Italian"}},
            "whisper": {
                "model_size": "small",
                "language": "it",
                "use_vocal_separation": True,
                "use_prompt": True,
                "custom_prompt": "custom",
                "custom_prompt_priority": True,
            },
            "hallucinations": {
                "silence_threshold": 0.3,
                "repetition_threshold": 9,
                "known_phrases": ["x"],
            },
            "models": {
                "nllb": "nllb-model",
                "audio_separator": "sep-model",
                "translategemma": "gemma-model",
            },
            "nllb": {
                "num_beams": 6,
                "length_penalty": 1.5,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
            },
            "vad": {"min_silence_duration_ms": 750},
            "performance": {
                "whisper_beam": 2,
                "nllb_batch": 4,
                "whisper_workers": 5,
                "ffmpeg_threads": 6,
            },
        }
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="irrelevant")),
            patch("warnings.filterwarnings"),
            patch(
                "modules.config._get_yaml_module",
                return_value=MagicMock(safe_load=MagicMock(return_value=conf), YAMLError=ValueError),
            ),
        ):
            self.assertTrue(config.load_config(optimizer, logger))

        self.assertTrue(config.DEBUG_LOGGING)
        self.assertEqual(config.TRANSLATOR_ENGINE, "nllb")
        self.assertEqual(config.WHISPER_MODEL_SIZE, "small")
        self.assertEqual(config.FORCED_LANGUAGE, "it")
        self.assertEqual(config.INITIAL_PROMPT, "custom")
        self.assertEqual(config.NLLB_MODEL_ID, "nllb-model")
        self.assertEqual(config.AUDIO_SEPARATOR_MODEL_ID, "sep-model")
        self.assertEqual(config.TRANSLATEGEMMA_MODEL_ID, "gemma-model")
        self.assertEqual(config.NLLB_NUM_BEAMS, 6)
        self.assertEqual(config.NLLB_NO_REPEAT_NGRAM_SIZE, 3)
        self.assertEqual(config.VAD_MIN_SILENCE_MS, 750)
        self.assertEqual(optimizer.config["ffmpeg_threads"], 6)


if __name__ == "__main__":
    unittest.main()
