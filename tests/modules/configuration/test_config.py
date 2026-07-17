import copy
import importlib
import unittest
from unittest.mock import MagicMock, mock_open, patch


class TestConfig(unittest.TestCase):
    def setUp(self):
        global config
        from modules.configuration import config

        self.addCleanup(importlib.reload, config)

        original_target_languages = copy.deepcopy(config.TARGET_LANGUAGES)
        original_whisper_model_size = config.WHISPER_MODEL_SIZE
        original_forced_language = config.FORCED_LANGUAGE
        original_use_vocal_separation = config.USE_VOCAL_SEPARATION
        original_hallucination_silence_threshold = config.HALLUCINATION_SILENCE_THRESHOLD
        original_translator_engine = config.TRANSLATOR_ENGINE

        self.addCleanup(setattr, config, "TARGET_LANGUAGES", original_target_languages)
        self.addCleanup(setattr, config, "WHISPER_MODEL_SIZE", original_whisper_model_size)
        self.addCleanup(setattr, config, "FORCED_LANGUAGE", original_forced_language)
        self.addCleanup(setattr, config, "USE_VOCAL_SEPARATION", original_use_vocal_separation)
        self.addCleanup(setattr, config, "HALLUCINATION_SILENCE_THRESHOLD", original_hallucination_silence_threshold)
        self.addCleanup(setattr, config, "TRANSLATOR_ENGINE", original_translator_engine)

        # Reset globals
        config.TARGET_LANGUAGES = {}
        config.WHISPER_MODEL_SIZE = "small"
        config.FORCED_LANGUAGE = None
        config.USE_VOCAL_SEPARATION = True
        config.HALLUCINATION_SILENCE_THRESHOLD = 0.9
        config.TRANSLATOR_ENGINE = "nllb"

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists", return_value=True)
    def test_load_full_config(self, mock_exists, mock_file):
        tracked_globals = [
            "DEBUG_LOGGING",
            "TARGET_LANGUAGES",
            "WHISPER_MODEL_SIZE",
            "FORCED_LANGUAGE",
            "USE_VOCAL_SEPARATION",
            "INITIAL_PROMPT",
            "HALLUCINATION_SILENCE_THRESHOLD",
            "HALLUCINATION_REPETITION_THRESHOLD",
            "HALLUCINATION_PHRASES",
            "NLLB_MODEL_ID",
            "AUDIO_SEPARATOR_MODEL_ID",
            "NLLB_NUM_BEAMS",
            "NLLB_LENGTH_PENALTY",
            "VAD_MIN_SILENCE_MS",
        ]
        for global_name in tracked_globals:
            original_value = copy.deepcopy(getattr(config, global_name))
            self.addCleanup(setattr, config, global_name, original_value)

        fake_yaml = MagicMock()
        fake_yaml.YAMLError = ValueError

        # Return dict directly, bypassing parser
        fake_yaml.safe_load.return_value = {
            "debug_logging": True,
            "target_languages": {"de": {"code": "deu_Latn", "label": "German"}},
            "whisper": {
                "model_size": "medium",
                "language": "en",
                "use_vocal_separation": False,
                "use_prompt": True,
                "custom_prompt": "Smart prompt",
                "custom_prompt_priority": True,
            },
            "hallucinations": {"silence_threshold": 0.5, "repetition_threshold": 10, "known_phrases": ["bad phrase"]},
            "models": {"nllb": "facebook/nllb-distilled", "audio_separator": "custom.ckpt"},
            "nllb": {"num_beams": 3, "length_penalty": 0.8},
            "vad": {"min_silence_duration_ms": 300},
            "performance": {"whisper_beam": 2, "nllb_batch": 16},
        }

        optimizer = MagicMock()
        optimizer.config = {}
        log = MagicMock()

        with patch("modules.configuration.config._get_yaml_module", return_value=fake_yaml):
            res = config.load_config(optimizer, log)

        self.assertTrue(res)
        self.assertTrue(config.DEBUG_LOGGING)
        self.assertEqual(config.WHISPER_MODEL_SIZE, "medium")
        self.assertEqual(config.FORCED_LANGUAGE, "en")
        self.assertFalse(config.USE_VOCAL_SEPARATION)
        self.assertEqual(config.INITIAL_PROMPT, "Smart prompt")
        self.assertEqual(config.HALLUCINATION_SILENCE_THRESHOLD, 0.5)
        self.assertEqual(config.NLLB_MODEL_ID, "facebook/nllb-distilled")
        self.assertEqual(config.NLLB_NUM_BEAMS, 3)
        self.assertEqual(config.VAD_MIN_SILENCE_MS, 300)

        # Optimizer overrides
        self.assertEqual(optimizer.config["whisper_beam"], 2)
        self.assertEqual(optimizer.config["nllb_batch"], 16)

    @patch("os.path.exists", return_value=False)
    def test_load_config_missing(self, mock_exists):
        log = MagicMock()
        res = config.load_config(MagicMock(), log)
        self.assertTrue(res)  # Returns True but uses defaults
        # Should have populated defaults
        self.assertIn("es", config.TARGET_LANGUAGES)

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists", return_value=True)
    def test_load_config_invalid(self, mock_exists, mock_file):
        class FakeYAMLError(Exception):
            pass

        fake_yaml = MagicMock()
        fake_yaml.YAMLError = FakeYAMLError
        fake_yaml.safe_load.side_effect = FakeYAMLError("YAML Error")

        log = MagicMock()
        with patch("modules.configuration.config._get_yaml_module", return_value=fake_yaml):
            res = config.load_config(MagicMock(), log)
        self.assertFalse(res)
        log.assert_called_with(unittest.mock.ANY, "ERROR")

    def test_get_nllb_code(self):
        config.TARGET_LANGUAGES = {"xx": {"code": "xxx_Latn"}}
        self.assertEqual(config.get_nllb_code("xx"), "xxx_Latn")
        self.assertEqual(config.get_nllb_code("es"), "spa_Latn")  # fallback
        self.assertEqual(config.get_nllb_code("unknown"), "eng_Latn")  # default


if __name__ == "__main__":
    unittest.main()
