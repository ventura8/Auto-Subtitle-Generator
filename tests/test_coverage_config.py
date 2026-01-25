from modules import config
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure modules can be imported
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


class TestCoverageConfig(unittest.TestCase):

    def test_load_whisper_config_extra(self):
        w_conf = {
            "model_size": "base",
            "language": False,
            "use_vocal_separation": False,
            "use_prompt": False
        }
        config._load_whisper_config(w_conf, MagicMock())
        self.assertEqual(config.WHISPER_MODEL_SIZE, "base")
        self.assertIsNone(config.FORCED_LANGUAGE)
        self.assertFalse(config.USE_VOCAL_SEPARATION)
        self.assertIsNone(config.INITIAL_PROMPT)

    def test_load_whisper_config_custom_prompt_empty(self):
        w_conf = {
            "use_prompt": True,
            "custom_prompt": ""
        }
        config._load_whisper_config(w_conf, MagicMock())
        # Should stay as default or None if not provided
        self.assertIsNotNone(config.INITIAL_PROMPT)

    def test_load_hallucination_config(self):
        h_conf = {
            "silence_threshold": 0.5,
            "repetition_threshold": 10,
            "known_phrases": ["test phrase"]
        }
        config._load_hallucination_config(h_conf, MagicMock())
        self.assertEqual(config.HALLUCINATION_SILENCE_THRESHOLD, 0.5)
        self.assertEqual(config.HALLUCINATION_REPETITION_THRESHOLD, 10)
        self.assertEqual(config.HALLUCINATION_PHRASES, ["test phrase"])

    def test_load_performance_overrides_empty(self):
        opt = MagicMock()
        config._load_performance_overrides({}, opt, MagicMock())
        # No change

    def test_load_performance_overrides_full(self):
        p_conf = {
            "whisper_beam": 1,
            "nllb_batch": 2,
            "whisper_workers": 3,
            "ffmpeg_threads": 4
        }
        opt = MagicMock()
        opt.config = {}
        config._load_performance_overrides(p_conf, opt, MagicMock())
        self.assertEqual(opt.config["whisper_beam"], 1)
        self.assertEqual(opt.config["nllb_batch"], 2)
        self.assertEqual(opt.config["whisper_workers"], 3)
        self.assertEqual(opt.config["ffmpeg_threads"], 4)

    def test_load_nllb_config(self):
        n_conf = {
            "num_beams": 1,
            "length_penalty": 1.2,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 2
        }
        config._load_nllb_config(n_conf, MagicMock())
        self.assertEqual(config.NLLB_NUM_BEAMS, 1)
        self.assertEqual(config.NLLB_LENGTH_PENALTY, 1.2)
        self.assertEqual(config.NLLB_REPETITION_PENALTY, 1.3)
        self.assertEqual(config.NLLB_NO_REPEAT_NGRAM_SIZE, 2)

    def test_load_type_and_model_config(self):
        conf = {
            "file_types": {"extensions": [".mp4"]},
            "models": {
                "nllb": "nllb-model",
                "audio_separator": "sep-model"
            }
        }
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
        with patch("os.path.exists", return_value=True), \
                patch("builtins.open", side_effect=Exception("Error")):
            res = config.load_config(MagicMock(), MagicMock())
            self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
