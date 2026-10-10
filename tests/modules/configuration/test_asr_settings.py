import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from modules.configuration import asr_settings, config

REPO_ROOT = Path(__file__).resolve().parents[3]
CANARY_SHA = "d455706339a6b32e1aa40f82c713a482a0c938e2"
PARAKEET_SHA = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"


def _warnings(log):
    """Return the messages logged at WARNING level."""
    return [call.args[0] for call in log.call_args_list if call.args[1:] == ("WARNING",)]


class AsrSettingsTestCase(unittest.TestCase):
    def setUp(self):
        self.addCleanup(asr_settings.set_cli_override, None)
        self.addCleanup(asr_settings.reset)
        asr_settings.set_cli_override(None)
        asr_settings.reset()
        self.log = MagicMock()

    def load(self, data):
        asr_settings.load(data, self.log)
        return _warnings(self.log)


class TestAsrDefaults(AsrSettingsTestCase):
    def test_defaults_after_reset(self):
        self.assertEqual(asr_settings.active_engine(), "whisper")
        self.assertEqual(asr_settings.routes(), {"ro": "canary"})
        self.assertEqual(asr_settings.max_segment_seconds(), 15.0)
        self.assertFalse(asr_settings.force_detected_language())

    def test_model_ids_and_pinned_revisions(self):
        self.assertEqual(asr_settings.model_id("canary"), "nvidia/canary-1b-v2")
        self.assertEqual(asr_settings.model_id("parakeet"), "nvidia/parakeet-tdt-0.6b-v3")
        self.assertEqual(asr_settings.model_revision("canary"), CANARY_SHA)
        self.assertEqual(asr_settings.model_revision("parakeet"), PARAKEET_SHA)

    def test_routes_returns_a_copy(self):
        asr_settings.routes()["de"] = "parakeet"
        self.assertEqual(asr_settings.routes(), {"ro": "canary"})

    def test_reset_does_not_share_default_routes(self):
        self.load({"asr": {"routes": {"de": "parakeet"}}})
        asr_settings.reset()
        self.assertEqual(asr_settings.DEFAULTS["routes"], {"ro": "canary"})
        self.assertEqual(asr_settings.routes(), {"ro": "canary"})

    def test_missing_section_keeps_defaults_and_logs_summary(self):
        self.assertEqual(self.load({}), [])
        self.assertEqual(asr_settings.active_engine(), "whisper")
        self.log.assert_called_once()
        self.assertIn("ASR Engine: whisper", self.log.call_args.args[0])

    def test_null_section_keeps_defaults(self):
        self.assertEqual(self.load({"asr": None, "whisper": None}), [])
        self.assertEqual(asr_settings.routes(), {"ro": "canary"})


class TestAsrEngineKey(AsrSettingsTestCase):
    def test_engine_is_case_insensitive(self):
        self.assertEqual(self.load({"asr": {"engine": " Canary "}}), [])
        self.assertEqual(asr_settings.active_engine(), "canary")

    def test_auto_engine_is_accepted(self):
        self.load({"asr": {"engine": "auto"}})
        self.assertEqual(asr_settings.active_engine(), "auto")

    def test_invalid_engine_warns_and_keeps_default(self):
        warnings = self.load({"asr": {"engine": "nemo"}})
        self.assertEqual(len(warnings), 1)
        self.assertIn("asr.engine", warnings[0])
        self.assertEqual(asr_settings.active_engine(), "whisper")

    def test_non_string_engine_warns(self):
        self.assertEqual(len(self.load({"asr": {"engine": 3}})), 1)
        self.assertEqual(asr_settings.active_engine(), "whisper")

    def test_non_mapping_section_warns_and_keeps_defaults(self):
        warnings = self.load({"asr": ["canary"]})
        self.assertEqual(len(warnings), 1)
        self.assertIn("asr section must be a mapping", warnings[0])
        self.assertEqual(asr_settings.active_engine(), "whisper")


class TestAsrCliOverride(AsrSettingsTestCase):
    def test_override_wins_over_config_engine(self):
        asr_settings.set_cli_override("parakeet")
        self.load({"asr": {"engine": "canary"}})
        self.assertEqual(asr_settings.active_engine(), "parakeet")
        self.assertIn("(--asr)", self.log.call_args.args[0])

    def test_override_survives_reset(self):
        asr_settings.set_cli_override("canary")
        asr_settings.reset()
        self.assertEqual(asr_settings.active_engine(), "canary")

    def test_clearing_override_restores_config_engine(self):
        self.load({"asr": {"engine": "auto"}})
        asr_settings.set_cli_override("canary")
        asr_settings.set_cli_override(None)
        self.assertEqual(asr_settings.active_engine(), "auto")

    def test_invalid_override_raises(self):
        with self.assertRaisesRegex(ValueError, "whisper, canary, parakeet, auto"):
            asr_settings.set_cli_override("bogus")


class TestAsrRoutes(AsrSettingsTestCase):
    def test_routes_replace_defaults_and_normalise_keys(self):
        self.assertEqual(self.load({"asr": {"routes": {"DE": "Parakeet", "ro_RO": "canary", "ja": "whisper"}}}), [])
        self.assertEqual(asr_settings.routes(), {"de": "parakeet", "ro": "canary", "ja": "whisper"})

    def test_yaml_false_key_maps_to_norwegian(self):
        self.load({"asr": {"routes": {False: "whisper"}}})
        self.assertEqual(asr_settings.routes(), {"no": "whisper"})

    def test_null_routes_disable_routing(self):
        self.assertEqual(self.load({"asr": {"routes": None}}), [])
        self.assertEqual(asr_settings.routes(), {})
        self.assertIn("Routes: none", self.log.call_args.args[0])

    def test_non_mapping_routes_warn_and_keep_default(self):
        warnings = self.load({"asr": {"routes": ["ro"]}})
        self.assertEqual(len(warnings), 1)
        self.assertEqual(asr_settings.routes(), {"ro": "canary"})

    def test_unsupported_language_for_nvidia_engine_is_skipped(self):
        warnings = self.load({"asr": {"routes": {"ja": "canary", "de": "canary"}}})
        self.assertEqual(len(warnings), 1)
        self.assertIn("canary does not support 'ja'", warnings[0])
        self.assertEqual(asr_settings.routes(), {"de": "canary"})

    def test_route_to_auto_or_unknown_engine_is_skipped(self):
        warnings = self.load({"asr": {"routes": {"ro": "auto", "de": 5}}})
        self.assertEqual(len(warnings), 2)
        self.assertIn("engine must be one of", warnings[0])
        self.assertEqual(asr_settings.routes(), {})

    def test_invalid_language_keys_are_skipped(self):
        warnings = self.load({"asr": {"routes": {"romanian": "canary", True: "whisper", 7: "whisper"}}})
        self.assertEqual(len(warnings), 3)
        self.assertIn("ISO 639-1", warnings[0])
        self.assertEqual(asr_settings.routes(), {})


class TestAsrSegmentCap(AsrSettingsTestCase):
    def test_numeric_string_is_accepted(self):
        self.assertEqual(self.load({"asr": {"max_segment_seconds": "25"}}), [])
        self.assertEqual(asr_settings.max_segment_seconds(), 25.0)

    def test_value_below_range_is_clamped_with_warning(self):
        warnings = self.load({"asr": {"max_segment_seconds": 0}})
        self.assertEqual(asr_settings.max_segment_seconds(), 5.0)
        self.assertIn("outside [5, 30]", warnings[0])

    def test_value_above_range_is_clamped_with_warning(self):
        self.assertEqual(len(self.load({"asr": {"max_segment_seconds": 31}})), 1)
        self.assertEqual(asr_settings.max_segment_seconds(), 30.0)

    def test_unusable_values_warn_and_keep_default(self):
        for value in ("abc", None, True, "nan", [15]):
            with self.subTest(value=value):
                self.assertEqual(len(self.load({"asr": {"max_segment_seconds": value}})), 1)
                self.assertEqual(asr_settings.max_segment_seconds(), 15.0)
                self.log.reset_mock()


class TestForceDetectedLanguage(AsrSettingsTestCase):
    def test_true_is_applied(self):
        self.assertEqual(self.load({"whisper": {"force_detected_language": True}}), [])
        self.assertTrue(asr_settings.force_detected_language())

    def test_non_boolean_warns_and_keeps_default(self):
        warnings = self.load({"whisper": {"force_detected_language": "yes"}})
        self.assertIn("force_detected_language", warnings[0])
        self.assertFalse(asr_settings.force_detected_language())

    def test_non_mapping_whisper_section_warns(self):
        self.assertEqual(len(self.load({"whisper": "large"})), 1)
        self.assertFalse(asr_settings.force_detected_language())


class TestAsrModelIds(AsrSettingsTestCase):
    def test_custom_id_drops_the_pinned_revision(self):
        asr_settings.load_model_ids({"canary": " org/canary-ft ", "parakeet": "nvidia/parakeet-tdt-0.6b-v3"})
        self.assertEqual(asr_settings.model_id("canary"), "org/canary-ft")
        self.assertIsNone(asr_settings.model_revision("canary"))
        self.assertEqual(asr_settings.model_revision("parakeet"), PARAKEET_SHA)

    def test_blank_or_non_string_ids_keep_defaults(self):
        asr_settings.load_model_ids({"canary": "  ", "parakeet": 3})
        self.assertEqual(asr_settings.model_id("canary"), "nvidia/canary-1b-v2")
        self.assertEqual(asr_settings.model_id("parakeet"), "nvidia/parakeet-tdt-0.6b-v3")

    def test_reset_restores_model_ids(self):
        asr_settings.load_model_ids({"parakeet": "org/parakeet-ft"})
        asr_settings.reset()
        self.assertEqual(asr_settings.model_revision("parakeet"), PARAKEET_SHA)


class TestAsrThroughLoadConfig(AsrSettingsTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(config._reset_config_defaults)
        self.optimizer = MagicMock()
        self.optimizer.config = {}

    def load_config(self, data):
        fake_yaml = MagicMock()
        fake_yaml.YAMLError = ValueError
        fake_yaml.safe_load.return_value = data
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open()),
            patch("modules.configuration.config._get_yaml_module", return_value=fake_yaml),
        ):
            return config.load_config(self.optimizer, self.log)

    def test_section_and_model_ids_are_applied(self):
        data = {"asr": {"engine": "auto", "routes": {"de": "parakeet"}}, "models": {"canary": "org/c"}, "performance": {"asr_batch": 6}}
        self.assertTrue(self.load_config(data))
        self.assertEqual(asr_settings.active_engine(), "auto")
        self.assertEqual(asr_settings.routes(), {"de": "parakeet"})
        self.assertEqual(asr_settings.model_id("canary"), "org/c")
        self.assertEqual(self.optimizer.config["asr_batch"], 6)

    def test_invalid_asr_batch_fails_like_other_performance_pins(self):
        self.assertFalse(self.load_config({"performance": {"asr_batch": 0}}))
        self.log.assert_called_with(unittest.mock.ANY, "ERROR")
        self.assertNotIn("asr_batch", self.optimizer.config)

    def test_bad_section_warns_but_still_loads_other_sections(self):
        data = {"asr": {"engine": "nemo", "max_segment_seconds": "x"}, "vad": {"min_silence_duration_ms": 321}}
        self.assertTrue(self.load_config(data))
        self.assertEqual(len(_warnings(self.log)), 2)
        self.assertEqual(asr_settings.active_engine(), "whisper")
        self.assertEqual(config.VAD_MIN_SILENCE_MS, 321)

    def test_each_load_resets_previous_asr_values(self):
        self.load_config({"asr": {"engine": "canary", "max_segment_seconds": 8}, "models": {"parakeet": "org/p"}})
        self.load_config({})
        self.assertEqual(asr_settings.active_engine(), "whisper")
        self.assertEqual(asr_settings.max_segment_seconds(), 15.0)
        self.assertEqual(asr_settings.model_id("parakeet"), "nvidia/parakeet-tdt-0.6b-v3")

    def test_cli_override_survives_two_loads(self):
        asr_settings.set_cli_override("parakeet")
        self.load_config({"asr": {"engine": "canary"}})
        self.load_config({"asr": {"engine": "whisper"}})
        self.assertEqual(asr_settings.active_engine(), "parakeet")

    def test_shipped_config_yaml_loads(self):
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(REPO_ROOT)
        self.assertTrue(config.load_config(self.optimizer, self.log))
        self.assertFalse([message for message in _warnings(self.log) if "asr" in message.lower()])
        self.assertIn(asr_settings.active_engine(), asr_settings.ENGINES)


if __name__ == "__main__":
    unittest.main()
