import unittest

from modules.asr.routing import Route, resolve

ROUTES = {"ro": "canary", "de": "parakeet", "ja": "parakeet"}


class TestExplicitEngines(unittest.TestCase):
    def test_whisper_is_never_rerouted(self):
        self.assertEqual(resolve("whisper", "ro", ROUTES), Route("whisper", None))

    def test_canary_with_supported_language(self):
        self.assertEqual(resolve("canary", "ro", {}), Route("canary", None))

    def test_parakeet_with_supported_language(self):
        self.assertEqual(resolve("parakeet", "mt", {}), Route("parakeet", None))

    def test_unsupported_language_falls_back_with_reason(self):
        self.assertEqual(resolve("canary", "ja", ROUTES), Route("whisper", "canary does not support 'ja'"))

    def test_missing_language_falls_back_with_reason(self):
        self.assertEqual(resolve("parakeet", None, ROUTES), Route("whisper", "no language detected"))

    def test_unknown_engine_falls_back_with_reason(self):
        self.assertEqual(resolve("nemo", "ro", ROUTES), Route("whisper", "unknown ASR engine 'nemo'"))


class TestAutoRouting(unittest.TestCase):
    def test_routed_language_uses_its_engine(self):
        self.assertEqual(resolve("auto", "ro", ROUTES), Route("canary", None))
        self.assertEqual(resolve("auto", "de", ROUTES), Route("parakeet", None))

    def test_unrouted_language_uses_whisper_without_reason(self):
        self.assertEqual(resolve("auto", "fr", ROUTES), Route("whisper", None))

    def test_no_language_uses_whisper_without_reason(self):
        self.assertEqual(resolve("auto", None, ROUTES), Route("whisper", None))

    def test_route_to_unsupported_language_falls_back_with_reason(self):
        self.assertEqual(resolve("auto", "ja", ROUTES), Route("whisper", "parakeet does not support 'ja'"))


if __name__ == "__main__":
    unittest.main()
