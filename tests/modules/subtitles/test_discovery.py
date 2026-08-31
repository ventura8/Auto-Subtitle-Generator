"""Tests for discovery helpers in modules.subtitles.discovery."""

import os
import tempfile
import unittest

from modules.subtitles import discovery


class TestDiscovery(unittest.TestCase):
    def test_extract_srt_language(self):
        self.assertEqual(discovery.extract_srt_language("movie.en.srt", "movie."), "en")
        self.assertEqual(discovery.extract_srt_language("movie.ro.srt", "movie."), "ro")
        self.assertIsNone(discovery.extract_srt_language("movie.en.srt.tmp", "movie."))
        self.assertIsNone(discovery.extract_srt_language("other.en.srt", "movie."))
        self.assertIsNone(discovery.extract_srt_language("movie.srt", "movie."))

    def test_find_existing_srt_languages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for fname in ["video.en.srt", "video.es.srt", "video.fr.tmp", "unrelated.srt"]:
                with open(os.path.join(temp_dir, fname), "w", encoding="utf-8") as f:
                    f.write("1\n00:00:01,000 --> 00:00:02,000\nHello\n")

            discovered = discovery.find_existing_srt_languages(temp_dir, "video")
            self.assertEqual(discovered, ["en", "es"])

    def test_find_existing_srt_languages_oserror(self):
        discovered = discovery.find_existing_srt_languages("/non/existent/path/for/tests", "video")
        self.assertEqual(discovered, [])

    def test_is_usable_language(self):
        self.assertTrue(discovery.is_usable_language("en"))
        self.assertTrue(discovery.is_usable_language("  RO  "))
        self.assertFalse(discovery.is_usable_language(""))
        self.assertFalse(discovery.is_usable_language("   "))
        self.assertFalse(discovery.is_usable_language(None))
        self.assertFalse(discovery.is_usable_language("und"))
        self.assertFalse(discovery.is_usable_language("undetermined"))
        self.assertFalse(discovery.is_usable_language("unknown"))

    def test_get_usable_languages(self):
        langs = ["en", "und", "es", "unknown", "fr", "undetermined", ""]
        self.assertEqual(discovery.get_usable_languages(langs), ["en", "es", "fr"])

    def test_prioritize_recorded_language(self):
        self.assertEqual(discovery.prioritize_recorded_language("ro", ["en", "es"]), ["ro", "en", "es"])
        self.assertEqual(discovery.prioritize_recorded_language("en", ["en", "es"]), ["en", "es"])
        self.assertEqual(discovery.prioritize_recorded_language("und", ["en", "es"]), ["en", "es"])
        self.assertEqual(discovery.prioritize_recorded_language("unknown", ["en", "es"]), ["en", "es"])
        self.assertEqual(discovery.prioritize_recorded_language(None, ["en", "es"]), ["en", "es"])


if __name__ == "__main__":
    unittest.main()
