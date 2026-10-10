import unittest

from modules.asr import languages


class TestNvidiaLanguageSet(unittest.TestCase):
    def test_has_the_25_european_languages(self):
        self.assertEqual(len(languages.NVIDIA_EU25), 25)
        self.assertIn("ro", languages.NVIDIA_EU25)
        self.assertIn("mt", languages.NVIDIA_EU25)
        self.assertNotIn("ja", languages.NVIDIA_EU25)

    def test_codes_are_normalised(self):
        self.assertEqual({languages.normalize_iso639_1(code) for code in languages.NVIDIA_EU25}, languages.NVIDIA_EU25)


class TestNormalizeIso6391(unittest.TestCase):
    def test_accepts_case_and_region_variants(self):
        for value in ("ro", "RO", " Ro ", "ro-RO", "ro_RO", "ro-latn-RO"):
            with self.subTest(value=value):
                self.assertEqual(languages.normalize_iso639_1(value), "ro")

    def test_rejects_names_garbage_and_non_strings(self):
        for value in ("romanian", "haw", "r", "", "r1", "és", "-ro", None, False, 12, ["ro"]):
            with self.subTest(value=value):
                self.assertIsNone(languages.normalize_iso639_1(value))


class TestFoldRomanianDiacritics(unittest.TestCase):
    def test_cedilla_forms_become_comma_below(self):
        self.assertEqual(languages.fold_romanian_diacritics("şţŞŢ"), "șțȘȚ")

    def test_other_text_is_unchanged(self):
        text = "Bună ziua, și însemnătăți!"
        self.assertEqual(languages.fold_romanian_diacritics(text), text)


if __name__ == "__main__":
    unittest.main()
