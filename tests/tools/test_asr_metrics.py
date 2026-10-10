import importlib.util
import os
import unittest


def _load_asr_metrics_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    module_path = os.path.join(repo_root, "tests", "tools", "asr_metrics.py")
    spec = importlib.util.spec_from_file_location("asr_metrics_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load tests/tools/asr_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


METRICS = _load_asr_metrics_module()
_NUMPY_INSTALLED = importlib.util.find_spec("numpy") is not None


class TestNormalisers(unittest.TestCase):
    def test_keep_diacritics_folds_cedilla_to_comma_below(self):
        self.assertEqual(METRICS.normalise_keep_diacritics("Şcoala ţării"), "școala țării")
        self.assertEqual(METRICS.normalise_keep_diacritics("ŞŢ"), "șț")

    def test_keep_diacritics_composes_decomposed_input(self):
        # "a" + combining breve must compare equal to the precomposed "ă".
        self.assertEqual(METRICS.normalise_keep_diacritics("măr"), "măr")

    def test_keep_diacritics_strips_typographic_punctuation(self):
        text = "„Bună ziua”, a spus el — «salut»… Totul_e bine?!"
        self.assertEqual(METRICS.normalise_keep_diacritics(text), "bună ziua a spus el salut totul e bine")

    def test_punctuation_becomes_a_word_boundary(self):
        self.assertEqual(METRICS.normalise_keep_diacritics("într-o   zi"), "într o zi")

    def test_open_asr_drops_every_diacritic(self):
        self.assertEqual(METRICS.normalise_open_asr("Şcoala, ȘTIINȚĂ și Îngheţată!"), "scoala stiinta si inghetata")

    def test_open_asr_applies_compatibility_decomposition(self):
        self.assertEqual(METRICS.normalise_open_asr("ﬁnal ①"), "final 1")


class TestEditDistance(unittest.TestCase):
    def test_classic_examples(self):
        self.assertEqual(METRICS._edit_distance("kitten", "sitting"), 3)
        self.assertEqual(METRICS._edit_distance("", "abc"), 3)
        self.assertEqual(METRICS._edit_distance("abc", ""), 3)
        self.assertEqual(METRICS._edit_distance(["a", "b"], ["a", "b"]), 0)

    def test_word_and_char_units(self):
        self.assertEqual(METRICS.error_counts("ana are mere", "ana are pere multe", "word"), (2, 3))
        self.assertEqual(METRICS.error_counts("ab c", "abc", "char"), (1, 4))

    def test_unknown_unit_raises(self):
        with self.assertRaisesRegex(ValueError, "unit must be one of"):
            METRICS.error_counts("a", "b", "phoneme")


class TestCorpusRate(unittest.TestCase):
    def test_sums_errors_over_the_corpus(self):
        # Per-utterance mean would be (1/1 + 0/9) / 2 = 0.5; the corpus rate is 1/10.
        pairs = [("unu", "doi"), ("a b c d e f g h i", "a b c d e f g h i")]
        self.assertAlmostEqual(METRICS.corpus_rate(pairs, METRICS.normalise_keep_diacritics, "word"), 0.1)

    def test_normaliser_decides_whether_diacritics_count(self):
        pairs = [("Școala e mare.", "scoala e mare")]
        self.assertAlmostEqual(METRICS.corpus_rate(pairs, METRICS.normalise_keep_diacritics, "word"), 1 / 3)
        self.assertEqual(METRICS.corpus_rate(pairs, METRICS.normalise_open_asr, "word"), 0.0)
        self.assertAlmostEqual(METRICS.corpus_rate(pairs, METRICS.normalise_keep_diacritics, "char"), 1 / 13)

    def test_cedilla_variant_is_not_an_error(self):
        pairs = [("ţară şi", "țară și")]
        self.assertEqual(METRICS.corpus_rate(pairs, METRICS.normalise_keep_diacritics, "word"), 0.0)

    def test_empty_reference_corpus(self):
        self.assertEqual(METRICS.corpus_rate([("", "")], METRICS.normalise_open_asr, "word"), 0.0)
        self.assertEqual(METRICS.corpus_rate([("...", "ceva")], METRICS.normalise_open_asr, "word"), float("inf"))
        self.assertEqual(METRICS.corpus_rate([], METRICS.normalise_open_asr, "word"), 0.0)


class TestDiacriticErrorRate(unittest.TestCase):
    def test_counts_substituted_and_deleted_diacritics(self):
        # Reference diacritics: ș, ă, ț, ă -> four; hypothesis drops ș and ț.
        pairs = [("Școală", "scoală"), ("țară", "tară")]
        self.assertAlmostEqual(METRICS.diacritic_error_rate(pairs), 0.5)

    def test_deleted_diacritic_inside_a_longer_replacement(self):
        # "îî" -> "x": the first î is substituted, the second deleted.
        self.assertEqual(METRICS.diacritic_error_rate([("aîîb", "axb")]), 1.0)

    def test_errors_on_plain_letters_are_ignored(self):
        self.assertEqual(METRICS.diacritic_error_rate([("mână bună", "până bună")]), 0.0)

    def test_insertions_and_cedilla_variants_are_not_errors(self):
        self.assertEqual(METRICS.diacritic_error_rate([("şi", "și foarte")]), 0.0)

    def test_corpus_without_diacritics_scores_zero(self):
        self.assertEqual(METRICS.diacritic_error_rate([("abc", "xyz")]), 0.0)
        self.assertEqual(METRICS.diacritic_error_rate([]), 0.0)


@unittest.skipUnless(_NUMPY_INSTALLED, "numpy comes with the ml dependency group")
class TestPairedClusterBootstrap(unittest.TestCase):
    def test_identical_systems_give_a_zero_interval(self):
        errors = [1, 2, 0, 3]
        low, high = METRICS.paired_cluster_bootstrap(errors, errors, [10, 10, 10, 10], ["a", "a", "b", "c"], n=200)
        self.assertEqual((low, high), (0.0, 0.0))

    def test_constant_difference_is_recovered_exactly(self):
        # System a makes one more error per 10-word utterance than b in every cluster: delta 0.1.
        low, high = METRICS.paired_cluster_bootstrap([3, 2, 1], [2, 1, 0], [10, 10, 10], [1, 2, 3], n=500)
        self.assertAlmostEqual(low, 0.1)
        self.assertAlmostEqual(high, 0.1)

    def test_interval_brackets_the_point_estimate_and_is_reproducible(self):
        err_a = [5, 1, 4, 0, 6, 2]
        err_b = [2, 1, 1, 0, 3, 2]
        words = [20, 18, 22, 15, 25, 19]
        clusters = [7, 7, 8, 9, 9, 10]
        first = METRICS.paired_cluster_bootstrap(err_a, err_b, words, clusters, n=2000, seed=3)
        second = METRICS.paired_cluster_bootstrap(err_a, err_b, words, clusters, n=2000, seed=3)
        point = (sum(err_a) - sum(err_b)) / sum(words)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], point)
        self.assertGreaterEqual(first[1], point)
        self.assertLess(first[0], first[1])
        self.assertIsInstance(first[0], float)


if __name__ == "__main__":
    unittest.main()
