"""Tests for the NVIDIA ASR cue builder in modules.asr.cues."""

import unittest

from modules.asr import cues
from modules.subtitles.timestamp_utils import format_timestamp


def _stamp(token, start, end):
    """Build one Parakeet decode stamp."""
    return {"token": token, "start": start, "end": end}


def _word(start, end, text):
    """Build one timed word."""
    return (start, end, text)


class TestWordsFromStamps(unittest.TestCase):
    def test_merges_chunks_into_words(self):
        stamps = [_stamp(" Bună", 0.0, 0.32), _stamp(" zi", 0.4, 0.56), _stamp("ua", 0.56, 0.8), _stamp(",", 0.8, 0.8)]
        words = cues.words_from_stamps(stamps, 0.0)
        self.assertEqual([word[2] for word in words], ["Bună", "ziua,"])
        self.assertEqual(words[0][:2], (0.0, 0.32))
        self.assertEqual(words[1][:2], (0.4, 0.8))

    def test_leading_token_without_space_starts_a_word(self):
        words = cues.words_from_stamps([_stamp("Da", 0.1, 0.3), _stamp(" sigur", 0.4, 0.7)], 0.0)
        self.assertEqual(words, [(0.1, 0.3, "Da"), (0.4, 0.7, "sigur")])

    def test_punctuation_keeps_word_end(self):
        words = cues.words_from_stamps([_stamp(" gata", 1.0, 1.4), _stamp(".", 1.4, 1.4)], 0.0)
        self.assertEqual(words, [(1.0, 1.4, "gata.")])

    def test_offset_is_applied(self):
        words = cues.words_from_stamps([_stamp(" salut", 0.08, 0.4)], 12.5)
        self.assertAlmostEqual(words[0][0], 12.58)
        self.assertAlmostEqual(words[0][1], 12.9)

    def test_empty_and_partial_tokens(self):
        self.assertEqual(cues.words_from_stamps([], 3.0), [])
        stamps = [_stamp(None, 0.0, 0.1), _stamp("", 0.1, 0.1), _stamp(" ", 0.1, 0.2), _stamp(" ok", 0.3, 0.5)]
        self.assertEqual(cues.words_from_stamps(stamps, 0.0), [(0.3, 0.5, "ok")])

    def test_whitespace_word_absorbs_following_chunk(self):
        words = cues.words_from_stamps([_stamp(" ", 0.0, 0.1), _stamp("abc", 0.1, 0.3)], 0.0)
        self.assertEqual(words, [(0.0, 0.3, "abc")])

    def test_romanian_comma_below_preserved(self):
        words = cues.words_from_stamps([_stamp(" și", 0.0, 0.2), _stamp(" ța", 0.3, 0.4), _stamp("ră", 0.4, 0.6)], 0.0)
        self.assertEqual([word[2] for word in words], ["și", "țară"])


class TestCuesFromWords(unittest.TestCase):
    def test_empty_words(self):
        self.assertEqual(cues.cues_from_words([]), [])

    def test_splits_on_sentence_end(self):
        words = [_word(0.0, 0.3, "Bună"), _word(0.35, 0.7, "ziua."), _word(0.75, 1.0, "Ce"), _word(1.05, 1.3, "faci?")]
        self.assertEqual(cues.cues_from_words(words), [(0.0, 0.7, "Bună ziua."), (0.75, 1.3, "Ce faci?")])

    def test_sentence_end_inside_closing_quote(self):
        words = [_word(0.0, 0.3, "„Gata.”"), _word(0.35, 0.7, "Apoi")]
        self.assertEqual(len(cues.cues_from_words(words)), 2)

    def test_splits_on_pause(self):
        words = [_word(0.0, 0.3, "unu"), _word(0.35, 0.6, "doi"), _word(1.1, 1.4, "trei")]
        result = cues.cues_from_words(words)
        self.assertEqual(result, [(0.0, 0.6, "unu doi"), (1.1, 1.4, "trei")])

    def test_short_gap_does_not_split(self):
        words = [_word(0.0, 0.3, "unu"), _word(0.79, 1.0, "doi")]
        self.assertEqual(cues.cues_from_words(words), [(0.0, 1.0, "unu doi")])

    def test_splits_before_char_overflow(self):
        limits = cues.Limits(10, 7.0, 0.5, 0.8)
        words = [_word(0.0, 0.2, "abcd"), _word(0.2, 0.4, "efgh"), _word(0.4, 0.6, "ij")]
        result = cues.cues_from_words(words, limits)
        self.assertEqual([cue[2] for cue in result], ["abcd efgh", "ij"])

    def test_exact_char_limit_fits(self):
        limits = cues.Limits(9, 7.0, 0.5, 0.8)
        words = [_word(0.0, 0.2, "abcd"), _word(0.2, 0.4, "efgh")]
        self.assertEqual([cue[2] for cue in cues.cues_from_words(words, limits)], ["abcd efgh"])

    def test_splits_before_duration_overflow(self):
        words = [_word(float(second), second + 0.6, "da") for second in range(10)]
        result = cues.cues_from_words(words)
        self.assertEqual(len(result), 2)
        self.assertLessEqual(result[0][1] - result[0][0], cues.DEFAULT_LIMITS.max_seconds)
        self.assertEqual(result[1][0], 7.0)

    def test_single_overlong_word_stands_alone(self):
        limits = cues.Limits(5, 7.0, 0.5, 0.8)
        words = [_word(0.0, 0.2, "ab"), _word(0.2, 0.6, "abcdefghij"), _word(0.6, 0.8, "cd")]
        self.assertEqual([cue[2] for cue in cues.cues_from_words(words, limits)], ["ab", "abcdefghij", "cd"])

    def test_cue_text_preserves_diacritics(self):
        words = [_word(0.0, 0.3, "Știți"), _word(0.35, 0.7, "țara?")]
        self.assertEqual(cues.cues_from_words(words), [(0.0, 0.7, "Știți țara?")])


class TestCuesFromText(unittest.TestCase):
    def test_whitespace_only_text(self):
        self.assertEqual(cues.cues_from_text("  \n\t ", (0.0, 5.0), []), [])

    def test_short_text_gives_one_cue(self):
        self.assertEqual(cues.cues_from_text("  Bună  ziua ", (2.0, 4.0), []), [(2.0, 4.0, "Bună ziua")])

    def test_three_sentences_are_proportional_and_contiguous(self):
        text = "Aaaa aaaa. Bbbb bbbb. Cccc cccc."
        result = cues.cues_from_text(text, (10.0, 16.0), [])
        self.assertEqual([cue[2] for cue in result], ["Aaaa aaaa.", "Bbbb bbbb.", "Cccc cccc."])
        self.assertEqual(result[0][0], 10.0)
        self.assertAlmostEqual(result[0][1], 12.0)
        self.assertAlmostEqual(result[1][1], 14.0)
        self.assertEqual(result[-1][1], 16.0)
        self.assertEqual([cue[1] for cue in result[:-1]], [cue[0] for cue in result[1:]])

    def test_boundary_snaps_to_nearest_pause_midpoint(self):
        result = cues.cues_from_text("Aaaa aaaa. Bbbb bbbb.", (0.0, 6.0), [(1.0, 1.4), (3.4, 4.0), (5.0, 5.2)])
        self.assertAlmostEqual(result[0][1], 3.7)
        self.assertAlmostEqual(result[1][0], 3.7)

    def test_pause_beyond_snap_window_is_ignored(self):
        result = cues.cues_from_text("Aaaa aaaa. Bbbb bbbb.", (0.0, 6.0), [(4.2, 4.4)])
        self.assertAlmostEqual(result[0][1], 3.0)

    def test_pause_at_snap_window_edge_is_used(self):
        result = cues.cues_from_text("Aaaa aaaa. Bbbb bbbb.", (0.0, 6.0), [(3.9, 4.1)])
        self.assertAlmostEqual(result[0][1], 4.0)

    def test_long_text_without_punctuation_is_wrapped(self):
        text = " ".join(["cuvânt"] * 30)
        result = cues.cues_from_text(text, (0.0, 6.0), [])
        self.assertGreater(len(result), 1)
        self.assertTrue(all(len(cue[2]) <= cues.DEFAULT_LIMITS.max_chars for cue in result))
        self.assertEqual(" ".join(cue[2] for cue in result), text)

    def test_long_sentence_splits_at_clauses(self):
        text = "Primul segment este destul de lung, al doilea segment este tot lung, iar al treilea încheie fraza aici."
        result = cues.cues_from_text(text, (0.0, 6.0), [], cues.Limits(40, 7.0, 0.5, 0.8))
        self.assertEqual(result[0][2], "Primul segment este destul de lung,")
        self.assertEqual(" ".join(cue[2] for cue in result), text)

    def test_short_clauses_are_packed_together(self):
        text = "Unu, doi, trei, patru, cinci, șase, șapte, opt, nouă, zece."
        result = cues.cues_from_text(text, (0.0, 6.0), [], cues.Limits(30, 7.0, 0.5, 0.8))
        self.assertEqual([cue[2] for cue in result], ["Unu, doi, trei, patru, cinci,", "șase, șapte, opt, nouă, zece."])

    def test_long_span_splits_by_duration(self):
        text = "Aceasta este o propoziție scurtă dar rară"
        result = cues.cues_from_text(text, (0.0, 20.0), [])
        self.assertGreater(len(result), 1)
        self.assertEqual(result[-1][1], 20.0)

    def test_single_long_word_stays_whole(self):
        word = "a" * 100
        self.assertEqual(cues.cues_from_text(word, (0.0, 3.0), []), [(0.0, 3.0, word)])

    def test_min_seconds_enforced_when_span_allows(self):
        text = "Da. " + "Bine, mulțumesc foarte mult pentru tot ce ați făcut astăzi pentru noi."
        result = cues.cues_from_text(text, (0.0, 6.0), [])
        self.assertEqual(result[0][2], "Da.")
        self.assertAlmostEqual(result[0][1] - result[0][0], 0.8)

    def test_min_seconds_shrinks_for_short_span(self):
        result = cues.cues_from_text("A. B. C. D.", (0.0, 1.0), [])
        self.assertEqual(len(result), 4)
        for start, end, _text in result:
            self.assertGreaterEqual(end - start, 0.25 - 1e-9)
        self.assertEqual(result[-1][1], 1.0)

    def test_snapped_boundaries_stay_monotonic(self):
        result = cues.cues_from_text("Aa. Bb. Cc. Dd.", (0.0, 4.0), [(0.4, 0.6)])
        starts = [cue[0] for cue in result]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(all(end - start >= 0.8 - 1e-9 for start, end, _text in result))
        self.assertEqual(result[-1][1], 4.0)

    def test_inverted_span_does_not_go_backwards(self):
        result = cues.cues_from_text("Aa. Bb.", (5.0, 4.0), [])
        self.assertEqual([cue[:2] for cue in result], [(5.0, 5.0), (5.0, 5.0)])

    def test_diacritics_preserved(self):
        result = cues.cues_from_text("Știm că țara e frumoasă. Așa e.", (0.0, 4.0), [])
        self.assertEqual([cue[2] for cue in result], ["Știm că țara e frumoasă.", "Așa e."])


class TestFinalizeCues(unittest.TestCase):
    def test_drops_empty_and_strips(self):
        result = cues.finalize_cues([(0.0, 1.0, "  "), (1.0, 2.0, " text ")])
        self.assertEqual(result, [(1.0, 2.0, "text")])

    def test_sorts_and_removes_overlap(self):
        result = cues.finalize_cues([(2.0, 3.0, "b"), (0.0, 2.5, "a")])
        self.assertEqual(result, [(0.0, 2.5, "a"), (2.5, 3.0, "b")])

    def test_end_after_start_after_ms_rounding(self):
        result = cues.finalize_cues([(1.0001, 1.0004, "x"), (1.0004, 1.0002, "y")])
        for start, end, _text in result:
            self.assertGreater(format_timestamp(end), format_timestamp(start))
        self.assertEqual(result[0], (1.0, 1.001, "x"))
        self.assertEqual(result[1], (1.001, 1.002, "y"))

    def test_negative_start_clamped(self):
        self.assertEqual(cues.finalize_cues([(-0.2, 0.5, "a")]), [(0.0, 0.5, "a")])

    def test_empty_input(self):
        self.assertEqual(cues.finalize_cues([]), [])


if __name__ == "__main__":
    unittest.main()
