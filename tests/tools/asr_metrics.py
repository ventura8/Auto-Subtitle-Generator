"""Scoring helpers for the ASR benchmark: text normalisers, corpus WER/CER and a paired bootstrap.

Two normalisers are reported side by side. ``normalise_keep_diacritics`` keeps
Romanian diacritics (after folding the cedilla variants Whisper often emits to
the comma-below forms FLEURS uses), so a dropped ``ș`` counts as an error.
``normalise_open_asr`` strips every combining mark, as the Open ASR
leaderboard does, so only recognition errors on the base letters count.

Punctuation is replaced by a space rather than deleted, so ``într-o`` scores
as two words under both normalisers, consistently for reference and hypothesis.
"""

import difflib
import importlib
import itertools
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence

CEDILLA_TO_COMMA_BELOW = str.maketrans({"ş": "ș", "ţ": "ț", "Ş": "Ș", "Ţ": "Ț"})
ROMANIAN_DIACRITICS = frozenset("ăâîșț")  # ă â î ș ț
UNITS = ("word", "char")

# Anything that is neither a word character nor whitespace, plus the underscore \w lets through.
# Covers „ ” « » — … as well as ASCII punctuation.
_PUNCTUATION = re.compile(r"[^\w\s]|_")
_WHITESPACE = re.compile(r"\s+")

Pair = tuple[str, str]


def _strip_punctuation(text: str) -> str:
    """Replace punctuation with spaces and collapse runs of whitespace."""
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", text)).strip()


def normalise_keep_diacritics(text: str) -> str:
    """NFC, lowercase, fold cedilla ş/ţ to comma-below ș/ț and strip punctuation; diacritics are kept."""
    folded = unicodedata.normalize("NFC", text).translate(CEDILLA_TO_COMMA_BELOW)
    return _strip_punctuation(folded.lower())


def normalise_open_asr(text: str) -> str:
    """NFKD, drop combining marks, lowercase and strip punctuation (Open ASR style, diacritics removed)."""
    decomposed = unicodedata.normalize("NFKD", text)
    base = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _strip_punctuation(base.lower())


def _edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """Levenshtein distance between two token sequences, iterative with two rows."""
    previous = list(range(len(hyp) + 1))
    for row, ref_token in enumerate(ref, start=1):
        current = [row]
        for column, hyp_token in enumerate(hyp, start=1):
            current.append(min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + (ref_token != hyp_token)))
        previous = current
    return previous[-1]


def _tokens(text: str, unit: str) -> list[str]:
    """Split already-normalised text into words or characters (spaces count as characters)."""
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    return text.split() if unit == "word" else list(text)


def error_counts(ref: str, hyp: str, unit: str) -> tuple[int, int]:
    """Return ``(errors, reference length)`` in ``unit`` ("word" or "char") for normalised texts."""
    ref_tokens = _tokens(ref, unit)
    return _edit_distance(ref_tokens, _tokens(hyp, unit)), len(ref_tokens)


def _totals(counts: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """Sum a list of ``(errors, length)`` pairs."""
    return sum(count[0] for count in counts), sum(count[1] for count in counts)


def _rate(errors: int, ref_len: int) -> float:
    """Errors over reference length; an empty reference scores 0.0 without errors and infinity with them."""
    if ref_len == 0:
        return 0.0 if errors == 0 else float("inf")
    return errors / ref_len


def corpus_rate(pairs: Iterable[Pair], normaliser: Callable[[str], str], unit: str) -> float:
    """Corpus error rate: total edits over total reference length, not a mean of per-utterance rates.

    An empty reference corpus scores 0.0 without errors and infinity with them.
    """
    return _rate(*_totals([error_counts(normaliser(ref), normaliser(hyp), unit) for ref, hyp in pairs]))


def _block_diacritic_errors(ref_block: str, hyp_block: str) -> int:
    """Count diacritic reference characters a non-equal alignment block got wrong (substituted or deleted)."""
    paired = itertools.zip_longest(ref_block, hyp_block)
    return sum(1 for ref_char, hyp_char in paired if ref_char in ROMANIAN_DIACRITICS and ref_char != hyp_char)


def _diacritic_counts(ref: str, hyp: str) -> tuple[int, int]:
    """Return ``(errors, diacritic characters in ref)`` from a character alignment of two normalised texts."""
    matcher = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    errors = sum(_block_diacritic_errors(ref[i1:i2], hyp[j1:j2]) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")
    return errors, sum(1 for char in ref if char in ROMANIAN_DIACRITICS)


def diacritic_error_rate(pairs: Iterable[Pair]) -> float:
    """Share of reference ă â î ș ț characters the hypothesis did not reproduce at the aligned position.

    Both sides go through ``normalise_keep_diacritics``. Characters are aligned
    with ``difflib.SequenceMatcher`` (no junk heuristic); inside a replaced
    block, characters pair up by position and unpaired reference characters are
    deletions. A corpus without diacritics scores 0.0.
    """
    errors, total = _totals([_diacritic_counts(normalise_keep_diacritics(ref), normalise_keep_diacritics(hyp)) for ref, hyp in pairs])
    return errors / total if total else 0.0


def paired_cluster_bootstrap(err_a, err_b, words, clusters, n=10_000, seed=0) -> tuple[float, float]:
    """95 % CI of ``rate(a) - rate(b)`` resampling whole clusters (FLEURS sentence ids) with replacement.

    ``err_a``/``err_b`` are per-utterance edit counts of the two systems,
    ``words`` the per-utterance reference lengths and ``clusters`` the cluster
    id of each utterance. Utterances of one sentence read by several speakers
    are not independent, so they are always drawn together.
    """
    np = importlib.import_module("numpy")
    _ids, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    sums_a, sums_b, sums_words = (np.bincount(inverse, weights=np.asarray(values, dtype=float)) for values in (err_a, err_b, words))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n)
    for index in range(n):
        pick = rng.integers(0, len(sums_words), len(sums_words))
        deltas[index] = (sums_a[pick].sum() - sums_b[pick].sum()) / sums_words[pick].sum()
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)
