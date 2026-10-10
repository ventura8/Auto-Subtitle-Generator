"""Language codes shared by the NVIDIA ASR engines and their configuration."""

# The 25 European languages Canary-1B-v2 and Parakeet-TDT-0.6B-v3 were trained on.
NVIDIA_EU25 = frozenset("bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk".split())

# Legacy cedilla forms (ş ţ Ş Ţ) mapped to the correct Romanian comma-below letters (ș ț Ș Ț).
_ROMANIAN_COMMA_BELOW = str.maketrans({"ş": "ș", "ţ": "ț", "Ş": "Ș", "Ţ": "Ț"})


def normalize_iso639_1(value: object) -> str | None:
    """Return the lowercase two-letter code in ``value`` ("RO", "ro-RO", "ro_RO" -> "ro"), or None."""
    if not isinstance(value, str):
        return None
    code = value.strip().lower().replace("_", "-").split("-", 1)[0]
    if len(code) == 2 and code.isascii() and code.isalpha():
        return code
    return None


def fold_romanian_diacritics(text: str) -> str:
    """Replace cedilla s/t with the comma-below letters Romanian actually uses."""
    return text.translate(_ROMANIAN_COMMA_BELOW)
