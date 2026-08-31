"""Discovery helpers for resumable subtitle artifacts."""

import os


def extract_srt_language(filename, prefix):
    """Extract a language token from a matching SRT filename."""
    if filename.startswith(prefix) and filename.endswith(".srt") and not filename.endswith(".tmp"):
        return filename[len(prefix) : -4] or None
    return None


def find_existing_srt_languages(folder, base_name):
    """Scan a folder for existing <base_name>.<lang>.srt files."""
    prefix = f"{base_name}."
    candidates = []
    try:
        for filename in sorted(os.listdir(folder)):
            lang = extract_srt_language(filename, prefix)
            if lang:
                candidates.append(lang)
    except OSError:
        pass
    return candidates


def is_usable_language(language):
    """Return whether a language value identifies a language rather than an unknown marker."""
    normalized = str(language or "").strip().lower()
    return bool(normalized) and normalized not in {"und", "undetermined", "unknown"}


def get_usable_languages(languages):
    """Return discovered language codes excluding unknown markers."""
    return [language for language in languages if is_usable_language(language)]


def prioritize_recorded_language(recorded_source_lang, discovered_languages):
    """Put a usable recorded language first and retain other usable discoveries."""
    usable_languages = get_usable_languages(discovered_languages)
    if not is_usable_language(recorded_source_lang):
        return usable_languages
    return [recorded_source_lang, *[lang for lang in usable_languages if lang != recorded_source_lang]]
