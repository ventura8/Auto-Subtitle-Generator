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
