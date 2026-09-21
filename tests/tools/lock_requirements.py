"""Write a pip-audit requirements file from poetry.lock for the *current* platform.

Auditing the installed environment misses whatever is not installed (CI
installs without the ``ml`` group, so torch was never audited there) and
audits stale local installs instead of the lock. Auditing the lock directly
needs care, because a package can appear several times with different
environment markers: ``torch`` is listed once for non-CPython interpreters
(2.10.0), once for macOS (2.14.0) and once for the CUDA wheel (2.14.0+cu132).
Taking the first entry per name audited a version this project never installs.

Each entry's ``markers`` are evaluated against the running interpreter and
platform; entries that do not apply are skipped, and when several still apply
the highest version wins.

Usage: python tests/tools/lock_requirements.py OUTPUT_PATH [LOCK_PATH]
"""

import sys
import tomllib
from pathlib import Path

from packaging.markers import InvalidMarker, Marker, UndefinedEnvironmentName
from packaging.version import InvalidVersion, Version

AUDITED_GROUPS = {"main", "ml"}
# Poetry writes this for an entry whose markers can never be satisfied (e.g. a
# duplicate version kept only for another resolution branch); it is never installed.
POETRY_EMPTY_MARKER = "<empty>"


def marker_applies(markers, groups=AUDITED_GROUPS):
    """Return whether a lock entry's markers hold here; unreadable markers count as applying.

    Poetry writes either one marker string or a table of per-group markers
    (``{ml = 'platform_system == "Windows"'}``); with a table the entry applies
    when the marker of any audited group applies.
    """
    if not markers:
        return True
    if isinstance(markers, dict):
        return _group_markers_apply(markers, groups)
    return _marker_text_applies(markers)


def _group_markers_apply(markers, groups):
    """Return whether any audited group's marker applies; no audited group means no restriction."""
    relevant = [text for group, text in markers.items() if group in groups]
    if not relevant:
        return True
    return any(_marker_text_applies(text) for text in relevant)


def _marker_text_applies(marker_text):
    """Evaluate one PEP 508 marker string against this interpreter and platform."""
    if not marker_text:
        return True
    if marker_text.strip() == POETRY_EMPTY_MARKER:
        return False
    try:
        return bool(Marker(marker_text).evaluate())
    except (InvalidMarker, UndefinedEnvironmentName, ValueError, TypeError):
        return True


def _version_key(version_text):
    """Sort key that tolerates non-PEP 440 versions by ranking them lowest."""
    try:
        return (1, Version(version_text))
    except InvalidVersion:
        return (0, Version("0"))


def select_requirements(lock_data, groups=AUDITED_GROUPS):
    """Return ``{name: version}`` for the lock entries that apply to this platform."""
    selected = {}
    for package in lock_data.get("package", []):
        if not _entry_applies(package, groups):
            continue
        name, version = package["name"], package["version"]
        if name not in selected or _version_key(version) > _version_key(selected[name]):
            selected[name] = version
    return selected


def _entry_applies(package, groups):
    """Return whether a lock entry is in an audited group, complete, and applicable here."""
    if not set(package.get("groups", [])) & groups:
        return False
    if not package.get("name") or not package.get("version"):
        return False
    return marker_applies(package.get("markers"), groups)


def write_requirements(output_path, lock_path="poetry.lock"):
    """Write ``name==version`` lines for the applicable lock entries; return their count."""
    lock_data = tomllib.loads(Path(lock_path).read_text(encoding="utf-8"))
    selected = select_requirements(lock_data)
    lines = [f"{name}=={version}" for name, version in sorted(selected.items())]
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main(argv):
    """CLI entry point."""
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    count = write_requirements(argv[1], argv[2] if len(argv) > 2 else "poetry.lock")
    print(f"lock_requirements: {count} packages for this platform -> {argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
