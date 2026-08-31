"""Fail CI/local gates when suppression patterns are introduced."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INCLUDE_SUFFIXES = {".py", ".pyw", ".ps1", ".psm1", ".psd1", ".toml", ".ini", ".yml", ".yaml"}
EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
}

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("noqa suppression", re.compile(r"#\s*noqa\b", re.IGNORECASE)),
    ("ruff file-level noqa", re.compile(r"#\s*ruff:\s*noqa\b", re.IGNORECASE)),
    ("flake8 file-level noqa", re.compile(r"#\s*flake8:\s*noqa\b", re.IGNORECASE)),
    ("nosec suppression", re.compile(r"#\s*nosec\b", re.IGNORECASE)),
    ("pragma no cover suppression", re.compile(r"pragma:\s*no\s*cover", re.IGNORECASE)),
    ("type ignore suppression", re.compile(r"#\s*type:\s*ignore\b", re.IGNORECASE)),
    ("pylint disable suppression", re.compile(r"#\s*pylint:\s*disable\b", re.IGNORECASE)),
    ("warning ignore filter", re.compile(r"warnings\.filterwarnings\(\s*['\"]ignore['\"]")),
    ("pytest warning ignore", re.compile(r"^\s*ignore::", re.IGNORECASE)),
    ("pytest ignore filter", re.compile(r"^\s*ignore(?:::|:)", re.IGNORECASE)),
    ("ruff per-file-ignores", re.compile(r"^\s*per-file-ignores\s*=", re.IGNORECASE)),
    ("ruff per-file-ignores table", re.compile(r"^\s*\[tool\.ruff(?:\.lint)?\.per-file-ignores\]\s*$", re.IGNORECASE)),
    ("ruff/flake8 extend-ignore", re.compile(r"^\s*extend-ignore\s*=", re.IGNORECASE)),
    (
        "powershell suppressmessage attribute",
        re.compile(r"\[\s*(?:System\.)?Diagnostics\.CodeAnalysis\.SuppressMessage(?:Attribute)?\s*\(", re.IGNORECASE),
    ),
]


def _iter_candidate_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in INCLUDE_SUFFIXES:
            continue

        parts = set(path.parts)
        if parts.intersection(EXCLUDE_DIR_NAMES):
            continue

        files.append(path)
    return files


def main() -> int:
    violations = _collect_violations()

    if violations:
        print("Suppression policy violations found:")
        for violation in violations:
            print(f"  - {violation}")
        return 1

    print("No suppression patterns found.")
    return 0


def _collect_violations() -> list[str]:
    """Scan candidate files and return matched suppression-policy violations."""
    violations: list[str] = []
    for file_path in _iter_candidate_files():
        relative_path = file_path.relative_to(ROOT)
        lines = _read_text_lines(file_path)
        _append_line_violations(relative_path, lines, violations)
    return violations


def _check_warning_call_violation(
    node: ast.Call,
    warning_module_aliases: set[str],
    direct_filter_names: set[str],
    relative_path: Path,
) -> str | None:
    """Return violation message if node represents an ignored warning filter call."""
    if not _is_warning_filter_call(node.func, warning_module_aliases, direct_filter_names):
        return None
    action = _resolve_warning_filter_action(node)
    if action != "ignore":
        return None
    call_target = _stringify_call_target(node.func)
    return f"{relative_path}:{getattr(node, 'lineno', '?')}: warning ignore filter detected: {call_target}"


def _collect_python_warning_filter_violations(relative_path: Path, file_text: str) -> list[str]:
    """Find warning-ignore filters including aliases and direct imports."""
    violations: list[str] = []
    try:
        tree = ast.parse(file_text)
    except SyntaxError:
        return violations

    warning_module_aliases, direct_filter_names = _collect_warning_filter_aliases(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            violation = _check_warning_call_violation(node, warning_module_aliases, direct_filter_names, relative_path)
            if violation:
                violations.append(violation)

    return violations


def _record_warning_import(node: ast.Import, aliases: set[str]) -> None:
    """Record warnings module alias from direct import."""
    for alias in node.names:
        if alias.name == "warnings":
            aliases.add(alias.asname or alias.name)


def _record_warning_import_from(node: ast.ImportFrom, direct_filters: set[str]) -> None:
    """Record direct warning filter imports."""
    if node.module != "warnings":
        return
    for alias in node.names:
        if alias.name in {"filterwarnings", "simplefilter"}:
            direct_filters.add(alias.asname or alias.name)


def _collect_warning_filter_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Collect warnings module aliases and direct filter function names."""
    warning_module_aliases = {"warnings"}
    direct_filter_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            _record_warning_import(node, warning_module_aliases)
        elif isinstance(node, ast.ImportFrom):
            _record_warning_import_from(node, direct_filter_names)

    return warning_module_aliases, direct_filter_names


def _is_warning_filter_call(func: ast.expr, warning_module_aliases: set[str], direct_filter_names: set[str]) -> bool:
    """Return True when call target resolves to warnings.filterwarnings/simplefilter."""
    if isinstance(func, ast.Attribute) and func.attr in {"filterwarnings", "simplefilter"}:
        return isinstance(func.value, ast.Name) and func.value.id in warning_module_aliases
    if isinstance(func, ast.Name):
        return func.id in direct_filter_names
    return False


def _resolve_keyword_warning_filter_action(keywords: list[ast.keyword]) -> str | None:
    """Resolve action argument from keyword arguments."""
    for keyword in keywords:
        if keyword.arg == "action":
            keyword_value = _extract_string_constant(keyword.value)
            if keyword_value:
                return keyword_value.lower()
    return None


def _resolve_warning_filter_action(call: ast.Call) -> str | None:
    """Resolve the warning filter action from positional/keyword call arguments."""
    if call.args:
        positional = _extract_string_constant(call.args[0])
        if positional:
            return positional.lower()

    return _resolve_keyword_warning_filter_action(call.keywords)


def _extract_string_constant(node: ast.expr) -> str | None:
    """Extract constant string values from AST nodes."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _stringify_call_target(func: ast.expr) -> str:
    """Return a readable call target name for diagnostics."""
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}(...)"
    if isinstance(func, ast.Name):
        return f"{func.id}(...)"
    return "<unknown-warning-call>"


def _read_text_lines(file_path: Path) -> list[str]:
    """Read text file lines with fallback replacement for bad bytes."""
    try:
        return file_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return file_path.read_text(encoding="utf-8", errors="replace").splitlines()


def _append_line_violations(relative_path: Path, lines: list[str], violations: list[str]) -> None:
    """Append all suppression-pattern violations for one file."""
    for line_number, line in enumerate(lines, start=1):
        for label, pattern in PATTERNS:
            if pattern.search(line):
                violations.append(f"{relative_path}:{line_number}: {label}: {line.strip()}")

    full_text = "\n".join(lines)
    violations.extend(_collect_python_warning_filter_violations(relative_path, full_text))


if __name__ == "__main__":
    sys.exit(main())
