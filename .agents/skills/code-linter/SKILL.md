______________________________________________________________________

## name: code-linter description: Run ruff, flake8, pylint, mypy, pyright, bandit, pip-audit, and radon complexity checks over repository files without suppression comments.

# Code Linter Skill

Use this skill to lint, format, type-check, and audit all Python modules, tests, scripts, Markdown files, and configurations without using suppression comments or inline ignores.

## Hard Rules

1. **Zero Suppressions Allowed**: Never add `# noqa`, `# type: ignore`, `# pylint: disable`, `# bandit: disable`, or warning-ignore filters.
1. `python tests/tools/check_no_suppressions.py` is mandatory and fails on any violation across product code, tests, and configs.
1. Keep Cyclomatic Complexity < 10 (Radon A-rank) across every single function.
1. Auto-format first with safe tools, then re-lint, and fix remaining errors manually.

## Workflow

### 1. Zero-Suppression Verification

```powershell
poetry run python tests/tools/check_no_suppressions.py
```

### 2. Auto-Formatting & Delinting

Always run safe auto-fixers before manual edits:

```powershell
# Python code formatting
poetry run ruff check --fix auto_subtitle.py modules tests
poetry run ruff format auto_subtitle.py modules tests

# Markdown documentation formatting
poetry run mdformat README.md AGENTS.md docs .github .agent .agents
```

### 3. Comprehensive Python Linting & Type Checking

```powershell
# Fast syntax & style linting
poetry run ruff check auto_subtitle.py modules tests
poetry run flake8 auto_subtitle.py modules tests

# Deep static analysis (Pylint)
poetry run pylint auto_subtitle.py modules tests

# Static Type Checking
poetry run mypy auto_subtitle.py modules
poetry run pyright
```

### 4. Cyclomatic Complexity & Maintainability (Radon)

```powershell
# Cyclomatic Complexity (Must be rank A, < 10)
poetry run radon cc -s -n B auto_subtitle.py modules tests

# Maintainability Index (Must be rank A)
poetry run radon mi auto_subtitle.py modules tests

# Halstead Metrics
poetry run radon hal auto_subtitle.py modules tests
```

### 5. Markdown Linting

```powershell
poetry run pymarkdown scan README.md AGENTS.md docs .github .agent .agents
```

### 6. Security & Dependency Auditing

```powershell
# Static Application Security Testing (Bandit)
poetry run bandit -c pyproject.toml -q -r auto_subtitle.py modules -lll -iii

# Known vulnerability scan for dependencies
poetry run pip-audit
```

## How to Fix Common Lint Failures

- **`C901` or Radon Complexity B/C/D**: The function has too many branching points (if/elif/else/for/while). Break it down into modular, single-responsibility helper functions.
- **Type Checking Errors (`mypy`/`pyright`)**: Add explicit type annotations (`Optional[Path]`, `tuple[str, ...]`, `dict[str, Any]`), handle `None` checks explicitly with `if var is None: return`, or cast safely using runtime guards.
- **Line Length**: Wrap function parameters, strings, or lists across multiple lines adhering to PEP 8 / Black standards (140-char max).
- **Security Alerts**: Avoid shell injection by passing argument sequences to `subprocess.Popen`/`subprocess.run` (never `shell=True` with raw user strings), and validate input paths.
