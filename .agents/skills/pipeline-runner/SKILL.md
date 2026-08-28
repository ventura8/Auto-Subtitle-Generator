______________________________________________________________________

## name: pipeline-runner description: Execute local pipeline checks, linting, security scans, unit tests, code coverage gates (>=90%), and metric reports on Windows/PowerShell.

# Local Pipeline Runner Skill

Use this skill to validate project code quality, formatting, security, unit tests, per-file code coverage, and maintainability metrics locally before committing changes.

## Instructions

1. **New files**: Before finishing a change set that adds files, run applicable lint, security, and tests on those paths. New paths are never exempt from quality gates.
1. Run pipeline commands with live CLI streaming and persistent log capture when troubleshooting.
1. Before hand-editing lint failures, always run automatic formatters / delinters with safe autofix first (e.g. `ruff check --fix`, `ruff format`, `mdformat`), re-lint, and only then manually fix remaining issues.
1. **Canonical Gate**: Always use `.\run_local_pipeline.ps1` from the repository root as the ultimate single source of truth for CI/local parity.

## Full Pipeline Command

```powershell
.\run_local_pipeline.ps1
```

## Pipeline Execution Stages & Invariants

The local pipeline enforces the following strict stages in sequence:

1. **Environment Pre-flight**:

   - Verifies `.venv\Scripts\python.exe` exists.
   - Requires dependencies installed via `.\install_dependencies.ps1`.

1. **Zero-Suppression Policy Check**:

   - Executes `python tests/tools/check_no_suppressions.py`.
   - **Hard Rule**: Scans all tracked python files, tests, and configuration for forbidden suppression directives (`# noqa`, `# type: ignore`, `pylint: disable`, warning ignore filters).
   - If any suppression is found, the gate fails immediately. Fix the underlying issue; never introduce suppressions.

1. **Markdown Quality (De-lint & Scan)**:

   - Formats docs using `mdformat README.md AGENTS.md docs .github .agent .agents`.
   - Scans using `pymarkdown scan README.md AGENTS.md docs .github .agent .agents`.

1. **Python Formatting & Linting**:

   - `ruff format --check auto_subtitle.py modules tests`
   - `ruff check auto_subtitle.py modules tests`
   - `flake8 auto_subtitle.py modules tests`
   - `pylint auto_subtitle.py modules tests`
   - `mypy auto_subtitle.py modules`
   - `pyright`

1. **Security Scanning**:

   - `bandit -c pyproject.toml -q -r auto_subtitle.py modules -lll -iii`
   - `pip-audit`

1. **Cyclomatic Complexity & Maintainability Metrics**:

   - `radon cc -s -n B auto_subtitle.py modules tests`: Enforces rank **A** (Cyclomatic Complexity < 10) across all functions.
   - `radon mi auto_subtitle.py modules tests`: Enforces Maintainability Index rank **A**.
   - `radon hal auto_subtitle.py modules tests`: Generates Halstead metrics report.

1. **Test Suite & Code Coverage**:

   - Runs `pytest` with `--cov=auto_subtitle --cov=modules --cov-branch --cov-fail-under=90`.
   - **Per-file coverage >= 90%**: Explicitly enforces that every individual core module achieves >= 90% line and branch coverage:
     - `auto_subtitle.py`
     - `modules/configuration/config.py`
     - `modules/pipeline/isolated_translator.py`
     - `modules/models.py`
     - `modules/pipeline/transcription.py`
     - `modules/pipeline/translation.py`
     - `modules/utils.py`

1. **Artifact & Badge Generation**:

   - Generates `assets/coverage.svg` via `genbadge`.
   - Updates `coverage_summary.md` and metrics reports.

## Troubleshooting Failures

- **Coverage drops below 90%**: Inspect uncovered lines in `coverage_summary.md` or `coverage.json`, and add targeted unit tests under `tests/modules/` or `tests/orchestration/`.
- **Complexity >= 10 (Radon B-rank)**: Split complex branching or loops into small, focused, pure helper functions. Do not suppress.
- **Suppression scanner failure**: Remove any `# noqa`, `# type: ignore`, or `# pylint: disable`. Fix the actual type signature or linter warning.
- **Subprocess or mock issue**: Ensure subprocess calls mock `subprocess.Popen` / `subprocess.run` cleanly and handle Windows exit codes and process cleanup correctly.
