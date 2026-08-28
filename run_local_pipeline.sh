#!/usr/bin/env bash
# Canonical Linux / macOS Quality Gate for Auto-Subtitle-Generator
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PY="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$VENV_PY" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PY." >&2
    echo "Please run ./install_dependencies.sh first." >&2
    exit 1
fi

invoke_poetry() {
    "$VENV_PY" -m poetry "$@"
}

echo "==> Step 1: Initialize & Verify Poetry"
invoke_poetry --version

echo "==> Step 2: Validate Poetry lockfile"
invoke_poetry check

echo "==> Step 3: Ensure test dependencies (main + dev, no ml)"
invoke_poetry install --only main,dev --no-root

echo "==> Step 4: Enforce zero-suppression policy"
invoke_poetry run python tests/tools/check_no_suppressions.py

if command -v pwsh >/dev/null 2>&1; then
    echo "==> Step 5: Run PowerShell Lint (via pwsh)"
    pwsh -NoProfile -Command '& ./.github/scripts/Invoke-PowerShellLint.ps1 -ScriptPaths @("./install_dependencies.ps1", "./run_local_pipeline.ps1", "./.github/scripts/Invoke-PowerShellLint.ps1") -MaxCyclomaticComplexity 9 -MaxNestingDepth 4'
fi

echo "==> Step 6: Run Markdown auto-delinter (mdformat)"
invoke_poetry run mdformat --check README.md AGENTS.md docs .github

echo "==> Step 7: Run Markdown linter (pymarkdown)"
invoke_poetry run pymarkdown scan README.md AGENTS.md docs .github

echo "==> Step 8: Run isort import order check"
invoke_poetry run isort --check-only --filter-files auto_subtitle.py modules tests

echo "==> Step 9: Run Black format check"
invoke_poetry run black --check auto_subtitle.py modules tests

echo "==> Step 10: Run Taplo format check"
invoke_poetry run taplo format --check pyproject.toml poetry.toml

echo "==> Step 11: Run Ruff"
invoke_poetry run ruff check modules auto_subtitle.py tests

echo "==> Step 11b: Run Ruff format check"
invoke_poetry run ruff format --check .

echo "==> Step 12: Run Flake8"
invoke_poetry run flake8 modules auto_subtitle.py tests

echo "==> Step 13: Run Pylint"
invoke_poetry run pylint modules auto_subtitle.py

echo "==> Step 14: Run Pylint on tests (errors-only)"
PYTHONPATH=. invoke_poetry run pylint tests --errors-only

echo "==> Step 15: Run mypy"
invoke_poetry run mypy auto_subtitle.py modules tests

echo "==> Step 16: Run pyright"
invoke_poetry run pyright auto_subtitle.py modules tests

echo "==> Step 17: Run Bandit Security Scan"
invoke_poetry run bandit -c pyproject.toml -q -r auto_subtitle.py modules -lll -iii

echo "==> Step 18: Run dependency vulnerability scan (pip-audit)"
invoke_poetry run pip-audit

echo "==> Step 20: Run Radon Complexity (A-grade enforced)"
invoke_poetry run radon cc auto_subtitle.py modules tests/modules tests/orchestration -s -a | tee radon_report.txt
if grep -E "[[:space:]]-[[:space:]][B-F][[:space:]]\(" radon_report.txt >/dev/null; then
    echo "ERROR: Radon complexity gate failed. All functions/methods must be grade A (scores 1-5)." >&2
    exit 1
fi

echo "==> Step 21: Run Radon Maintainability Index (A-grade enforced)"
invoke_poetry run radon mi auto_subtitle.py modules tests/modules tests/orchestration -s | tee radon_mi_report.txt
if grep -E "[[:space:]]-[[:space:]][B-F][[:space:]]\(" radon_mi_report.txt >/dev/null; then
    echo "ERROR: Radon MI gate failed. All files must have A-grade maintainability." >&2
    exit 1
fi

echo "==> Step 22: Run Radon Halstead Metrics"
invoke_poetry run radon hal auto_subtitle.py modules tests/modules tests/orchestration | tee radon_hal_report.txt

echo "==> Step 23: Run Tests with Coverage"
invoke_poetry run pytest -m "not e2e" -o addopts= --strict-config --strict-markers --cov=auto_subtitle --cov=modules --cov-branch --cov-report=xml --cov-report=json --cov-report=term --cov-fail-under=90 tests/

echo "==> Step 24: Enforce per-file coverage >= 90%"
for cov_file in "auto_subtitle.py" "modules/configuration/config.py" "modules/pipeline/isolated_translator.py" "modules/models.py" "modules/pipeline/transcription.py" "modules/pipeline/translation.py" "modules/utils.py"; do
    echo "   -> Checking $cov_file"
    invoke_poetry run coverage report --include="$cov_file" --fail-under=90 -m
done

echo "==> Step 25: Generate Badge and Summary"
if [ -f "coverage.xml" ]; then
    invoke_poetry run genbadge coverage -i coverage.xml -o assets/coverage.svg
    invoke_poetry run python tests/tools/transform_metrics.py coverage.xml
fi

echo -e "\n=== Local validation pipeline passed successfully! ==="
