______________________________________________________________________

## name: fix-file description: Apply a focused, minimal-diff fix to a target file end-to-end, validating complexity, zero suppressions, linters, and targeted tests.

# Fix File Skill

Use this skill to apply the smallest safe fix to a target file while ensuring strict compliance with repository invariants.

## Precision Repair Workflow

### 1. Identify Minimal Scope

- Identify the exact root cause of the bug or lint failure.
- Avoid broad refactors or unrelated code cleanup in the same pass.

### 2. Apply Safe Changes

- Maintain Cyclomatic Complexity below 10 for every function.
- Do NOT add `# noqa`, `# type: ignore`, `# pylint: disable`, or warning filters.
- Preserve atomic file writing and resume semantics for subtitle files.
- Preserve Windows-safe subprocess execution and cleanup.

### 3. Validate the Touched File

```powershell
# Check zero suppressions
poetry run python tests/tools/check_no_suppressions.py

# Format and lint touched file
poetry run ruff check --fix <target_file_path>
poetry run ruff format <target_file_path>
poetry run flake8 <target_file_path>
poetry run pylint <target_file_path>

# Type check
poetry run mypy <target_file_path>
poetry run pyright <target_file_path>

# Radon Complexity check (< 10)
poetry run radon cc -s -n B <target_file_path>
```

### 4. Run Targeted Tests & Validate Coverage

```powershell
# Run matching test file
poetry run pytest tests/modules/test_<module_name>.py -v

# Check per-file coverage
poetry run coverage report --include=<target_file_path> --fail-under=90 -m
```

### 5. Final Full Gate Verification

Before completing the task, run the canonical pipeline:

```powershell
.\run_local_pipeline.ps1
```
