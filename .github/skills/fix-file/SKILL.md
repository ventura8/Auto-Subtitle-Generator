______________________________________________________________________

## name: fix-file description: Apply a focused, minimal-diff fix to a target file end-to-end, validating complexity, zero suppressions, linters, and targeted tests.

# Fix File Skill

## Goal

Apply the smallest safe fix to a target file and validate it against project invariants.

## Precision Workflow

1. Identify the failing behavior and the minimal edit scope.
1. Apply focused changes without unrelated refactors.
1. Ensure complexity remains < 10 and no suppressions are introduced.
1. Run targeted linters, type checks, and unit tests.
1. Report exactly what changed and why.

## Validation Commands

```powershell
poetry run python tests/tools/check_no_suppressions.py
poetry run ruff check <target_file_path>
poetry run ruff format --check <target_file_path>
poetry run flake8 <target_file_path>
poetry run pylint <target_file_path>
poetry run pytest <target_test_path>
.\run_local_pipeline.ps1
```
