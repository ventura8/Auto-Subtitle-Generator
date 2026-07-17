______________________________________________________________________

## name: fix-file user-invocable: true description: Use when you need a targeted single-file fix workflow for one file end-to-end, including lint checks, impacted tests, and minimal safe edits.

______________________________________________________________________

# Fix File Skill

## Goal

Apply the smallest safe fix to a target file and validate it.

## Workflow

1. Identify the failing behavior and the minimal edit scope.
1. Apply focused changes without unrelated refactors.
1. Run targeted tests first, then broader checks if needed.
1. Report exactly what changed and why.

## Commands

```powershell
poetry run python tests/tools/check_no_suppressions.py
poetry run ruff check <target_file_path>
poetry run ruff format --check <target_file_path>
poetry run flake8 <target_file_path>
poetry run pytest <target_test_path>
.\run_local_pipeline.ps1
```

## Project-Specific Guardrails

- Keep orchestration in auto_subtitle.py and logic in modules/.
- Do not weaken shutdown and process cleanup behavior.
- Preserve resume and atomic-save semantics.
- Keep complexity below 10.
- Never add suppression patterns (`noqa`, `type: ignore`, warning-ignore filters).
