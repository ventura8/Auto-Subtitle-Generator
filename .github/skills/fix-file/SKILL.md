---
name: fix-file
user-invocable: true
description: "Use when you need to fix one file end-to-end: lint, test impact, and apply minimal safe changes for this project."
---

# Fix File Skill

## Goal
Apply the smallest safe fix to a target file and validate it.

## Workflow
1. Identify the failing behavior and the minimal edit scope.
2. Apply focused changes without unrelated refactors.
3. Run targeted tests first, then broader checks if needed.
4. Report exactly what changed and why.

## Commands
```powershell
poetry run ruff check .
poetry run ruff format --check .
poetry run flake8 <target_file_path>
poetry run pytest tests/
```

## Project-Specific Guardrails
- Keep orchestration in auto_subtitle.py and logic in modules/.
- Do not weaken shutdown and process cleanup behavior.
- Preserve resume and atomic-save semantics.
- Keep complexity below 10.
