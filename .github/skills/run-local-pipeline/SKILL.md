______________________________________________________________________

## name: run-local-pipeline description: Execute local pipeline checks, linting, security scans, unit tests, code coverage gates (>=90%), and metric reports on Windows/PowerShell.

# Run Local Pipeline Skill

Use this skill to run the canonical full validation gate covering all checks end-to-end.

## Goal

Execute the project quality gate exactly as contributors do locally.

## Command

```powershell
.\run_local_pipeline.ps1
```

## Success Criteria

- Zero-suppression check passes (`check_no_suppressions.py`).
- Markdown auto-delint and scan pass (`mdformat` + `pymarkdown`).
- Ruff formatting and linting pass.
- Flake8, Pylint, Mypy, and Pyright pass.
- Bandit and Pip-audit security checks pass.
- Radon Cyclomatic Complexity (grade A, CC 1-5) and Maintainability Index (grade A) pass.
- Test suite passes with $\\ge 90%$ overall and per-file coverage. The per-file list includes every `modules/asr/*.py` module and `modules/configuration/asr_settings.py` (see `AGENTS.md`).
- The real-library NVIDIA ASR checks (`tests/e2e/test_real_nvidia_asr.py`) are e2e tests: not part of the local gate, run in the CI `e2e_nvidia_asr` job.
- Coverage badge and summary reports are generated.
