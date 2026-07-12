______________________________________________________________________

## name: setup-dependencies user-invocable: true description: "Use when bootstrapping or repairing local setup: Python, Poetry, FFmpeg, and project dependency installation."

# Setup Dependencies Skill

## Goal

Bring a local machine to a runnable state for this repository.

## Workflow

1. Verify Python 3.12+ availability.
1. Run install_dependencies.ps1 for venv, FFmpeg, and Poetry setup.
1. Validate the launcher and command-line execution path.

## Commands

```powershell
./install_dependencies.ps1
poetry run python auto_subtitle.py --help
```

## Project Policy

- FFmpeg setup must follow the official FFmpeg download path for Windows users.
- Keep setup idempotent so reruns do not break existing installations.
