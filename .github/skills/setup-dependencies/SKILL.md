______________________________________________________________________

## name: setup-dependencies description: Bootstrap, install, or repair local development setup: Python 3.12+, Poetry virtualenv, PyTorch CUDA 13.2 / cuDNN, and local FFmpeg on Windows.

# Setup Dependencies Skill

## Goal

Bring the local machine to a functional, GPU-accelerated state for Auto-Subtitle-Generator.

## Workflow

1. Verify Python 3.12 availability.
1. Run `.\install_dependencies.ps1` for virtual environment, PyTorch CUDA, cuDNN, and FFmpeg setup.
1. Validate launcher and command-line execution.

## Commands

```powershell
.\install_dependencies.ps1
poetry run python auto_subtitle.py --help
```
