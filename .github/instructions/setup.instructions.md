______________________________________________________________________

## applyTo: "{install_dependencies.ps1,run_local_pipeline.ps1,pyproject.toml,.github/workflows/ci.yml}" description: Use when editing setup, installation, dependency, and local validation configuration files.

# Setup Instructions

## Dependency Management

- Keep `pyproject.toml` as the sole source of truth for dependencies.
- Ensure all dependencies are compatible with Python 3.12 (specifically `< 3.13`).
- Maintain PyTorch CUDA 13.2 explicit wheel repository index configuration.
- Avoid loose `pip install` side paths; keep all dev/test dependencies in Poetry groups.

## Installer and Environment

- Maintain idempotent setup behavior in `install_dependencies.ps1`.
- Preserve local `.venv` paths and standard launcher configurations (`start.bat`).
- Sourcing of FFmpeg must support official local paths and system PATH.

## Validation Pipeline

- Keep `run_local_pipeline.ps1` as the canonical local gate.
- Preserve zero-suppression enforcement, linting, type checks, security audits, and per-file >= 90% coverage gates.
