______________________________________________________________________

## applyTo: "{install_dependencies.ps1,run_local_pipeline.ps1,pyproject.toml,.github/workflows/ci.yml}" description: "Use when editing setup, installation, dependency, and local validation configuration files."

# Setup Instructions

## Dependency Management

- Keep Poetry as the source of truth for Python dependencies.
- Ensure versions and indexes remain compatible with Python 3.12+.
- Keep test-only dependencies in Poetry dev groups and avoid pip requirement
  file side-paths.

## Installer and Environment

- Maintain idempotent setup behavior in install_dependencies.ps1.
- Preserve local .venv assumptions used by launcher scripts.
- Keep FFmpeg sourcing aligned with project policy and documentation.

## Validation Pipeline

- Keep run_local_pipeline.ps1 as the canonical local gate.
- Preserve lint + test + coverage sequence for consistent feedback.
- Fail fast with clear error messages when commands fail.
