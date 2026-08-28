______________________________________________________________________

## applyTo: "\*\*/\*.ps1" description: Use when editing PowerShell scripts for setup, local pipeline, or utility automation in this repository.

# PowerShell Instructions

## Script Safety

- Use `Set-StrictMode -Version Latest` for strict variable and invocation semantics.
- Set `$ErrorActionPreference = "Stop"` for predictable fail-fast behavior.
- Validate process exit codes and throw explicit, informative error messages.

## Setup Consistency

- Keep `install_dependencies.ps1` idempotent so multiple runs safely repair or verify the environment.
- Preserve local `.venv` paths and launcher compatibility.
- Ensure FFmpeg installation adheres to Windows official static builds.

## Maintainability

- Decompose complex shell logic into focused script functions with clear error handling.
- Keep terminal output informative with structured progress indicators (`==> Step Name`).
