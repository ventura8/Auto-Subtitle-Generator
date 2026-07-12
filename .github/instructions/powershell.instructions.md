______________________________________________________________________

## applyTo: "\*\*/\*.ps1" description: "Use when editing PowerShell scripts for setup, local pipeline, or utility automation in this repository."

# PowerShell Instructions

## Script Safety

- Use Set-StrictMode -Version Latest for new scripts where possible.
- Keep $ErrorActionPreference = "Stop" for fail-fast behavior.
- Validate command exits and throw clear errors when commands fail.

## Setup Consistency

- Keep install_dependencies.ps1 idempotent.
- Preserve local .venv assumptions and launcher compatibility.
- Keep FFmpeg setup aligned with the official FFmpeg download path policy for
  Windows.

## Maintainability

- Use small helper functions for repeated command execution patterns.
- Keep user output clear by labeling major steps.
- Avoid interactive prompts unless explicitly needed.
