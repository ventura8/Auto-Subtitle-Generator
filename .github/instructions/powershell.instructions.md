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
- Use an FFmpeg already on `PATH` (both `ffmpeg` and `ffprobe`) before downloading
  one, in the same order as `modules/media/ffmpeg_utils.get_ffmpeg_paths()`.
- The fallback download is pinned to a versioned gyan.dev release build
  (`GyanD/codexffmpeg` release tag) and verified by SHA256. Never pin a rolling
  autobuild: upstream prunes those after about two weeks, which breaks fresh
  installs. A failed download or hash check exits non-zero.

## Maintainability

- Decompose complex shell logic into focused script functions with clear error handling.
- `.github/scripts/Invoke-PowerShellLint.ps1` fails any function over cyclomatic
  complexity 9 or nesting depth 4, on top of PSScriptAnalyzer warnings.
  SonarQube Cloud also caps cognitive complexity at 15 per function.
- Keep terminal output informative with structured progress indicators (`==> Step Name`).
