______________________________________________________________________

## name: release-prep user-invocable: true description: Use when preparing a release commit and GitHub release notes; derives version from current branch name like feature/v1.1.2 or feature/1.1.2.

______________________________________________________________________

# Release Prep Skill

## Goal

Prepare a complete release package: version bump, release notes, GitHub release
body, docs/agent synchronization, and commit metadata.

## Version Source

Derive release version from current branch name.

```powershell
$branch = git branch --show-current
if ($branch -match '(?<version>v?\d+\.\d+\.\d+)$') {
  $version = $Matches.version
  if (-not $version.StartsWith('v')) { $version = "v$version" }
} else {
  throw "Branch name must end with semantic version, e.g. feature/v1.1.2"
}
```

## Workflow

1. Resolve version from branch name.
1. Update runtime/project version markers.
1. Create or update `docs/releases/$version.md`.
1. Create or update `docs/releases/$version-github-release.md`.
1. Sync README/docs/AGENTS/skill references.
1. Run local validation (`./run_local_pipeline.ps1`).
1. Amend commit title/body with a complete release change summary.

## Typical Files

- `pyproject.toml`
- `modules/runtime/logging_utils.py`
- `docs/releases/$version.md`
- `docs/releases/$version-github-release.md`
- `README.md`
- `AGENTS.md`
- `.agent/workflows/release-prep.md`

## Guardrails

- Do not use suppression directives to pass checks.
- Preserve architecture boundaries (`auto_subtitle.py` orchestration only,
  reusable logic in `modules/`).
- Keep release notes factual and aligned with actual diff.
