______________________________________________________________________

## description: Prepare release artifacts, docs sync, and commit metadata from branch-derived version.

# Release Prep Workflow

## Goal

Create a complete release-ready change set using the version parsed from the
current branch name.

## Steps

### 1. Derive Version from Branch

```powershell
$branch = git branch --show-current
if ($branch -match '(?<version>v?\d+\.\d+\.\d+)$') {
  $version = $Matches.version
  if (-not $version.StartsWith('v')) { $version = "v$version" }
} else {
  throw "Branch must end with semantic version (example: feature/v1.1.2)."
}
```

### 2. Update Version and Runtime Markers

- Update `pyproject.toml` project version.
- Update visible runtime version strings if present.

### 3. Update Release Docs

- Create/update `docs/releases/$version.md`.
- Create/update `docs/releases/$version-github-release.md`.
- Sync related references in README/docs/AGENTS/skills.

### 4. Validate and Finalize Commit

```powershell
./run_local_pipeline.ps1
```

- Stage all release changes.
- Amend commit title/body with complete change summary.
