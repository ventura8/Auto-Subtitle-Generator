______________________________________________________________________

## name: release-prep description: Prepare release artifacts, release notes, pyproject.toml version sync, GitHub release descriptions, and commit metadata derived from branch name.

# Release Prep Skill

## Goal

Prepare a complete release package: version bump, release notes, GitHub release body, docs/agent synchronization, and commit metadata.

## Version Derivation

Derive release version directly from current branch name:

```powershell
$branch = git branch --show-current
if ($branch -match '(?<version>v?\d+\.\d+\.\d+)$') {
    $version = $Matches.version
    if (-not $version.StartsWith('v')) { $version = "v$version" }
    $rawVersion = $version.TrimStart('v')
} else {
    throw "Branch name must end with semantic version, e.g. feature/v1.1.2 or release/1.1.2"
}
```

## Workflow

1. Resolve version from branch name.
1. Update `pyproject.toml` `[project] version = "$rawVersion"`.
1. Create or update `docs/releases/$version.md`.
1. Create or update `docs/releases/${version}_github_description.md`.
   Start it with the summary paragraph, never an `#` H1 title: the GitHub
   release title is pinned to the bare tag (`vX.Y.Z`), so a leading H1
   renders as a duplicate oversized title.
1. Sync `README.md`, `docs/`, `AGENTS.md`, and skill references.
1. Run full local validation (`.\run_local_pipeline.ps1`).
1. Prepare commit title/body with a comprehensive release summary
   (`release: vX.Y.Z – <summary>` + detailed multiline body).
1. Stage **all** release files, then decide **before any commit command**:
   - if the release commit is `HEAD` **and** unpushed → `git commit --amend`
     with that title/body (one release commit on the branch; `--amend` only
     rewrites `HEAD`, so never amend a pushed commit or one with commits on
     top);
   - otherwise → `git commit` with the same title/body (normal commit).
1. Only after that commit exists, tag `v$rawVersion` and push the tag to
   trigger `release.yml`.
