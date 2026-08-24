______________________________________________________________________

## name: release-prep description: > Prepare release artifacts, release notes, pyproject.toml version sync, GitHub release descriptions, and commit metadata derived from branch name.

# Release Prep Skill

Use this skill when cutting a release, updating version metadata, generating
release documentation, and preparing commit metadata.

## Version Derivation

The release version is derived directly from the current git branch:

```powershell
$branch = git branch --show-current
# Matches: feature/v1.1.3, release/1.1.3, v1.1.3, 1.1.3
if ($branch -match '(?<version>v?\d+\.\d+\.\d+)$') {
    $version = $Matches.version
    if (-not $version.StartsWith('v')) { $version = "v$version" }
    $rawVersion = $version.TrimStart('v')
} else {
    throw "Branch name must end with semantic version, e.g. release/v1.1.3 or feature/1.1.3"
}
```

## Release Workflow

### 1. Update Project Version

Update `pyproject.toml` `[project] version = "$rawVersion"` to match the branch
version.

### 2. Review All Changes Since Base

```powershell
$base = git merge-base HEAD main 2>$null
if (-not $base) { $base = git merge-base HEAD master }
git log --oneline "$base..HEAD"
git diff --stat "$base..HEAD"
```

### 3. Generate Release Documentation

Create two release docs under `docs/releases/`:

1. **Full Release Notes**: `docs/releases/v{version}.md`

   - Comprehensive internal notes: all changes, quality gates, architectural
     notes.

1. **GitHub Release Description**: `docs/releases/v{version}_github_description.md`

   - Concise public-facing body used verbatim as the GitHub Release body.
   - Naming convention: `v{N.N.N}_github_description.md` (underscore, no
     hyphen).
   - If this file exists when the `v{version}` tag is pushed, the
     `.github/workflows/release.yml` workflow picks it up automatically and
     uses it as the release body. If absent, the workflow falls back to
     GitHub's auto-generated release notes.

### 4. Sync Documentation & Agents

- Update `README.md` version references and badges if needed.
- Update `AGENTS.md` and architecture notes.
- Verify skill documentation consistency.

### 5. Validate Full Pipeline

```powershell
.\run_local_pipeline.ps1
```

### 6. Commit and Tag

Prepare clear, descriptive commit messages summarising the release features,
performance improvements, bug fixes, and test coverage enhancements.

Push the version tag to trigger the automated release:

```powershell
git tag v$rawVersion
git push origin v$rawVersion
```

The `.github/workflows/release.yml` workflow will then:

1. Validate the tag matches `pyproject.toml` version.
1. Detect `docs/releases/v{version}_github_description.md`.
1. Create the GitHub Release with the authored description as the body.
