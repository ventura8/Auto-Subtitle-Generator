---
name: sonarqube-analysis
description: Run SonarQube Cloud analysis locally or in CI, interpret the quality gate, and fix reported issues without suppression comments.
---

# SonarQube Analysis Skill

Use this skill to scan the repository with SonarQube Cloud (sonarcloud.io), read the
quality gate result, and resolve findings. Sonar complements the existing linters; it
does not replace them. `run_local_pipeline.sh` remains the authoritative local gate.

## Hard Rules

1. **Zero Suppressions Allowed**: never add `# NOSONAR`, `sonar.issue.ignore.*` rules,
   or "Won't fix" / "False positive" resolutions to silence a finding. Fix the code.
1. Never commit `SONAR_TOKEN`. It lives in GitHub Actions secrets and, locally, only in
   the shell environment.
1. Do not widen `sonar.exclusions` to hide uncovered code. Exclusions are for files with
   no runtime behaviour (type stubs, generated assets) only.
1. Coverage must come from a real test run. Never hand-edit `coverage.xml`.

## Analysis Mode

This project uses **CI-based analysis**, and Automatic Analysis is switched **off**
on the SonarCloud project (Administration -> Analysis method). That is deliberate and
must stay that way: Automatic Analysis scans server-side and cannot ingest a coverage
report, so coverage would read 0% and the gate's "coverage on new code" condition
would fail. The two modes also conflict — with both enabled the scanner aborts with
"You are running CI analysis while Automatic Analysis is enabled".

## Verify Before You Fix

Sonar findings are suggestions, not facts. Confirm each one against the code before
changing anything; several rules misfire on this codebase's idioms:

- **S7504 (unnecessary `list()`)** — usually a false positive here. `for x in list(c)`
  is deliberate wherever the loop mutates `c`; removing the copy raises
  "changed size during iteration" at runtime.
- **S8997 (use `monkeypatch`)** — inapplicable. These suites are `unittest.TestCase`
  based, where pytest fixtures are not injected; `self.addCleanup(setattr, ...)` is
  already the correct idiom.
- **S125 (commented-out code)** — often flags explanatory comments documenting
  platform constants, not dead code.
- **S5863 (identical actual/expected)** — genuine when it flags a tautology, but
  calling the same accessor twice to assert caching is valid. Name the two calls so
  the intent is explicit rather than deleting the assertion.

Taint rules (the `pythonsecurity:` prefix) track data flow from a source to a
sink and do not recognise validation helpers as sanitizers. `S8707` kept firing
on the translation worker through two rounds of correct path validation; it only
cleared once the manifest stopped arriving as a path on the command line and
came in over stdin instead, removing the sink. If a taint finding survives
validation you believe is sound, the answer is to remove the source or the sink,
not to add more checks.

Never silence a finding to clear the gate. If a rule is genuinely wrong for this
repository, leave the code correct and explain why in the review, rather than adding
`# NOSONAR` or an exclusion.

## Configuration

All analysis settings live in `sonar-project.properties` at the repository root:

- `sonar.sources` — product code (`auto_subtitle.py`, `modules`, `launcher`)
- `sonar.tests` — `tests`, declared separately so test files are not counted as
  uncovered product code
- `sonar.python.coverage.reportPaths` — `coverage.xml`
- `sonar.python.version` — `3.12`, matching `requires-python`

`sonar.projectKey` and `sonar.organization` must match the SonarCloud project. Both
default to the GitHub slug on import; confirm under **Administration → Update Key**.

## Workflow

### 1. Produce a Fresh Coverage Report

Sonar reads coverage from a file; it never runs the tests itself. A stale or missing
`coverage.xml` silently reports 0% coverage on new code and fails the gate.

```bash
poetry run pytest -m "not e2e" --cov=auto_subtitle --cov=modules --cov-branch --cov-report=xml tests/
```

### 2. Run the Scan Locally

Requires a user token from **My Account → Security** on sonarcloud.io.

```bash
export SONAR_TOKEN="<your-token>"
npx --yes sonarqube-scanner -Dsonar.host.url=https://sonarcloud.io
```

The scanner picks up `sonar-project.properties` automatically from the repository root.

### 3. Read the Quality Gate

CI enforces the gate in the `🛰️ SonarQube Cloud Analysis` job, which runs after
`static_analysis` and `technical_validation` and reuses their coverage artifact. The job
is skipped for pull requests from forks, because secrets are not exposed there.

Check the result at:
`https://sonarcloud.io/summary/new_code?id=ventura8_Auto-Subtitle-Generator`

### 4. Fix Findings by Category

- **Bug / Vulnerability**: fix immediately; these block the gate.
- **Security Hotspot**: review each one and either fix it or justify it in the code's
  own structure (for example, validate a path before use rather than marking the
  hotspot safe in the Sonar UI).
- **Code Smell — Cognitive Complexity**: Sonar's cognitive complexity is not Radon's
  cyclomatic complexity; a function can pass Radon A and still fail Sonar. Extract
  nested branches into named helpers.
- **Duplicated Blocks**: factor the shared logic into `modules/utils.py` or the relevant
  subpackage rather than leaving parallel copies.
- **Coverage on New Code**: add real tests. The project already enforces ≥ 90 % per file
  in CI; Sonar additionally gates coverage on changed lines.

## Troubleshooting

- **"Project not found"**: `sonar.projectKey` or `sonar.organization` does not match the
  SonarCloud project. Verify both on the project's Administration page.
- **Coverage reports 0 %**: `coverage.xml` was missing or generated from a different
  path layout. Regenerate it from the repository root so filenames stay root-relative.
- **Gate never completes in CI**: the quality gate step polls the server; confirm the
  scan step above it actually uploaded a report.
