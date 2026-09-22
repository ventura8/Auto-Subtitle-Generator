---
description: Scan the repository with SonarQube Cloud, enforce the quality gate, and fix findings without suppressions.
---

# SonarQube Scan Workflow

Use this workflow to run a SonarQube Cloud analysis and resolve what it reports.
See `.agents/skills/sonarqube-analysis/SKILL.md` for rule-by-rule guidance.

## Steps

### 1. Verify Token & Configuration

```bash
test -n "$SONAR_TOKEN" && echo "token present" || echo "export SONAR_TOKEN first"
cat sonar-project.properties
```

### 2. Generate Fresh Coverage

Sonar never runs tests itself; it reads `coverage.xml`. A stale file reports 0 %.

```bash
poetry run pytest -m "not e2e" --cov=auto_subtitle --cov=modules --cov-branch --cov-report=xml tests/
```

### 3. Run the Scanner

```bash
npx --yes sonarqube-scanner -Dsonar.host.url=https://sonarcloud.io
```

### 4. Review the Quality Gate

Open `https://sonarcloud.io/summary/new_code?id=ventura8_Auto-Subtitle-Generator`
and triage in this order: Bugs → Vulnerabilities → Security Hotspots → Coverage on
new code → Cognitive Complexity → Duplications.

### 5. Fix and Re-verify

- Apply minimal, behaviour-preserving fixes. Never add `# NOSONAR` or resolve a
  finding as "Won't fix" to clear the gate.
- Re-run the full local gate to confirm no regression elsewhere:

```bash
./run_local_pipeline.sh
```

### 6. Confirm in CI

The `🛰️ SonarQube Cloud Analysis` job in `.github/workflows/ci.yml` re-runs the scan
and blocks the build on the gate. It is skipped for fork pull requests, where the
`SONAR_TOKEN` secret is unavailable.
