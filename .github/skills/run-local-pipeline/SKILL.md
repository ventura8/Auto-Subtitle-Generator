---
name: run-local-pipeline
user-invocable: true
description: "Use when validating repository health with the full local quality gate including lint, tests, and coverage artifact generation."
---

# Run Local Pipeline Skill

## Goal
Execute the project quality gate exactly as contributors do locally.

## Workflow
1. Ensure dependencies are installed with Poetry.
2. Run the pipeline script.
3. Capture failures and map them to actionable file-level fixes.

## Command
```powershell
./run_local_pipeline.ps1
```

## Success Criteria
- Lint checks pass.
- Tests pass.
- Coverage output is generated and badge update step succeeds.
