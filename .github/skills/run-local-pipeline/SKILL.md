______________________________________________________________________

## name: run-local-pipeline user-invocable: true description: Use when contributors or agents need to run the canonical full local validation gate covering all checks end-to-end.

______________________________________________________________________

# Run Local Pipeline Skill

## Goal

Execute the project quality gate exactly as contributors do locally.

## Workflow

1. Ensure dependencies are installed with Poetry.
1. Run the pipeline script.
1. Capture failures and map them to actionable file-level fixes.

## Command

```powershell
./run_local_pipeline.ps1
```

## Success Criteria

- Markdown auto-delint and lint checks pass.
- Zero-suppression policy check passes.
- Lint and type checks pass.
- Security checks pass.
- Tests pass.
- Coverage output is generated and badge update step succeeds.
