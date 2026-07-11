---
name: architecture-review
user-invocable: true
description: "Use when reviewing or implementing architecture-level changes across orchestration, modules, process isolation, or model lifecycle flows."
---

# Architecture Review Skill

## Goal
Validate and execute architecture-level changes without violating reliability or performance constraints.

## Workflow
1. Identify impacted boundaries: orchestration, modules, process isolation, utility layers.
2. Verify state ownership and model lifecycle (load, offload, cleanup).
3. Implement minimal structural change with explicit rationale.
4. Update tests/documentation for behavior or architecture contract changes.

## Checks
```powershell
.\run_local_pipeline.ps1
```

## Guardrails
- Keep auto_subtitle.py orchestration-centric.
- Preserve shutdown safety and atomic persistence behavior.
- Keep complexity under project limits by decomposition.
