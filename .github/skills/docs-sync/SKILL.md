---
name: docs-sync
user-invocable: true
description: "Use when code, setup, or architecture changes require synchronized updates to README and docs/*.md files."
---

# Docs Sync Skill

## Goal
Keep project documentation accurate after code or setup changes.

## Workflow
1. Identify documentation affected by the change.
2. Update the minimum set of docs to remove drift.
3. Verify command examples and paths reflect real repository files.

## Typical Targets
- README.md
- docs/project_overview.md
- docs/pipeline_logic.md
- docs/configuration.md
- docs/development_standards.md

## Guardrails
- Prefer precise, actionable updates over broad rewrites.
- Keep naming consistent with actual script/module names.
- Avoid introducing instructions that conflict with existing automation.
