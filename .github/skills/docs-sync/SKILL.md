______________________________________________________________________

## name: docs-sync user-invocable: true description: Use when documentation synchronization is needed after code/setup/architecture changes, including release notes, AGENTS.md, workflow documentation, and skill documentation updates.

______________________________________________________________________

# Docs Sync Skill

## Goal

Keep project documentation accurate after code or setup changes.

## Workflow

1. Identify documentation affected by the change.
1. Update the minimum set of docs to remove drift.
1. Verify command examples and paths reflect real repository files.

## Typical Targets

- README.md
- docs/project_overview.md
- docs/pipeline_logic.md
- docs/configuration.md
- docs/development_standards.md
- docs/releases/\*.md
- AGENTS.md
- .agent/workflows/\*.md
- .github/skills/\*/SKILL.md

## Guardrails

- Prefer precise, actionable updates over broad rewrites.
- Keep naming consistent with actual script/module names.
- Avoid introducing instructions that conflict with existing automation.
