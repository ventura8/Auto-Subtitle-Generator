______________________________________________________________________

## name: docs-sync description: Synchronize documentation across README, docs/, AGENTS.md, instructions, workflows, and agent skills whenever architecture or behavior changes.

# Docs Sync Skill

## Goal

Keep project documentation accurate and prevent drift across architecture, settings, and skill definitions.

## Workflow

1. Identify documentation affected by changes.
1. Update relevant markdown files with verified, factual details.
1. Auto-delint and scan Markdown files (`mdformat` and `pymarkdown`).

## Typical Targets

- `README.md`
- `AGENTS.md`
- `docs/` (`configuration.md`, `development_standards.md`, `hardware_optimization.md`, `instructions.md`, `pipeline_logic.md`, `project_overview.md`, `releases/`)
- `.agents/skills/`
- `.github/skills/`
- `.agent/workflows/`
