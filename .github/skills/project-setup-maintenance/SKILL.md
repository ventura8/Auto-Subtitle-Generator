______________________________________________________________________

## name: project-setup-maintenance user-invocable: true description: "Use when adding, updating, or repairing project setup scripts, dependency configuration, and onboarding steps."

# Project Setup Maintenance Skill

## Goal

Keep onboarding and local environment setup reliable, reproducible, and aligned
with repository standards.

## Workflow

1. Update setup/dependency files with minimal targeted changes.
1. Keep install paths, virtual environment assumptions, and launcher behavior
   consistent.
1. Run local validation for setup and smoke-check entrypoints.

## Commands

```powershell
./install_dependencies.ps1
./run_local_pipeline.ps1
```

## Guardrails

- Avoid breaking existing local installs during script updates.
- Keep setup steps explicit and deterministic for Windows users.
- Synchronize setup changes with README/instructions documentation.
