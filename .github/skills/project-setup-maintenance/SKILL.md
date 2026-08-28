______________________________________________________________________

## name: project-setup-maintenance description: Add, update, or repair project setup scripts, dependency configurations, and onboarding steps.

# Project Setup Maintenance Skill

## Goal

Keep onboarding and local environment setup reliable, reproducible, and aligned with repository standards.

## Workflow

1. Update setup/dependency configurations in `install_dependencies.ps1` or `pyproject.toml`.
1. Maintain local `.venv` assumptions and launcher compatibility.
1. Validate bootstrap script and run the local pipeline.

## Commands

```powershell
.\install_dependencies.ps1
.\run_local_pipeline.ps1
```
