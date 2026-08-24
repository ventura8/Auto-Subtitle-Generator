______________________________________________________________________

## name: docs-sync description: Synchronize documentation across README, docs/, AGENTS.md, instructions, workflows, and agent skills whenever architecture or behavior changes.

# Docs Sync Skill

Use this skill to prevent documentation drift across the repository whenever code, configuration, pipelines, or skills are modified.

## Documentation Matrix

When making changes, ensure corresponding documents are kept in sync:

| Change Scope | Documents to Update |
| --- | --- |
| **Pipeline / Quality Gates** | `run_local_pipeline.ps1`, `AGENTS.md`, `.agents/skills/pipeline-runner/`, `docs/development_standards.md` |
| **Model / Hardware Tuning** | `modules/models.py`, `docs/hardware_optimization.md`, `docs/pipeline_logic.md`, `README.md` |
| **Configuration Options** | `config.yaml`, `docs/configuration.md`, `modules/configuration/config.py` |
| **Dependencies / Setup** | `install_dependencies.ps1`, `pyproject.toml`, `docs/instructions.md`, `README.md` |
| **Releases / Versions** | `docs/releases/`, `pyproject.toml`, `AGENTS.md`, `README.md` |
| **Agent / IDE Customizations** | `AGENTS.md`, `.agent/workflows/`, `.agents/skills/`, `.github/instructions/` |

## Workflow

1. Identify all touched modules, entrypoints, and settings.
1. Review related documentation files for stale command examples, outdated parameter tables, or incorrect version references.
1. Update docs with concise, factual, and verified information.
1. Auto-delint and scan modified Markdown files:

```powershell
poetry run mdformat README.md AGENTS.md docs .github .agent .agents
poetry run pymarkdown scan README.md AGENTS.md docs .github .agent .agents
```
