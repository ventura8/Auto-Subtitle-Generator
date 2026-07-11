# Agent Setup for This Repository

This repository includes Copilot workspace customizations and reusable skills.

## Always-On Instructions
- copilot-instructions.md
- instructions/python.instructions.md
- instructions/tests.instructions.md
- instructions/powershell.instructions.md
- instructions/architecture.instructions.md
- instructions/setup.instructions.md

## Skills
- skills/fix-file/SKILL.md: focused single-file fix workflow.
- skills/run-local-pipeline/SKILL.md: full lint/test/coverage gate workflow.
- skills/setup-dependencies/SKILL.md: setup and dependency bootstrap workflow.
- skills/architecture-review/SKILL.md: architecture-level review and implementation workflow.
- skills/project-setup-maintenance/SKILL.md: setup/dependency maintenance workflow.
- skills/docs-sync/SKILL.md: documentation synchronization workflow.

## Recommended Usage
1. Use fix-file for targeted bug fixes.
2. Use run-local-pipeline before finalizing substantial changes.
3. Use setup-dependencies for onboarding and machine repair.
4. Use architecture-review before or during structural pipeline changes.
5. Use project-setup-maintenance when changing install or dependency behavior.
6. Use docs-sync whenever setup/architecture behavior changes.

## Notes
- Keep skills project-specific and deterministic.
- Update skill descriptions with clear trigger phrases so discovery remains reliable.
