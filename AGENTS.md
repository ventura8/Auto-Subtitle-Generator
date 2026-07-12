# Agent Setup for This Repository

This repository includes Copilot workspace customizations and reusable skills.

## Always-On Instructions
- .github/copilot-instructions.md
- .github/instructions/python.instructions.md
- .github/instructions/tests.instructions.md
- .github/instructions/powershell.instructions.md
- .github/instructions/architecture.instructions.md
- .github/instructions/setup.instructions.md

## Skills
- .github/skills/fix-file/SKILL.md: focused single-file fix workflow.
- .github/skills/run-local-pipeline/SKILL.md: full lint/test/coverage gate workflow.
- .github/skills/setup-dependencies/SKILL.md: setup and dependency bootstrap workflow.
- .github/skills/architecture-review/SKILL.md: architecture-level review and implementation workflow.
- .github/skills/project-setup-maintenance/SKILL.md: setup/dependency maintenance workflow.
- .github/skills/docs-sync/SKILL.md: documentation synchronization workflow.

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