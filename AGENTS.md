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
- .github/skills/run-local-pipeline/SKILL.md: full lint/test/coverage gate
  workflow.
- .github/skills/setup-dependencies/SKILL.md: setup and dependency bootstrap
  workflow.
- .github/skills/architecture-review/SKILL.md: architecture-level review and
  implementation workflow.
- .github/skills/project-setup-maintenance/SKILL.md: setup/dependency
  maintenance workflow.
- .github/skills/docs-sync/SKILL.md: documentation synchronization workflow.
- .github/skills/markdown-quality/SKILL.md: markdown auto-delint/lint workflow
  for docs quality.
- .github/skills/pr-comment-resolution/SKILL.md: PR comment resolution workflow
  for CodeRabbit and human review feedback using GitHub CLI + MCP.

## Recommended Usage

1. Use fix-file for targeted bug fixes.
1. Use run-local-pipeline before finalizing substantial changes.
1. Use setup-dependencies for onboarding and machine repair.
1. Use architecture-review before or during structural pipeline changes.
1. Use project-setup-maintenance when changing install or dependency behavior.
1. Use docs-sync whenever setup/architecture behavior changes.
1. Use markdown-quality when updating README/docs/.github markdown content.
1. Use pr-comment-resolution when addressing PR review feedback and resolving
   CodeRabbit/human comments.

## Notes

- Keep skills project-specific and deterministic.
- Update skill descriptions with clear trigger phrases so discovery remains
  reliable.
