______________________________________________________________________

## name: markdown-quality description: Run mdformat auto-delinter and pymarkdown quality scan across all markdown documents.

# Markdown Quality Skill

## Goal

Keep all Markdown documentation consistently formatted and lint-clean.

## Commands

```powershell
# Auto-delint format
poetry run mdformat README.md AGENTS.md docs .github .agent .agents

# Scan quality
poetry run pymarkdown scan README.md AGENTS.md docs .github .agent .agents
```

## Success Criteria

- Clean formatting without manual alignment hacks.
- Zero errors reported by `pymarkdown scan`.
