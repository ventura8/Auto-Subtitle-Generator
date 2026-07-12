______________________________________________________________________

## name: markdown-quality user-invocable: true description: "Use when linting Markdown docs, auto-fixing formatting drift, or integrating Markdown quality checks into local/CI pipelines."

# Markdown Quality Skill

## Goal

Keep Markdown documentation consistently formatted and lint-clean across README,
docs/, and .github/.

## Workflow

1. Run the auto-delinter to normalize Markdown formatting.
1. Run the Markdown linter to catch policy/style issues.
1. Apply targeted fixes and re-run checks.

## Commands

```powershell
poetry run mdformat README.md AGENTS.md docs .github
poetry run pymarkdown scan README.md AGENTS.md docs .github
```

## Success Criteria

- Markdown files are auto-formatted by mdformat.
- pymarkdown scan exits successfully.
- CI and local pipeline stages pass without Markdown issues.
