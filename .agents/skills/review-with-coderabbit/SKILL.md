______________________________________________________________________

## name: review-with-coderabbit description: Run CodeRabbit CLI reviews on local git changes or replay stored findings, categorizing into main issues and nitpicks, verifying validity before applying minimal fixes. disable-model-invocation: true

# Review with CodeRabbit Skill

Use this skill when the user explicitly requests a CodeRabbit review or asks to fix findings discovered by the CodeRabbit plugin/CLI.

## Two Modes

| Mode | Trigger | Command |
| --- | --- | --- |
| **Review** | User asks for a new CodeRabbit review | `coderabbit review --agent [scope flags]` |
| **Findings** | User asks to fix stored plugin findings | `coderabbit review findings --agent` |

## Hard Rules

1. **User-Gated**: Only run when the user explicitly invokes this skill. Do not auto-run.
1. **Verify Against Code**: Group findings into **Main Issues** (critical, major) and **Nitpicks** (minor, trivial). Inspect real code before editing.
1. **Act on Valid Findings Only**: Fix valid defects with minimal diffs. Reject invalid suggestions with clear explanations.
1. **Zero Suppressions**: Never add `# noqa`, `# type: ignore`, or `# pylint: disable` to silence a CodeRabbit finding.
1. **Final Summary Report**: End with a structured summary indicating total fixed, skipped, and blocked items.

## Progress Checklist

```text
CodeRabbit Review Progress:
- [ ] Verify CodeRabbit CLI is installed (`coderabbit --version`)
- [ ] Verify authentication (`coderabbit auth status`)
- [ ] Determine review scope (uncommitted / committed / all)
- [ ] Execute review or replay stored findings
- [ ] Categorize findings (Main Issues vs Nitpicks)
- [ ] Verify each finding against project constraints
- [ ] Apply minimal safe fixes for valid findings
- [ ] Validate fixes with `.\run_local_pipeline.ps1`
- [ ] Provide comprehensive final summary report
```

## Review Scope Options

- **Uncommitted changes**: `coderabbit review --agent --uncommitted`
- **Uncommitted + untracked files**: `coderabbit review --agent --uncommitted --include-untracked`
- **Committed changes vs main**: `coderabbit review --agent --committed`
- **All changes**: `coderabbit review --agent --include-untracked`

## Verification & Pipeline Check

After applying any CodeRabbit fixes:

```powershell
.\run_local_pipeline.ps1
```
