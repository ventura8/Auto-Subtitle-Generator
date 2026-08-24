______________________________________________________________________

## description: Run CodeRabbit CLI reviews on local git changes or replay stored findings, verifying validity before applying minimal fixes.

# Review with CodeRabbit Workflow

Use this workflow to perform local CodeRabbit code reviews or resolve stored findings.

## Steps

### 1. Ensure CodeRabbit CLI & Auth

```powershell
coderabbit --version
coderabbit auth status
```

### 2. Run Review or Load Findings

```powershell
# Review uncommitted changes
coderabbit review --agent --uncommitted --include-untracked

# Or replay stored findings
coderabbit review findings --agent
```

### 3. Verify & Fix Valid Findings

- Group findings into **Main Issues** and **Nitpicks**.
- Verify each item against project codebase and invariants.
- Apply minimal safe fixes for valid defects.
- Run `.\run_local_pipeline.ps1` to ensure no quality regressions or suppressions.
