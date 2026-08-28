______________________________________________________________________

## name: pr-comment-resolution description: Resolve PR review feedback from CodeRabbit and human reviewers using gh CLI / MCP, verifying validity before making edits, and replying with rationale before closing threads.

# PR Comment Resolution Skill

Use this skill when resolving GitHub pull request review comments and conversation threads from CodeRabbit or human reviewers.

## Hard Rules

1. **Verify Before Action**: For every review comment, verify whether the feedback is **Valid**, **Not Valid**, or **Blocked** against real codebase behavior, performance constraints, and project rules.
1. **Reply Before Resolve**: Always post a clear, detailed comment explaining what was changed (for valid feedback) or why it was skipped (for invalid feedback) before marking the thread resolved.
1. **No Silent Ignored Threads**: Process every unresolved thread systematically.
1. **Preserve Project Guardrails**: Never introduce `# noqa` or suppressions to silence a reviewer's complaint. Maintain complexity < 10 and >= 90% per-file coverage.

## Workflow

### 1. Identify PR and Fetch Threads

```powershell
# View PR info
gh pr view --json number,url,title,headRefName,baseRefName

# List review comments
gh pr view --comments

# Query review threads via GraphQL with pagination support
gh api graphql --paginate -F owner=":owner" -F repo=":repo" -F prNumber=<number> -f query='query($owner: String!, $repo: String!, $prNumber: Int!, $endCursor: String) { repository(owner: $owner, name: $repo) { pullRequest(number: $prNumber) { reviewThreads(first: 50, after: $endCursor) { pageInfo { hasNextPage endCursor } nodes { id isResolved comments(first: 50) { pageInfo { hasNextPage endCursor } nodes { id body author { login } } } } } } } }'
```

### 2. Classify Each Comment

| Classification | Meaning | Action |
| --- | --- | --- |
| **Valid** | Real defect, missing test, edge-case failure, or performance regression | Implement minimal safe fix + add test |
| **Not Valid** | False positive, contradictory to project design/rules, out of scope | Skip with polite, technical explanation |
| **Blocked** | Security, architecture, or breaking requirement requiring user input | Reply with the question; keep thread open |

### 3. Implement Fixes and Validate

Apply minimal code adjustments and verify with the local quality gate:

```powershell
# Targeted test & linter
poetry run ruff check <touched_file>
poetry run pytest tests/modules/test_<touched_module>.py

# Full validation gate
.\run_local_pipeline.ps1
```

### 4. Post Reply and Resolve

Post the rationale and resolve the thread using GitHub CLI or MCP Git tools.
