______________________________________________________________________

## name: pr-comment-resolution description: Resolve PR review feedback from CodeRabbit and human reviewers using gh CLI / MCP, verifying validity before making edits, and replying with rationale before closing threads.

# PR Comment Resolution Skill

## Goal

Resolve PR feedback with a rigorous, traceable workflow combining GitHub CLI and MCP review tools.

## Hard Rules

1. **Verify validity first**: Classify each comment as Valid, Not Valid, or Blocked before editing code.
1. **Reply before resolving**: Always post a clear reply detailing what changed or why the suggestion was skipped.
1. **No silent skips**: Address all open conversation threads.
1. **Preserve repository invariants**: No suppressions, complexity < 10, >= 90% per-file test coverage.

## Commands

```powershell
# View comments and reviews
gh pr view --comments
gh pr view --json comments,reviews

# Query review threads via GraphQL with pagination support
gh api graphql --paginate -F owner=":owner" -F repo=":repo" -F prNumber=<number> -f query='query($owner: String!, $repo: String!, $prNumber: Int!, $endCursor: String) { repository(owner: $owner, name: $repo) { pullRequest(number: $prNumber) { reviewThreads(first: 50, after: $endCursor) { pageInfo { hasNextPage endCursor } nodes { id isResolved comments(first: 50) { pageInfo { hasNextPage endCursor } nodes { id body author { login } } } } } } } }'
```
