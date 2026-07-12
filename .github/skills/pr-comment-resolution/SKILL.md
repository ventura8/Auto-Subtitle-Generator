______________________________________________________________________

## name: pr-comment-resolution user-invocable: true description: "Use when resolving pull request review comments (CodeRabbit + human) with GitHub CLI and MCP tooling while enforcing detailed replies before any thread is closed."

# PR Comment Resolution Skill

## Goal

Resolve all PR feedback with a traceable workflow that combines GitHub CLI and
MCP review/comment tools.

## Workflow

1. Enumerate open PR review threads and comments from both CodeRabbit and human
   reviewers.
1. Reproduce and fix each requested change with minimal, safe edits.
1. Post a detailed response per comment describing:
   - what changed,
   - why the change is correct,
   - which files/tests validate the fix.
1. Only mark a comment/thread resolved after the detailed response is posted.
1. Repeat until no unresolved review comments remain.

## Commands and Tools

```powershell
# Inspect review comments via GitHub CLI
gh pr view <pr-number> --comments

# Optional JSON inspection for automation
gh pr view <pr-number> --json comments,reviews,reviewThreads
```

- Use MCP Git tools for structured PR review flows when available.
- Prefer MCP comment/review actions when they improve traceability.
- Use GitHub CLI fallback commands when MCP capability is unavailable.

## Guardrails

- Never close a PR comment/thread without first posting a detailed reply.
- Treat CodeRabbit comments with the same rigor as human reviewer comments.
- Keep responses concrete and implementation-specific (avoid generic acknowledgments).
- Preserve existing project constraints (Windows safety, model loading, resume/atomic writes).
