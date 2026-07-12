______________________________________________________________________

## applyTo: "\*\*/\*.py" description: "Use when editing Python files in this repository, including modules, orchestration, and tests."

# Python File Instructions

## Architecture

- Place reusable logic in modules/.
- Keep auto_subtitle.py focused on orchestration flow.
- Avoid introducing hidden global state.

## Reliability

- Keep output writes atomic for generated subtitle assets.
- Preserve resume/skip behavior for already processed outputs.
- Maintain robust error handling around subprocess/model operations.

## Performance

- Keep GPU mapping explicit and deterministic.
- Avoid changes that can cause model spillover to shared system memory.
- Keep batch/thread tuning compatible with profile-based optimizer logic.

## Quality Bar

- Prefer small pure helper functions over large branching blocks.
- Keep complexity below 10 without noqa suppressions.
- Add or update tests for behavior changes.
