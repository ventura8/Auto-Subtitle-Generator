______________________________________________________________________

## applyTo: "\*\*/\*.py" description: Use when editing Python files in this repository, including modules, orchestration, and tests.

# Python File Instructions

## Architecture

- Place reusable logic in `modules/`.
- Keep `auto_subtitle.py` focused on orchestration flow.
- Avoid introducing hidden global mutable state.

## Reliability

- Keep output writes atomic for generated subtitle assets.
- Preserve resume/skip behavior for already processed outputs.
- Maintain robust error handling around subprocess/model operations.

## Performance

- Keep GPU mapping explicit and deterministic (no `device_map="auto"`).
- Avoid changes that can cause model spillover to shared system memory.
- Keep batch/thread tuning compatible with profile-based optimizer logic.

## Quality Bar

- Prefer small pure helper functions over large branching blocks.
- Keep complexity strictly below 10 (Radon A-rank) without any suppressions.
- **Zero suppressions**: Never use `# noqa`, `# type: ignore`, or `# pylint: disable`.
- Add or update tests in `tests/` for all behavior changes, maintaining >= 90% coverage.
