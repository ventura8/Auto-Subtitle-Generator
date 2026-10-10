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
- Keep every function at Radon grade A (CC 1-5) and every file at MI grade A, without any suppressions.
- Import `torch`, `transformers`, `faster_whisper`, `numpy`, `soundfile` and `huggingface_hub` only through `importlib` or `load_optional_torch()`; the quality gate runs without the `ml` group.
- Log every warning through `modules.utils.log` (the `models.py` `LOGGER` never reaches `subtitle_gen.log`).
- Never pass a string as audio to a transformers processor; it would be opened as a path or URL.
- **Zero suppressions**: Never use `# noqa`, `# type: ignore`, or `# pylint: disable`.
- Add or update tests in `tests/` for all behavior changes, maintaining >= 90% coverage.
