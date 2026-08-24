______________________________________________________________________

## applyTo: "{auto_subtitle.py,modules/\*\*/\*.py}" description: Use when changing pipeline architecture, module boundaries, orchestration flow, or model lifecycle behavior.

# Architecture Instructions

## Separation of Responsibilities

- Keep `auto_subtitle.py` strictly as an orchestrator (CLI arguments, batch loops, high-level staging).
- Put reusable business logic under `modules/`.
- Avoid moving heavy operational logic into CLI entrypoints.

## Process and Memory Model

- Keep isolated heavy translation execution patterns intact (`isolated_translator.py`).
- Preserve model offload/cleanup behavior before loading subsequent heavy components.
- Never use `device_map="auto"`. Use explicit device mapping.
- Avoid introducing hidden mutable shared state across modules.

## Reliability Contracts

- Preserve atomic write behavior for generated subtitles (`.tmp` write then atomic rename).
- Preserve resume/skip logic for already processed outputs.
- Keep subprocess shutdown paths robust on Windows.

## Performance Contracts

- Respect optimizer tier decisions (`ULTRA` / `HIGH` / `MID` / `LOW` / `CPU`).
- Avoid hardcoding profile overrides that bypass config and detection.
- Keep FFmpeg invocation patterns compatible with existing utility wrappers.
