______________________________________________________________________

## name: architecture-review description: Review and implement structural architecture changes across orchestration, modules, model lifecycle, and process isolation.

# Architecture Review Skill

Use this skill when designing or implementing structural changes across `auto_subtitle.py`, `modules/`, process isolation boundaries, or the model lifecycle.

## Architectural Principles

1. **Orchestration vs Business Logic**:

   - `auto_subtitle.py` is strictly an orchestrator: CLI parsing, batch loops, high-level staging, summary logging, and exit handling.
   - All domain logic, media extraction, transcription, translation, and IO formatting live under `modules/`.

1. **Model Lifecycle & Memory Isolation**:

   - Transcriber (`FasterWhisper`) and Translator (`NLLBTranslator` default, `TranslateGemmaTranslator` optional) are never memory-co-located during heavy execution.
   - Faster Whisper model instances are explicitly unloaded, and CUDA cache cleared, prior to translation.
   - Translation runs in an isolated child process (`isolated_translator.py`) to guarantee total OS-level VRAM reclamation on completion.

1. **Stateless Functions & Deterministic State**:

   - Pure functions with single responsibilities are preferred over global mutable state.
   - State across pipeline stages is passed explicitly using structured dictionaries or dataclasses.

1. **Reliability & Atomic Output**:

   - Temporary file writes + atomic rename prevent corrupted partial subtitle files.
   - Signal handling preserves completed files and cleanly shuts down background workers.

## Architectural Review Checklist

```text
Architecture Review:
- [ ] Are orchestration concerns kept out of `modules/`?
- [ ] Are core domain algorithms kept out of `auto_subtitle.py`?
- [ ] Is model loading explicit (no `device_map="auto"`)?
- [ ] Are process boundaries preserved for translation?
- [ ] Is Cyclomatic Complexity < 10 for all newly introduced functions?
- [ ] Are zero suppressions maintained across all files?
- [ ] Is test coverage >= 90% for touched modules?
```

## Validation

```powershell
.\run_local_pipeline.ps1
```
