______________________________________________________________________

## name: architecture-review description: Review and implement structural architecture changes across orchestration, modules, model lifecycle, and process isolation.

# Architecture Review Skill

## Goal

Validate and execute architecture-level changes without violating reliability or performance constraints.

## Hard Rules & Architectural Invariants

1. **Orchestration vs Business Logic**: Keep `auto_subtitle.py` strictly as an orchestrator; domain logic lives under `modules/`.
1. **Model Lifecycle & Isolation**: Faster Whisper and isolated translators (`NLLBTranslator` / `TranslateGemmaTranslator`) are never co-located during heavy execution.
1. **Subprocess Process Isolation**: Translation runs in `isolated_translator.py` with clean signal handling and VRAM reclamation.
1. **Reliability & Atomic Output**: Output subtitle files write to temporary `.tmp` files and atomically replace destination files.
1. **Zero Suppressions**: Never introduce `# noqa`, `# type: ignore`, `# pylint: disable`, `# bandit: disable`, or warning-ignore filters (see `AGENTS.md`).
1. **Quality Gates**: Maintain Cyclomatic Complexity < 10 and per-file test coverage >= 90%.

## Workflow

1. Identify impacted boundaries: orchestration (`auto_subtitle.py`), domain modules (`modules/`), process isolation (`isolated_translator.py`), or utilities.
1. Verify state ownership and model lifecycle (explicit device mapping, cleanup before heavy stages).
1. Implement minimal structural changes with explicit rationale.
1. Update tests and documentation for contract changes.

## Validation

```powershell
.\run_local_pipeline.ps1
```
