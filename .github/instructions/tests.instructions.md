______________________________________________________________________

## applyTo: "tests/\*\*/\*.py" description: "Use when creating or updating tests for unit, coverage, and pipeline behavior in this project."

# Test Instructions

## Coverage and Scope

- Cover behavioral changes with focused tests in tests/.
- Keep tests deterministic and isolated from network/hardware dependencies.
- Prefer mocking external processes and heavy model calls.

## Platform Compatibility

- Keep tests compatible with Windows and Linux where practical.
- For platform-specific attributes, use safe mocking patterns that tolerate
  missing attributes.

## Assertions

- Assert user-visible behavior and outputs, not internal implementation details
  unless required.
- Add negative-path tests for failures in FFmpeg/process/model flows where
  relevant.

## Validation Commands

- Primary: poetry run pytest tests/
- Full gate: run_local_pipeline.ps1
