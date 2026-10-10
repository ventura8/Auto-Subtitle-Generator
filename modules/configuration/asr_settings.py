"""ASR engine selection from the ``asr:`` section of config.yaml.

This lives outside ``config.py`` so that module keeps its maintainability grade; ``config.py``
calls :func:`reset`, :func:`load_model_ids` and :func:`load` at the matching points of its
own load cycle. :func:`load` never raises: each invalid key logs a WARNING and keeps its
default, because the translation worker treats a failed ``load_config`` as fatal.

``process_video`` reloads the config for every input, so the ``--asr`` value is kept apart
from the loaded state and survives :func:`reset`.
"""

import math
from collections.abc import Callable, Mapping
from typing import Any

from ..asr.languages import NVIDIA_EU25, normalize_iso639_1

ENGINES = ("whisper", "canary", "parakeet", "auto")
NVIDIA_ENGINES = ("canary", "parakeet")
ROUTE_ENGINES = ("whisper",) + NVIDIA_ENGINES
DEFAULT_MODELS = {"canary": "nvidia/canary-1b-v2", "parakeet": "nvidia/parakeet-tdt-0.6b-v3"}
# Offline loads fail without an explicit revision, so the shipped checkpoints are pinned.
MODEL_REVISIONS = {
    "nvidia/canary-1b-v2": "d455706339a6b32e1aa40f82c713a482a0c938e2",
    "nvidia/parakeet-tdt-0.6b-v3": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
}
# Languages where Canary beat Whisper large-v3 on the full FLEURS test split by at least 10 %
# relative with a 99 % paired CI (95 % for Romanian), VoxPopuli agreeing where it has the
# language (RTX 5090, 2026-10-10; docs/hardware_optimization.md). Everything else stays on Whisper.
CANARY_DEFAULT_LANGUAGES = ("bg", "et", "hr", "lt", "lv", "mt", "ro", "sk", "sl")
DEFAULTS: dict[str, Any] = {
    "engine": "auto",
    "routes": dict.fromkeys(CANARY_DEFAULT_LANGUAGES, "canary"),
    "max_segment_seconds": 15.0,
    "force_detected_language": False,
}
MIN_SEGMENT_SECONDS = 5.0
MAX_SEGMENT_SECONDS = 30.0

Logger = Callable[..., Any]

_STATE: dict[str, Any] = {}
_CLI: dict[str, str | None] = {"engine": None}


def reset() -> None:
    """Restore the defaults; a ``--asr`` override stays in force."""
    _STATE.clear()
    _STATE.update(DEFAULTS)
    _STATE["routes"] = dict(DEFAULTS["routes"])
    _STATE["models"] = dict(DEFAULT_MODELS)


def load(config_data: Mapping[str, Any], logger_func: Logger) -> None:
    """Apply ``asr.*`` and ``whisper.force_detected_language``, warning on each invalid key."""
    section = _section(config_data, "asr", logger_func)
    _apply_key(section, "engine", _parse_engine, logger_func)
    _apply_key(section, "routes", _parse_routes, logger_func)
    _apply_key(section, "max_segment_seconds", _parse_segment_cap, logger_func)
    _apply_key(_section(config_data, "whisper", logger_func), "force_detected_language", _parse_flag, logger_func)
    source = " (--asr)" if _CLI["engine"] else ""
    logger_func(f"[Config] ASR Engine: {active_engine()}{source}, Routes: {routes() or 'none'}, Max Segment: {max_segment_seconds():g}s")


def load_model_ids(models_config: Mapping[str, Any]) -> None:
    """Apply ``models.canary`` / ``models.parakeet``; a blank or non-string id keeps the default."""
    for engine in NVIDIA_ENGINES:
        value = models_config.get(engine)
        if isinstance(value, str) and value.strip():
            _STATE["models"][engine] = value.strip()


def set_cli_override(engine: str | None) -> None:
    """Pin the engine from ``--asr``; None clears the override."""
    if engine is not None and engine not in ENGINES:
        raise ValueError(f"Unknown ASR engine '{engine}'; expected one of: {', '.join(ENGINES)}")
    _CLI["engine"] = engine


def active_engine() -> str:
    """Return the requested engine: the ``--asr`` override, else ``asr.engine``."""
    return _CLI["engine"] or str(_STATE["engine"])


def routes() -> dict[str, str]:
    """Return a copy of the language -> engine routes used by ``auto``."""
    return dict(_STATE["routes"])


def max_segment_seconds() -> float:
    """Return the longest speech span handed to an NVIDIA engine, in seconds."""
    return float(_STATE["max_segment_seconds"])


def force_detected_language() -> bool:
    """Return whether the voted language is passed to Whisper as ``language=``."""
    return bool(_STATE["force_detected_language"])


def model_id(engine: str) -> str:
    """Return the Hugging Face model id configured for an NVIDIA engine."""
    return str(_STATE["models"][engine])


def model_revision(engine: str) -> str | None:
    """Return the pinned revision when the engine uses its shipped checkpoint, else None."""
    return MODEL_REVISIONS.get(model_id(engine))


def _section(config_data: Mapping[str, Any], name: str, logger_func: Logger) -> Mapping[str, Any]:
    """Return a config section as a mapping, warning when it is something else."""
    value = config_data.get(name)
    if value is None or isinstance(value, Mapping):
        return value or {}
    logger_func(f"[Config] {name} section must be a mapping; ignoring it for ASR settings.", "WARNING")
    return {}


def _apply_key(section: Mapping[str, Any], key: str, parser: Callable[[Any, Logger], Any], logger_func: Logger) -> None:
    """Store ``section[key]`` when present and the parser accepts it (returns non-None)."""
    if key not in section:
        return
    value = parser(section[key], logger_func)
    if value is not None:
        _STATE[key] = value


def _parse_engine(value: Any, logger_func: Logger) -> str | None:
    """Parse ``asr.engine`` case-insensitively."""
    engine = value.strip().lower() if isinstance(value, str) else ""
    if engine in ENGINES:
        return engine
    logger_func(f"[Config] Invalid asr.engine {value!r}; expected one of: {', '.join(ENGINES)}. Keeping the default.", "WARNING")
    return None


def _parse_routes(value: Any, logger_func: Logger) -> dict[str, str] | None:
    """Parse ``asr.routes``; an empty or null mapping disables routing, bad entries are skipped."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        logger_func("[Config] asr.routes must be a mapping of language -> engine. Keeping the default.", "WARNING")
        return None
    return _valid_routes(value, logger_func)


def _valid_routes(value: Mapping[Any, Any], logger_func: Logger) -> dict[str, str]:
    """Return the route entries that pass validation."""
    parsed = (_parse_route(language, engine, logger_func) for language, engine in value.items())
    return {entry[0]: entry[1] for entry in parsed if entry}


def _parse_route(raw_language: Any, raw_engine: Any, logger_func: Logger) -> tuple[str, str] | None:
    """Validate one route entry, returning ``(language, engine)`` or None after a warning."""
    language = _route_key(raw_language)
    engine = raw_engine.strip().lower() if isinstance(raw_engine, str) else ""
    problem = _route_problem(language, engine)
    if problem:
        logger_func(f"[Config] Ignoring asr.routes entry {raw_language!r}: {raw_engine!r} ({problem}).", "WARNING")
        return None
    return language, engine


def _route_key(raw_language: Any) -> str:
    """Normalise a route key; YAML 1.1 reads an unquoted ``no`` (Norwegian) as False."""
    if raw_language is False:
        return "no"
    return normalize_iso639_1(raw_language) or ""


def _route_problem(language: str, engine: str) -> str | None:
    """Explain why a normalised route cannot be used, or return None."""
    if not language:
        return "not a two-letter ISO 639-1 code"
    if engine not in ROUTE_ENGINES:
        return f"engine must be one of: {', '.join(ROUTE_ENGINES)}"
    if engine in NVIDIA_ENGINES and language not in NVIDIA_EU25:
        return f"{engine} does not support '{language}'"
    return None


def _parse_segment_cap(value: Any, logger_func: Logger) -> float | None:
    """Parse ``asr.max_segment_seconds``, clamped to [MIN_SEGMENT_SECONDS, MAX_SEGMENT_SECONDS]."""
    seconds = _to_float(value)
    if math.isnan(seconds):
        logger_func(f"[Config] Invalid asr.max_segment_seconds {value!r}. Keeping the default.", "WARNING")
        return None
    clamped = min(MAX_SEGMENT_SECONDS, max(MIN_SEGMENT_SECONDS, seconds))
    if clamped != seconds:
        logger_func(f"[Config] asr.max_segment_seconds {value!r} is outside [5, 30]; using {clamped:g}.", "WARNING")
    return clamped


def _to_float(value: Any) -> float:
    """Convert a config number to float; anything unusable (incl. booleans) becomes NaN."""
    if isinstance(value, bool):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _parse_flag(value: Any, logger_func: Logger) -> bool | None:
    """Accept only a YAML boolean for ``whisper.force_detected_language``."""
    if isinstance(value, bool):
        return value
    logger_func(f"[Config] whisper.force_detected_language must be true or false, not {value!r}. Keeping the default.", "WARNING")
    return None


reset()
