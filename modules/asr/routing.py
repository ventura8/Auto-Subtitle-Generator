"""Pick the ASR engine for one input from the requested engine and its language."""

from collections import namedtuple
from collections.abc import Mapping

from ..configuration.asr_settings import NVIDIA_ENGINES
from .languages import NVIDIA_EU25

# reason is None for a normal choice; otherwise it explains a fallback to Whisper and is logged as a WARNING.
Route = namedtuple("Route", ["engine", "reason"])


def resolve(requested: str, language: str | None, routes: Mapping[str, str]) -> Route:
    """Return the engine to run; an NVIDIA engine that cannot handle the language falls back to Whisper."""
    engine = _target_engine(requested, language, routes)
    if engine == "whisper":
        return Route("whisper", None)
    reason = _unsupported_reason(engine, language)
    return Route("whisper", reason) if reason else Route(engine, None)


def _target_engine(requested: str, language: str | None, routes: Mapping[str, str]) -> str:
    """Return the engine asked for; ``auto`` follows the routes and defaults to Whisper."""
    if requested != "auto":
        return requested
    return routes.get(language, "whisper") if language else "whisper"


def _unsupported_reason(engine: str, language: str | None) -> str | None:
    """Explain why ``engine`` cannot transcribe ``language``, or return None."""
    if engine not in NVIDIA_ENGINES:
        return f"unknown ASR engine '{engine}'"
    if language is None:
        return "no language detected"
    if language not in NVIDIA_EU25:
        return f"{engine} does not support '{language}'"
    return None
