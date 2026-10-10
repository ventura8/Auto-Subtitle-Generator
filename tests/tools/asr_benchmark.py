"""ASR benchmark harness: Whisper, Canary and Parakeet through the production code paths.

Run from the repository root, which is where ``config.yaml`` is read from, as the pipeline does::

    python -m tests.tools.asr_benchmark --engines whisper,canary,parakeet --langs ro_ro --limit 200 \\
        --voxpopuli ro --probes --longform 30 --code-switch --segment-sweep --canary-beams --device-evidence

Sections (each one optional):

- utterance: FLEURS (``--langs``) and VoxPopuli (``--voxpopuli``) WER/CER, RTF and
  memory per engine, with a paired cluster-bootstrap CI against Whisper;
- canary_beams: Canary greedy versus beam search on FLEURS Romanian;
- probes: characters per minute on silence, noise, a tone and reversed speech,
  end to end through VAD and filters;
- longform / segment_sweep: FLEURS Romanian joined into one long file with known
  boundaries (WER, dropped utterances, median cue-onset error), optionally at
  several ``max_segment_seconds`` caps;
- code_switch: 30 s of English then 5 min of Romanian; the spread language-ID vote must pick ``ro``.

Datasets land in the Hugging Face cache; results (JSON plus Markdown) and the
synthetic WAVs go only to ``--out`` (default ``.cache/asr_benchmark/``).
"""

import argparse
import functools
import importlib.metadata
import os
import platform
import shutil
import subprocess
import time
from collections import namedtuple
from typing import Any

from modules.asr import audio_source, language_id
from modules.configuration import asr_settings, config
from modules.models import OPTIMIZER, ModelManager
from tests.tools import asr_bench_data as data
from tests.tools import asr_bench_engines as engines
from tests.tools import asr_bench_scoring as scoring
from tests.tools.asr_bench_report import write_report

DEFAULT_OUT = os.path.join(".cache", "asr_benchmark")
LONGFORM_MINUTES = 30.0
LONGFORM_LANGUAGE = "ro"
SWEEP_SECONDS = (8.0, 12.0, 15.0, 20.0)
CANARY_BEAMS = (1, 4)
CODE_SWITCH_EN_SECONDS = 30.0
CODE_SWITCH_RO_SECONDS = 300.0
CODE_SWITCH_EN_UTTERANCES = 20
PROBE_LANGUAGE = "ro"
TOP_LANGUAGES = 5

Bench = namedtuple("Bench", ["args", "manager", "cache"])


def _cached(bench: Bench, key: tuple, loader: Any) -> Any:
    """Load a dataset or build a synthetic file once per run."""
    if key not in bench.cache:
        bench.cache[key] = loader()
    return bench.cache[key]


def _fleurs(bench: Bench, lang_dir: str, limit: int | None = None) -> data.Corpus:
    """Cached ``load_fleurs``."""
    return _cached(bench, ("fleurs", lang_dir, limit), functools.partial(data.load_fleurs, lang_dir, limit))


def _voxpopuli(bench: Bench, lang: str, limit: int | None) -> data.Corpus:
    """Cached ``load_voxpopuli``."""
    return _cached(bench, ("voxpopuli", lang, limit), functools.partial(data.load_voxpopuli, lang, limit))


def _romanian_sentences(bench: Bench) -> list:
    """FLEURS Romanian, one reading per sentence, for every synthetic file."""
    return data.one_per_sentence(_fleurs(bench, "ro_ro").utterances)


def _job(bench: Bench, engine: str, language: str, max_segment: float) -> engines.Job:
    """A whole-file run; the NVIDIA batch is ``--batch-size`` or the production VRAM-sized batch for this cap."""
    return engines.Job(engine, language, max_segment, engines.batch_size(bench.manager, engine, max_segment, bench.args.batch_size))


# --- utterance-level accuracy -------------------------------------------------------------------


def _batch_size(bench: Bench, engine: str) -> int:
    """Whisper is timed per utterance; the NVIDIA engines per batch, sized for the configured segment cap."""
    return engines.batch_size(bench.manager, engine, asr_settings.max_segment_seconds(), bench.args.batch_size)


def _utterance_cell(bench: Bench, engine: str, corpus: data.Corpus) -> dict:
    """Score one engine on one corpus."""
    print(f"[bench] {engine} on {corpus.name}: {len(corpus.utterances)} utterances")
    decoder = functools.partial(engines.UTTERANCE_DECODERS[engine], bench.manager, corpus.language)
    size = _batch_size(bench, engine)
    with engines.GpuSampler(bench.args.device_evidence) as sampler:
        engines.reset_torch_peak()
        texts, timings = scoring.decode_corpus(decoder, corpus.utterances, size)
    pairs = scoring.pairs_of(corpus, texts)
    return {
        "engine": engine,
        "corpus": corpus.name,
        "utterances": len(pairs),
        "batch_size": size,
        "audio_seconds": scoring.audio_seconds(timings),
        "rtf": scoring.rtf(timings),
        **scoring.scores(pairs),
        **sampler.fields(),
        "hypotheses": [{"id": utterance.uid, "ref": utterance.text, "hyp": text} for utterance, text in zip(corpus.utterances, texts)],
        "_errors": scoring.word_errors(pairs),
        "_clusters": [utterance.cluster for utterance in corpus.utterances],
    }


def _engine_cells(bench: Bench, engine: str, corpora: list) -> list[dict]:
    """Every corpus the engine supports, with the model loaded and warmed up once."""
    supported = [corpus for corpus in corpora if engines.supports(engine, corpus.language)]
    if supported:
        engines.prepare_engine(bench.manager, engine)
    return [_utterance_cell(bench, engine, corpus) for corpus in supported]


def utterance_section(bench: Bench) -> list[dict]:
    """FLEURS and VoxPopuli utterance-level accuracy, speed and memory for every engine."""
    args = bench.args
    corpora = [_fleurs(bench, lang_dir, args.limit) for lang_dir in args.langs]
    corpora += [_voxpopuli(bench, lang, args.limit) for lang in args.voxpopuli]
    cells: list[dict] = []
    for engine in args.engines:
        cells.extend(_engine_cells(bench, engine, corpora))
        engines.release_models(bench.manager)
    scoring.add_deltas(cells)
    return cells


def _beam_cell(bench: Bench, corpus: data.Corpus, beams: int) -> dict:
    """Canary at one beam width."""
    print(f"[bench] canary beams={beams} on {corpus.name}")
    decoder = functools.partial(engines.canary_beam_batch, beams, bench.manager, corpus.language)
    texts, timings = scoring.decode_corpus(decoder, corpus.utterances, _batch_size(bench, "canary"))
    pairs = scoring.pairs_of(corpus, texts)
    return {
        "engine": "canary",
        "beams": beams,
        "corpus": corpus.name,
        "utterances": len(pairs),
        "rtf": scoring.rtf(timings),
        **scoring.scores(pairs),
    }


def beams_section(bench: Bench) -> list[dict]:
    """Canary greedy versus beam search on FLEURS Romanian."""
    corpus = _fleurs(bench, "ro_ro", bench.args.limit)
    engines.prepare_engine(bench.manager, "canary")
    cells = [_beam_cell(bench, corpus, beams) for beams in bench.args.canary_beams]
    engines.release_models(bench.manager)
    return cells


# --- end-to-end runs ----------------------------------------------------------------------------


def _probe_cell(bench: Bench, engine: str, kind: str, path: str) -> dict:
    """Characters per minute an engine emits on audio that holds no speech (lower is better)."""
    job = _job(bench, engine, PROBE_LANGUAGE, asr_settings.max_segment_seconds())
    cues, info, _elapsed = engines.transcribe_file(bench.manager, job, path)
    return {
        "engine": engine,
        "probe": kind,
        "cues": len(cues),
        "chars_per_minute": sum(len(cue[2]) for cue in cues) / (data.PROBE_SECONDS / 60.0),
        "details": info,
        "text": " | ".join(cue[2] for cue in cues)[:500],
    }


def probes_section(bench: Bench) -> list[dict]:
    """Every engine end to end on silence, noise, a tone and reversed speech."""
    speech = _romanian_sentences(bench)
    paths = {kind: data.write_wav(bench.args.out, f"probe_{kind}", data.probe_audio(kind, speech)) for kind in data.PROBE_GENERATORS}
    cells: list[dict] = []
    for engine in bench.args.engines:
        engines.prepare_engine(bench.manager, engine)
        cells.extend(_probe_cell(bench, engine, kind, path) for kind, path in paths.items())
        engines.release_models(bench.manager)
    return cells


def _build_longform(bench: Bench, minutes: float) -> data.LongForm:
    """Synthetic Romanian long-form file of about ``minutes``."""
    utterances = data.take_seconds(_romanian_sentences(bench), minutes * 60.0)
    return data.build_longform(utterances, bench.args.out, f"longform_ro_{minutes:g}min")


def _longform(bench: Bench) -> data.LongForm:
    """Cached synthetic long-form file of ``--longform`` minutes (default 30)."""
    minutes = bench.args.longform or LONGFORM_MINUTES
    return _cached(bench, ("longform", minutes), functools.partial(_build_longform, bench, minutes))


def _longform_cell(bench: Bench, job: engines.Job, longform: data.LongForm) -> dict:
    """One engine at one segment cap on the synthetic long-form file."""
    print(f"[bench] {job.engine} long-form ({longform.seconds / 60:.1f} min, max segment {job.max_segment:g} s)")
    with engines.GpuSampler(bench.args.device_evidence) as sampler:
        engines.reset_torch_peak()
        cues, info, elapsed = engines.transcribe_file(bench.manager, job, longform.path)
    return {
        "engine": job.engine,
        "max_segment_seconds": job.max_segment,
        "batch_size": job.batch_size,
        "minutes": round(longform.seconds / 60.0, 2),
        "rtf": elapsed / longform.seconds,
        "details": info,
        **scoring.longform_metrics(longform.truth, cues),
        **sampler.fields(),
    }


def longform_section(bench: Bench) -> list[dict]:
    """Every engine end to end on the synthetic Romanian long-form file at the configured segment cap."""
    longform = _longform(bench)
    cells = []
    for engine in bench.args.engines:
        engines.prepare_engine(bench.manager, engine)
        cells.append(_longform_cell(bench, _job(bench, engine, LONGFORM_LANGUAGE, asr_settings.max_segment_seconds()), longform))
        engines.release_models(bench.manager)
    return cells


def sweep_section(bench: Bench) -> list[dict]:
    """The NVIDIA engines on the long-form file at each ``--segment-sweep`` cap (Whisper has no such cap)."""
    longform = _longform(bench)
    cells: list[dict] = []
    for engine in [engine for engine in bench.args.engines if engine != "whisper"]:
        engines.prepare_engine(bench.manager, engine)
        cells.extend(_longform_cell(bench, _job(bench, engine, LONGFORM_LANGUAGE, cap), longform) for cap in bench.args.segment_sweep)
        engines.release_models(bench.manager)
    return cells


# --- language-ID code-switch probe --------------------------------------------------------------


def _vote(whisper: Any, path: str) -> tuple[Any, float]:
    """The production spread language-ID vote over block-VAD speech spans, and its wall seconds."""
    source = audio_source.open_pcm_source(path, data.transcode_path(path))
    try:
        vad = audio_source.asr_vad_settings(asr_settings.max_segment_seconds(), config.VAD_MIN_SILENCE_MS)
        spans = audio_source.iter_speech_spans(source, vad)
        started = time.perf_counter()
        vote = language_id.detect_language(whisper, source, spans)
        return vote, time.perf_counter() - started
    finally:
        source.close()


def _vote_fields(vote: Any) -> dict:
    """JSON fields of a ``LanguageVote`` (or of no vote)."""
    if vote is None:
        return {"vote": None, "probability": None, "share": None, "distribution": {}}
    top = dict(list(vote.distribution.items())[:TOP_LANGUAGES])
    return {"vote": vote.language, "probability": vote.probability, "share": vote.share, "distribution": top}


def code_switch_section(bench: Bench) -> dict:
    """An English intro must not decide the language of a Romanian file, as faster-whisper's first window would."""
    english = data.take_seconds(_fleurs(bench, "en_us", CODE_SWITCH_EN_UTTERANCES).utterances, CODE_SWITCH_EN_SECONDS)
    audio, english_seconds = data.code_switch_audio(english, data.take_seconds(_romanian_sentences(bench), CODE_SWITCH_RO_SECONDS))
    path = data.write_wav(bench.args.out, "code_switch_en_ro", audio)
    whisper = bench.manager.get_whisper()
    vote, lid_seconds = _vote(whisper, path)
    first_language, first_probability, _all = whisper.detect_language(audio=audio)
    engines.release_models(bench.manager)
    fields = _vote_fields(vote)
    return {
        "english_seconds": round(english_seconds, 2),
        "total_seconds": round(len(audio) / data.SAMPLE_RATE, 2),
        **fields,
        "lid_seconds": lid_seconds,
        "first_window_language": first_language,
        "first_window_probability": first_probability,
        "passed": fields["vote"] == "ro",
    }


# --- run metadata -------------------------------------------------------------------------------


def _package_version(name: str) -> str | None:
    """Installed version of a distribution, or None when it is missing."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    """Short commit of the working tree, when git is available."""
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run([git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def _models() -> dict:
    """Model ids and pinned revisions the run used."""
    nvidia = asr_settings.NVIDIA_ENGINES
    return {
        "whisper": config.WHISPER_MODEL_SIZE,
        **{engine: asr_settings.model_id(engine) for engine in nvidia},
        "revisions": {engine: asr_settings.model_revision(engine) for engine in nvidia},
    }


def _meta(bench: Bench) -> dict:
    """Run metadata recorded with every report."""
    return {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "faster_whisper": _package_version("faster-whisper"),
        "hardware": OPTIMIZER.snapshot(),
        "models": _models(),
        "args": vars(bench.args),
    }


# --- command line -------------------------------------------------------------------------------


def _csv(value: str) -> list[str]:
    """Split a comma-separated option."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _engine_list(value: str) -> list[str]:
    """``--engines``: a comma-separated subset of whisper, canary, parakeet."""
    selected = _csv(value)
    unknown = [engine for engine in selected if engine not in engines.ENGINES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown engine(s) {unknown}; choose from {', '.join(engines.ENGINES)}")
    return selected


def _positive_int(value: str) -> int:
    """A strictly positive integer option."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return number


def _build_parser() -> argparse.ArgumentParser:
    """The benchmark's command line."""
    parser = argparse.ArgumentParser(prog="python -m tests.tools.asr_benchmark", description="ASR engine benchmark (see module docstring).")
    parser.add_argument("--engines", type=_engine_list, default=list(engines.ENGINES), help="comma-separated engines")
    parser.add_argument("--langs", type=_csv, default=["ro_ro"], help="comma-separated FLEURS directories, e.g. ro_ro,en_us")
    parser.add_argument("--limit", type=_positive_int, default=None, help="utterances per corpus (default: the full test split)")
    parser.add_argument("--voxpopuli", type=_csv, default=[], metavar="LANGS", help="comma-separated VoxPopuli languages, e.g. ro")
    parser.add_argument("--probes", action="store_true", help="non-speech probes, end to end")
    parser.add_argument("--longform", type=float, nargs="?", const=LONGFORM_MINUTES, default=None, metavar="MINUTES")
    parser.add_argument("--code-switch", action="store_true", help="30 s English + 5 min Romanian language-ID probe")
    parser.add_argument("--segment-sweep", type=float, nargs="*", default=None, metavar="SECONDS", help="default 8 12 15 20")
    parser.add_argument("--canary-beams", type=int, nargs="*", default=None, metavar="BEAMS", help="default 1 4")
    parser.add_argument("--batch-size", type=_positive_int, default=None, help="NVIDIA clips per call (default: production VRAM sizing)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output directory (default: .cache/asr_benchmark)")
    parser.add_argument("--device-evidence", action="store_true", help="record nvidia-smi compute-app rows of this process")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line; a bare ``--segment-sweep`` or ``--canary-beams`` takes the default values."""
    args = _build_parser().parse_args(argv)
    if args.segment_sweep == []:
        args.segment_sweep = list(SWEEP_SECONDS)
    if args.canary_beams == []:
        args.canary_beams = list(CANARY_BEAMS)
    return args


def _wants_utterances(args: argparse.Namespace) -> bool:
    """Any FLEURS or VoxPopuli corpus selected."""
    return bool(args.langs or args.voxpopuli)


SECTIONS = (
    ("utterance", _wants_utterances, utterance_section),
    ("canary_beams", lambda args: bool(args.canary_beams), beams_section),
    ("probes", lambda args: args.probes, probes_section),
    ("longform", lambda args: args.longform is not None, longform_section),
    ("segment_sweep", lambda args: bool(args.segment_sweep), sweep_section),
    ("code_switch", lambda args: args.code_switch, code_switch_section),
)


def _config_log(message: str, level: str = "INFO", **_kwargs: Any) -> None:
    """Show config warnings; the benchmark prints its own progress."""
    if level != "INFO":
        print(f"[config] {level}: {message}")


def _setup(args: argparse.Namespace) -> Bench:
    """Load config.yaml and detect the hardware exactly as the pipeline does before its first file."""
    config.load_config(OPTIMIZER, _config_log)
    OPTIMIZER.detect_hardware(verbose=False)
    os.makedirs(args.out, exist_ok=True)
    return Bench(args, ModelManager(), {})


def run_benchmark(bench: Bench) -> dict:
    """Run every selected section."""
    report: dict[str, Any] = {"meta": _meta(bench)}
    for key, enabled, runner in SECTIONS:
        if enabled(bench.args):
            report[key] = runner(bench)
    return report


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m tests.tools.asr_benchmark``."""
    args = parse_args(argv)
    bench = _setup(args)
    try:
        report = run_benchmark(bench)
    finally:
        engines.release_models(bench.manager)
    json_path, md_path = write_report(report, args.out)
    print(f"[bench] wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
