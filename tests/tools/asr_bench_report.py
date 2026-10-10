"""JSON and Markdown output of ``tests.tools.asr_benchmark``; both go only to the ``--out`` directory."""

import json
import os
import time
from typing import Any


def _pct(value: float) -> str:
    """A rate as a percentage."""
    return f"{value * 100:.2f}%"


def _num(value: float) -> str:
    """A plain number with three decimals."""
    return f"{value:.3f}"


def _ci(value: list) -> str:
    """A CI of a rate difference, in percentage points."""
    return f"[{value[0] * 100:+.2f}, {value[1] * 100:+.2f}] pp"


UTTERANCE_COLUMNS = (
    ("Corpus", "corpus", str),
    ("Engine", "engine", str),
    ("Utts", "utterances", str),
    ("WER", "wer", _pct),
    ("WER (Open ASR)", "wer_open_asr", _pct),
    ("CER", "cer", _pct),
    ("Diacritic err", "diacritic_error_rate", _pct),
    ("Δ WER vs Whisper (95 % CI)", "delta_wer_vs_whisper_ci95", _ci),
    ("RTF", "rtf", _num),
    ("Torch peak GiB", "torch_peak_gib", _num),
    ("Process GPU peak MiB", "process_peak_mib", str),
)
BEAM_COLUMNS = (("Beams", "beams", str), ("WER", "wer", _pct), ("WER (Open ASR)", "wer_open_asr", _pct), ("RTF", "rtf", _num))
PROBE_COLUMNS = (("Probe", "probe", str), ("Engine", "engine", str), ("Cues", "cues", str), ("Chars/min", "chars_per_minute", _num))
LONGFORM_COLUMNS = (
    ("Engine", "engine", str),
    ("Max segment s", "max_segment_seconds", str),
    ("WER", "wer", _pct),
    ("WER (Open ASR)", "wer_open_asr", _pct),
    ("Dropped spans", "dropped_spans", str),
    ("Median onset err s", "median_onset_error_s", _num),
    ("RTF", "rtf", _num),
    ("Process GPU peak MiB", "process_peak_mib", str),
)
TABLES = (
    ("utterance", "Utterance-level accuracy", UTTERANCE_COLUMNS),
    ("canary_beams", "Canary beam sweep (FLEURS ro)", BEAM_COLUMNS),
    ("probes", "Non-speech probes (end to end, after filters)", PROBE_COLUMNS),
    ("longform", "Synthetic long-form (FLEURS ro)", LONGFORM_COLUMNS),
    ("segment_sweep", "Segment-cap sweep (long-form)", LONGFORM_COLUMNS),
)


def _format(value: Any, formatter: Any) -> str:
    """One table cell; missing values show as a dash."""
    return "–" if value is None else formatter(value)


def _row(row: dict, columns: tuple) -> str:
    """One Markdown table row."""
    return "| " + " | ".join(_format(row.get(key), formatter) for _title, key, formatter in columns) + " |"


def table(rows: list[dict], columns: tuple) -> str:
    """A Markdown table of ``rows``."""
    header = "| " + " | ".join(column[0] for column in columns) + " |"
    rule = "|" + "|".join("---" for _column in columns) + "|"
    return "\n".join([header, rule, *(_row(row, columns) for row in rows), ""])


def _code_switch_lines(result: dict) -> str:
    """The code-switch probe as a short paragraph."""
    verdict = "PASS" if result["passed"] else "FAIL"
    return (
        f"{verdict}: the spread vote picked `{result['vote']}` (share {_format(result['share'], _num)}, "
        f"LID {result['lid_seconds']:.1f} s) on {result['total_seconds']:.0f} s of audio with a "
        f"{result['english_seconds']:.0f} s English intro; faster-whisper's first window says `{result['first_window_language']}`.\n"
    )


def _meta_lines(meta: dict) -> str:
    """Host and library versions."""
    hardware = meta["hardware"]
    return (
        f"Host `{meta['host']}`, GPU `{hardware.get('gpu_name')}` (profile {hardware.get('profile')}), "
        f"torch {meta['torch']}, transformers {meta['transformers']}, faster-whisper {meta['faster_whisper']}, "
        f"commit `{meta['git_commit']}`.\n"
    )


def render_markdown(report: dict) -> str:
    """The Markdown summary written next to the JSON."""
    parts = [f"# ASR benchmark ({report['meta']['started']})", "", _meta_lines(report["meta"])]
    for key, title, columns in TABLES:
        if report.get(key):
            parts += [f"## {title}", "", table(report[key], columns)]
    if "code_switch" in report:
        parts += ["## Code-switch probe", "", _code_switch_lines(report["code_switch"])]
    return "\n".join(parts)


def write_report(report: dict, out_dir: str) -> tuple[str, str]:
    """Write ``asr_benchmark_<timestamp>.json`` and ``.md`` into ``out_dir``."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(out_dir, f"asr_benchmark_{stamp}.json")
    md_path = os.path.join(out_dir, f"asr_benchmark_{stamp}.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=str)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    return json_path, md_path
