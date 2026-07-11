"""Isolated translation worker entrypoint and batch helpers."""

import sys
import os
import json
import traceback
import time
import gc
import tempfile
import warnings
from modules import config, utils
from modules.models import ModelManager, OPTIMIZER, log

try:
    import torch
except ImportError:
    torch = None

warnings.filterwarnings("ignore", category=UserWarning, message=".*expandable_segments not supported.*")
warnings.filterwarnings("ignore", message=".*The following generation flags are not valid.*")

PIVOT_ERROR_TYPES = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    json.JSONDecodeError,
)

MAIN_FATAL_ERROR_TYPES = (
    RuntimeError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    json.JSONDecodeError,
)


def _build_progress_values(current_audio_time, total_dur, start_real):
    """Calculate progress-bar timing values for the current batch."""
    elapsed = time.time() - start_real
    speed = current_audio_time / elapsed if elapsed > 0 else 0
    eta = (total_dur - current_audio_time) / speed if speed > 0 else 0
    timestamp = f"{utils.format_timestamp(current_audio_time)} / {utils.format_timestamp(total_dur)}"
    return speed, eta, timestamp


def _log_batch_texts(batch_items, batch_texts, translated_texts):
    """Log source and target strings for a translated batch."""
    for item, src_text, tgt_text in zip(batch_items, batch_texts, translated_texts):
        timestamp = utils.format_timestamp(item["start"])
        log(f"  [{timestamp}] SRC: {src_text}")
        log(f"  [{timestamp}] TGT: {tgt_text}")


def _cleanup_intermediate_memory(has_more_batches):
    """Collect memory between translation batches when more work remains."""
    if has_more_batches:
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _build_worker_runtime(batch_size):
    """Load config and create the translator runtime for a worker process."""
    config.load_config(OPTIMIZER, log)
    OPTIMIZER.detect_hardware(verbose=False)
    resolved_batch_size = batch_size if batch_size > 0 else OPTIMIZER.config["nllb_batch"]
    translator = ModelManager().get_nllb()
    return resolved_batch_size, translator


def _save_worker_output(output_file, translations):
    """Persist translated worker output to disk."""
    output_dir = os.path.dirname(output_file) or "."
    fd, temp_path = tempfile.mkstemp(dir=output_dir)
    os.close(fd)
    with open(temp_path, "w", encoding="utf-8") as temp_handle:
        json.dump(translations, temp_handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, output_file)


def _save_job_translations(output_file, translations, data):
    """Atomically persist per-job translations to disk."""
    if len(translations) != len(data):
        while len(translations) < len(data):
            translations.append("Translation Error")

    output_dir = os.path.dirname(output_file) or "."
    fd, temp_save_path = tempfile.mkstemp(dir=output_dir)
    os.close(fd)
    with open(temp_save_path, "w", encoding="utf-8") as temp_handle:
        json.dump(translations, temp_handle, ensure_ascii=False)
    os.replace(temp_save_path, output_file)


def _wait_for_parent_to_consume_output(output_file, lang):
    """Wait briefly for the parent process to consume a job output file."""
    wait_start = time.time()
    while os.path.exists(output_file):
        time.sleep(0.05)
        if time.time() - wait_start > 10:
            log(f"[Isolation] Warning: Parent timed out consuming {lang} output.", "WARNING")
            break


def _load_segments(input_file):
    """Load serialized subtitle segments from disk."""
    with open(input_file, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _build_worker_job(*worker_args):
    """Normalize the legacy worker argument tuple into a mapping."""
    keys = (
        "input_file",
        "output_file",
        "src_lang",
        "tgt_lang",
        "batch_size",
        "lang_label",
        "prefix_str",
    )
    return dict(zip(keys, worker_args))


def _run_worker_batches(data, translator, batch_size, job_config):
    """Process all worker batches and return translated strings."""
    translations = []
    total_dur = data[-1]["end"] if data else 0
    start_real = time.time()

    for batch_start in range(0, len(data), batch_size):
        batch_items = data[batch_start : batch_start + batch_size]
        batch_texts = [item["text"] for item in batch_items]
        current_audio_time = batch_items[-1]["end"]

        translated_texts = _translate_batch_chunk(
            translator,
            batch_texts,
            job_config["src_lang"],
            job_config["tgt_lang"],
        )
        translations.extend(translated_texts)
        speed, eta, timestamp = _build_progress_values(
            current_audio_time,
            total_dur,
            start_real,
        )

        if translated_texts:
            _log_batch_texts(batch_items, batch_texts, translated_texts)

        utils.print_progress_bar(
            current_audio_time,
            total_dur,
            prefix=job_config["prefix_str"],
            timestamp_str=timestamp,
            speed=speed,
            eta=eta,
        )

        _cleanup_intermediate_memory(batch_start + batch_size < len(data))

    return translations


def _run_job_batches(data, translator, job_config):
    """Process all manifest-driven translation batches for a job."""
    batch_texts = [item["text"] for item in data]
    batch_size = OPTIMIZER.config["nllb_batch"]
    translations = []
    total_dur = data[-1]["end"] if data else 0
    start_real = time.time()

    for batch_start in range(0, len(batch_texts), batch_size):
        chunk = batch_texts[batch_start : batch_start + batch_size]
        translations.extend(
            _translate_batch_chunk(
                translator,
                chunk,
                job_config["src_code"],
                job_config["tgt_code"],
            )
        )
        current_idx = min(batch_start + batch_size, len(data))
        current_audio_time = data[current_idx - 1]["end"]
        speed, eta, timestamp = _build_progress_values(
            current_audio_time,
            total_dur,
            start_real,
        )
        utils.print_progress_bar(
            current_audio_time,
            total_dur,
            prefix=job_config["prefix_str"],
            timestamp_str=timestamp,
            speed=speed,
            eta=eta,
        )
        _cleanup_intermediate_memory(batch_start + batch_size < len(batch_texts))

    return translations


def run_translation_worker(*worker_args):
    """Executes the translation job in isolation."""
    job_config = _build_worker_job(*worker_args)
    data = _load_segments(job_config["input_file"])

    log(
        f"[Isolation] Loaded {len(data)} segments for {job_config['lang_label']}.",
        level="DEBUG",
    )

    batch_size, translator = _build_worker_runtime(job_config["batch_size"])

    log(f"[Isolation] PID: {os.getpid()} | Batch Size: {batch_size} (Dynamic Scaling)")
    try:
        translations = _run_worker_batches(data, translator, batch_size, job_config)
    except (RuntimeError, ValueError) as e:
        log(f"[Isolation] Batch Failed: {e}")
        raise

    _save_worker_output(job_config["output_file"], translations)

    log("[Isolation] Success. Worker exiting.", level="DEBUG")


def _translate_batch_chunk(translator, chunk, src_code, tgt_code):
    """Refactored helper for inference."""
    try:
        return translator.translate(chunk, src_code, tgt_code)
    except (RuntimeError, ValueError, OSError) as batch_err:
        log(f"    [Batch Error] Chunk failed: {batch_err}", "ERROR")
        return ["Translation Error"] * len(chunk)


def _process_single_job(job, idx, total_jobs, translator):
    """Helper to process a single translation job within the batch."""
    lang = job.get("lang", "<unknown>")

    try:
        label = job.get("label", lang)
        job_config = {
            "lang": lang,
            "label": label,
            "tgt_code": job["tgt_code"],
            "input_file": job["input"],
            "output_file": job["output"],
            "src_code": job.get("src_code"),
            "prefix_str": f"  [Translate {idx + 1}/{total_jobs}] {label} ({job['tgt_code']})",
        }

        log(f"[Isolation] Job {idx + 1}/{total_jobs}: {label} ({job_config['tgt_code']})")

        data = _load_segments(job_config["input_file"])
        translations = _run_job_batches(data, translator, job_config)
        _save_job_translations(job_config["output_file"], translations, data)
        _wait_for_parent_to_consume_output(job_config["output_file"], lang)

    except (OSError, json.JSONDecodeError, RuntimeError, ValueError, KeyError) as e:
        log(f"[Isolation] Job {lang} failed: {e}", "ERROR")
        return
    finally:
        # Aggressive Cleanup
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _build_pivot_source_data(data, translations):
    """Build segment dictionaries from pivot translations."""
    pivot_data = []
    for translated_text, item in zip(translations, data):
        pivot_data.append(
            {
                "text": translated_text,
                "start": item["start"],
                "end": item["end"],
            }
        )
    return pivot_data


def _run_pivot_phase(pivot_job, translator):
    """Run optional pivot translation before target jobs."""
    if not pivot_job:
        return

    try:
        data = _load_segments(pivot_job["input"])
        if not data:
            return

        log("[Isolation] Running pivot pass in batch worker...")
        job_config = {
            "src_code": pivot_job["src_code"],
            "tgt_code": pivot_job["tgt_code"],
            "prefix_str": "  [Translate Pivot] English",
        }
        translations = _run_job_batches(data, translator, job_config)
        pivot_data = _build_pivot_source_data(data, translations)
        output_path = pivot_job["output"]
        output_dir = os.path.dirname(output_path) or "."
        fd, temp_path = tempfile.mkstemp(dir=output_dir)
        os.close(fd)
        with open(temp_path, "w", encoding="utf-8") as temp_handle:
            json.dump(pivot_data, temp_handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, output_path)

        if pivot_job.get("emit_en_output") and pivot_job.get("en_output"):
            _save_job_translations(pivot_job["en_output"], translations, data)
    except PIVOT_ERROR_TYPES as e:
        log(f"[Isolation] Pivot phase failed: {e}", "ERROR")


def run_batch_translation_worker(manifest_path):
    """Executes multiple translation jobs with a single model load."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    jobs = manifest.get("jobs", [])
    pivot_job = manifest.get("pivot")
    if not jobs and not pivot_job:
        log("[Isolation] No jobs in manifest. Exiting.")
        return

    # 1. Global Init (Load Model ONCE)
    log(f"[Isolation] Batch Mode: Processing {len(jobs)} targets...", level="INFO")
    config.load_config(OPTIMIZER, log)
    OPTIMIZER.detect_hardware(verbose=False)

    manager = ModelManager()
    translator = manager.get_nllb()

    _run_pivot_phase(pivot_job, translator)

    # 2. Process Each Job
    for idx, job in enumerate(jobs):
        _process_single_job(job, idx, len(jobs), translator)

    log("[Isolation] Batch Processing Complete.")


def _run_legacy_mode():
    """Parses arguments and runs legacy single-file mode."""
    if len(sys.argv) < 7:
        print(
            "Usage:\n"
            "  python isolated_translator.py --batch manifest.json\n"
            "  python isolated_translator.py input.json output.json src tgt "
            "batch_size label [step_current step_total]"
        )
        return sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    src_lang = sys.argv[3]
    tgt_lang = sys.argv[4]
    batch_size = int(sys.argv[5])
    lang_label = sys.argv[6]

    prefix_str = f"  [Translate] {lang_label}"
    if len(sys.argv) >= 9:
        step_current = sys.argv[7]
        step_total = sys.argv[8]
        prefix_str = f"  [Translate {step_current}/{step_total}] {lang_label}"

    log("[Isolation] Starting Translation Worker...", level="INFO")

    return run_translation_worker(input_file, output_file, src_lang, tgt_lang, batch_size, lang_label, prefix_str)


def main():
    """CLI entrypoint for isolated translation worker."""
    try:
        utils.init_console()

        # Mode 1: Batch Mode (Manifest)
        if len(sys.argv) == 3 and sys.argv[1] == "--batch":
            manifest_path = sys.argv[2]
            run_batch_translation_worker(manifest_path)
            sys.exit(0)

        # Mode 2: Legacy Single Mode
        _run_legacy_mode()

    except MAIN_FATAL_ERROR_TYPES as e:
        log(f"[Isolation] FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
