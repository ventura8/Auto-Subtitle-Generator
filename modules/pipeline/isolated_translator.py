"""Isolated translation worker entrypoint and batch helpers."""

import gc
import json
import os
import sys
import tempfile
import time
import traceback
from typing import Any

from modules import utils, workdir
from modules.configuration import config
from modules.models import OPTIMIZER, ModelManager, apply_dynamic_translation_batch
from modules.runtime.optional_imports import load_optional_torch
from modules.utils import log

torch: Any | None = load_optional_torch()

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
    """Log non-sensitive per-item translation progress."""
    _ = batch_texts
    _ = translated_texts
    total_items = len(batch_items)
    for index, item in enumerate(batch_items, start=1):
        timestamp = utils.format_timestamp(item["start"])
        log(f"  [{timestamp}] Item {index}/{total_items} translated.")


def _cleanup_intermediate_memory(has_more_batches):
    """Collect memory between translation batches when more work remains."""
    if has_more_batches:
        _safe_cuda_cleanup()


def _safe_cuda_cleanup():
    """Run GC and clear CUDA cache without surfacing cleanup-only failures."""
    gc.collect()
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except (RuntimeError, OSError, ValueError) as cleanup_err:
        log(f"[Isolation] CUDA cache cleanup warning: {cleanup_err}", "WARNING")


def _is_cuda_oom_error(error):
    """Return True when the exception text indicates a CUDA out-of-memory failure."""
    message = str(error).lower()
    return "out of memory" in message and "cuda" in message


def _translate_chunk_with_oom_fallback(translator, chunk, src_code, tgt_code):
    """Translate chunk with adaptive bisection fallback on CUDA OOM."""
    queue = [chunk]
    translated = []

    while queue:
        current_chunk = queue.pop(0)
        try:
            translated.extend(_translate_batch_chunk(translator, current_chunk, src_code, tgt_code))
            continue
        except RuntimeError as batch_err:
            if not _is_cuda_oom_error(batch_err):
                raise

            _safe_cuda_cleanup()
            if len(current_chunk) <= 1:
                raise

            midpoint = len(current_chunk) // 2
            left_chunk = current_chunk[:midpoint]
            right_chunk = current_chunk[midpoint:]
            log(
                f"    [Batch OOM] Reducing chunk size {len(current_chunk)} -> {len(left_chunk)} + {len(right_chunk)} and retrying.",
                "WARNING",
            )
            queue.insert(0, right_chunk)
            queue.insert(0, left_chunk)

    return translated


def _build_worker_runtime(batch_size):
    """Load config and create the translator runtime for a worker process."""
    if not config.load_config(OPTIMIZER, log):
        raise RuntimeError("Failed to load configuration in isolated worker")
    OPTIMIZER.detect_hardware(verbose=False)
    translator = _resolve_worker_translator(ModelManager())
    _tune_batch_for_free_vram(translator)
    resolved_batch_size = batch_size if batch_size > 0 else _resolve_translation_batch_size()
    return resolved_batch_size, translator


def _resolve_translation_batch_size():
    """Resolve translation batch size based on configured engine."""
    if config.TRANSLATOR_ENGINE == "translategemma":
        return int(OPTIMIZER.config.get("translategemma_batch", OPTIMIZER.config["nllb_batch"]))
    return int(OPTIMIZER.config["nllb_batch"])


def _tune_batch_for_free_vram(translator):
    """Refine the NLLB batch from the VRAM free once the model is resident."""
    if config.TRANSLATOR_ENGINE != "nllb":
        return
    model_id = getattr(translator, "model_id", config.NLLB_MODEL_ID)
    apply_dynamic_translation_batch(model_id, lambda message: log(message, "INFO"))


def _resolve_worker_translator(manager):
    """Resolve translator implementation from configured engine."""
    if config.TRANSLATOR_ENGINE == "nllb":
        return manager.get_nllb()
    if config.TRANSLATOR_ENGINE == "translategemma":
        return manager.get_translategemma()
    raise RuntimeError(f"Unsupported translation engine for isolated worker: {config.TRANSLATOR_ENGINE}")


def _discard_temp_file(temp_path):
    """Safely unlink a temporary file on disk if present."""
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _save_worker_output(output_file, translations):
    """Persist translated worker output to disk."""
    output_dir = os.path.dirname(output_file) or "."
    fd, temp_path = tempfile.mkstemp(dir=output_dir, prefix=f"{os.path.basename(output_file)}.", suffix=".tmp")
    os.close(fd)
    try:
        with open(temp_path, "w", encoding="utf-8") as temp_handle:
            json.dump(translations, temp_handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, output_file)
    except Exception:
        _discard_temp_file(temp_path)
        raise


def _save_job_translations(output_file, translations, data):
    """Atomically persist per-job translations to disk."""
    if len(translations) != len(data):
        raise RuntimeError(f"Translation count mismatch for {output_file}: {len(translations)} != {len(data)}")

    output_dir = os.path.dirname(output_file) or "."
    fd, temp_save_path = tempfile.mkstemp(dir=output_dir, prefix=f"{os.path.basename(output_file)}.", suffix=".tmp")
    os.close(fd)
    try:
        with open(temp_save_path, "w", encoding="utf-8") as temp_handle:
            json.dump(translations, temp_handle, ensure_ascii=False)
        os.replace(temp_save_path, output_file)
    except Exception:
        _discard_temp_file(temp_save_path)
        raise


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

        translated_texts = _translate_chunk_with_oom_fallback(
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
    """Process all manifest-driven translation batches for a job.

    Cues are batched in order of text length and the results put back in cue
    order. Every item in a batch is padded to the longest one and beam search
    runs until the longest finishes, so mixing a five-word cue with a
    forty-word one wastes most of the batch; neighbours of similar length keep
    the padding small and let larger batches actually pay off.
    """
    batch_texts = [item["text"] for item in data]
    batch_size = _resolve_translation_batch_size()
    order = sorted(range(len(batch_texts)), key=lambda index: len(batch_texts[index]))
    translations = [""] * len(batch_texts)
    start_real = time.time()

    for batch_start in range(0, len(order), batch_size):
        indices = order[batch_start : batch_start + batch_size]
        translated = _translate_batch_checked(translator, [batch_texts[index] for index in indices], job_config)
        for index, text in zip(indices, translated):
            translations[index] = text
        _print_job_progress(job_config["prefix_str"], min(batch_start + batch_size, len(order)), len(order), start_real)
        _cleanup_intermediate_memory(batch_start + batch_size < len(order))

    return translations


def _translate_batch_checked(translator, chunk, job_config):
    """Translate one batch and insist on exactly one result per input."""
    translated = _translate_chunk_with_oom_fallback(translator, chunk, job_config["src_code"], job_config["tgt_code"])
    if len(translated) != len(chunk):
        raise RuntimeError(f"Translator returned {len(translated)} results for a batch of {len(chunk)}")
    return translated


def _print_job_progress(prefix, done, total, start_real):
    """Report translation progress by cue count (cues are not processed in time order)."""
    elapsed = time.time() - start_real
    speed = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / speed if speed > 0 else 0
    utils.print_progress_bar(done, total, prefix=prefix, timestamp_str=f"{done}/{total} cues", speed=speed, eta=eta)


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

    if len(translations) != len(data):
        raise RuntimeError(f"Translation count mismatch for {job_config['output_file']}: {len(translations)} != {len(data)}")

    _save_worker_output(job_config["output_file"], translations)

    log("[Isolation] Success. Worker exiting.", level="DEBUG")


def _translate_batch_chunk(translator, chunk, src_code, tgt_code):
    """Refactored helper for inference."""
    try:
        return translator.translate(chunk, src_code, tgt_code)
    except (RuntimeError, ValueError, OSError) as batch_err:
        log(f"    [Batch Error] Chunk failed: {batch_err}", "ERROR")
        raise RuntimeError(f"Chunk translation failed: {batch_err}") from batch_err


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
        if len(translations) != len(data):
            raise RuntimeError(f"Translation count mismatch for {job_config['output_file']}: {len(translations)} != {len(data)}")
        _save_job_translations(job_config["output_file"], translations, data)
        _wait_for_parent_to_consume_output(job_config["output_file"], lang)

    except (OSError, RuntimeError, ValueError, KeyError) as e:
        log(f"[Isolation] Job {lang} failed: {e}", "ERROR")
        raise
    finally:
        # Aggressive Cleanup
        _safe_cuda_cleanup()


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


def _save_optional_reused_pivot_english_output(pivot_job, pivot_data):
    """Emit English job output from an existing pivot file when requested."""
    if not pivot_job.get("emit_en_output"):
        return

    en_output = pivot_job.get("en_output")
    if not en_output:
        return

    translations = [item["text"] for item in pivot_data]
    _save_job_translations(en_output, translations, pivot_data)


def _reuse_existing_pivot_output_if_available(pivot_job):
    """Reuse a previously generated pivot output file when it matches the current input.

    The pivot file survives failed or interrupted runs so it can be resumed.
    A leftover whose cue timings no longer line up with the current segments
    (for example after re-transcription) is discarded instead of reused.
    """
    output_path = pivot_job["output"]
    if not os.path.exists(output_path):
        return False

    pivot_data = _load_pivot_output_matching_input(output_path, pivot_job["input"])
    if pivot_data is None:
        log("[Isolation] Existing pivot output does not match current segments. Discarding it.", "WARNING")
        _discard_temp_file(output_path)
        return False
    log("[Isolation] Reusing existing pivot output. Skipping pivot pass.")
    _save_optional_reused_pivot_english_output(pivot_job, pivot_data)
    return True


def _load_pivot_output_matching_input(output_path, input_path):
    """Return the stored pivot data when its timings match the input segments, else None."""
    try:
        pivot_data = _load_segments(output_path)
        source_data = _load_segments(input_path)
    except (OSError, json.JSONDecodeError):
        return None
    return pivot_data if _pivot_matches_source(pivot_data, source_data) else None


def _pivot_matches_source(pivot_data, source_data):
    """Return True when every stored pivot cue lines up with its source cue."""
    if not isinstance(pivot_data, list) or not isinstance(source_data, list):
        return False
    if len(pivot_data) != len(source_data):
        return False
    return all(_pivot_item_matches(pivot_item, source_item) for pivot_item, source_item in zip(pivot_data, source_data))


def _pivot_item_matches(pivot_item, source_item):
    """Return True when a stored pivot cue carries text and the same timings as its source cue."""
    if not isinstance(pivot_item, dict) or not isinstance(source_item, dict):
        return False
    if not isinstance(pivot_item.get("text"), str):
        return False
    return pivot_item.get("start") == source_item.get("start") and pivot_item.get("end") == source_item.get("end")


def _run_pivot_phase(pivot_job, translator):
    """Run optional pivot translation before target jobs."""
    if not pivot_job:
        return

    try:
        _execute_pivot_phase(pivot_job, translator)
    except PIVOT_ERROR_TYPES as e:
        log(f"[Isolation] Pivot phase failed: {e}", "ERROR")
        raise


def _execute_pivot_phase(pivot_job, translator):
    """Execute pivot phase steps when a reusable output is unavailable."""
    if _reuse_existing_pivot_output_if_available(pivot_job):
        return

    data = _load_segments(pivot_job["input"])
    if not data:
        return

    _translate_and_save_pivot_output(pivot_job, translator, data)


def _translate_and_save_pivot_output(pivot_job, translator, data):
    """Translate pivot data and persist required outputs."""
    log("[Isolation] Running pivot pass in batch worker...")
    job_config = _build_pivot_job_config(pivot_job)
    translations = _run_job_batches(data, translator, job_config)
    if len(translations) != len(data):
        raise RuntimeError(f"Pivot translation count mismatch: {len(translations)} != {len(data)}")

    pivot_data = _build_pivot_source_data(data, translations)
    _save_pivot_output_atomic(pivot_job["output"], pivot_data)
    _save_optional_pivot_english_output(pivot_job, translations, data)


def _build_pivot_job_config(pivot_job):
    """Build translation configuration used for pivot pass execution."""
    return {
        "src_code": pivot_job["src_code"],
        "tgt_code": pivot_job["tgt_code"],
        "prefix_str": "  [Translate Pivot] English",
    }


def _save_pivot_output_atomic(output_path, pivot_data):
    """Persist pivot intermediate data via atomic temp-file replace."""
    output_dir = os.path.dirname(output_path) or "."
    fd, temp_path = tempfile.mkstemp(dir=output_dir, prefix=f"{os.path.basename(output_path)}.", suffix=".tmp")
    os.close(fd)
    try:
        with open(temp_path, "w", encoding="utf-8") as temp_handle:
            json.dump(pivot_data, temp_handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, output_path)
    except Exception:
        _discard_temp_file(temp_path)
        raise


def _save_optional_pivot_english_output(pivot_job, translations, data):
    """Write optional direct English output when requested by manifest."""
    if not pivot_job.get("emit_en_output"):
        return
    en_output = pivot_job.get("en_output")
    if not en_output:
        return
    _save_job_translations(en_output, translations, data)


class ManifestPathError(ValueError):
    """Raised when a manifest carries a path outside the work directory holding it."""


_MANIFEST_PATH_KEYS = ("input", "output", "en_output")
_MANIFEST_SUFFIX = ".manifest.json"


def _contained_path(work_root, candidate):
    """Return ``candidate`` resolved, or raise when it escapes ``work_root``."""
    resolved = os.path.realpath(candidate)
    if resolved != work_root and not resolved.startswith(work_root + os.sep):
        raise ManifestPathError(f"Manifest path escapes the work directory: {candidate}")
    return resolved


def _contain_job_paths(job, work_root):
    """Rewrite a job's path entries to validated paths inside the work directory."""
    for key in _MANIFEST_PATH_KEYS:
        value = job.get(key)
        if value:
            job[key] = _contained_path(work_root, value)


def _validated_manifest_path(manifest_path):
    """Return the resolved manifest path, refusing one outside a work directory.

    The parent always writes the manifest as ``<base>.manifest.json`` inside
    ``<base>`` + ``WORK_DIR_SUFFIX``. Requiring exactly that shape means a path
    handed to the worker on the command line cannot aim the batch run at an
    arbitrary file elsewhere on disk.
    """
    resolved = os.path.realpath(manifest_path)
    if not os.path.basename(resolved).endswith(_MANIFEST_SUFFIX):
        raise ManifestPathError(f"Manifest must be a {_MANIFEST_SUFFIX} file: {manifest_path}")

    work_root = os.path.dirname(resolved)
    if not os.path.basename(work_root).endswith(workdir.WORK_DIR_SUFFIX):
        raise ManifestPathError(f"Manifest must live in a {workdir.WORK_DIR_SUFFIX} directory: {manifest_path}")

    return resolved


def _load_contained_manifest(manifest_path):
    """Load the manifest, confining every path it carries to the manifest's own directory.

    The worker is spawned with a manifest inside the per-video work directory and
    every file it reads or writes belongs there. Resolving each path and refusing
    anything outside that directory stops a manipulated manifest from redirecting
    a read or a write elsewhere on disk.
    """
    resolved_manifest = _validated_manifest_path(manifest_path)
    work_root = os.path.dirname(resolved_manifest)

    with open(resolved_manifest, "r", encoding="utf-8") as file_handle:
        manifest = json.load(file_handle)

    for job in manifest.get("jobs", []):
        _contain_job_paths(job, work_root)

    pivot_job = manifest.get("pivot")
    if pivot_job:
        _contain_job_paths(pivot_job, work_root)

    return manifest


def run_batch_translation_worker(manifest_path):
    """Executes multiple translation jobs with a single model load."""
    manifest = _load_contained_manifest(manifest_path)

    jobs = manifest.get("jobs", [])
    pivot_job = manifest.get("pivot")
    if not jobs and not pivot_job:
        log("[Isolation] No jobs in manifest. Exiting.")
        return

    # 1. Global Init (Load Model ONCE)
    log(f"[Isolation] Batch Mode: Processing {len(jobs)} targets...", level="INFO")
    if not config.load_config(OPTIMIZER, log):
        raise RuntimeError("Failed to load configuration in isolated batch worker")
    OPTIMIZER.detect_hardware(verbose=False)

    manager = ModelManager()
    translator = _resolve_worker_translator(manager)
    _tune_batch_for_free_vram(translator)

    batch_failure = _run_batch_jobs_with_failure_capture(pivot_job, jobs, translator)

    log("[Isolation] Batch Processing Complete.")
    if batch_failure is not None:
        raise batch_failure


def _run_batch_jobs_with_failure_capture(pivot_job, jobs, translator):
    """Run pivot and job phases, returning the first captured failure if any."""
    try:
        _run_pivot_phase(pivot_job, translator)
    except PIVOT_ERROR_TYPES as e:
        return e

    for idx, job in enumerate(jobs):
        try:
            _process_single_job(job, idx, len(jobs), translator)
        except MAIN_FATAL_ERROR_TYPES as e:
            log(f"[Isolation] Job batch failure: {e}", "ERROR")
            return e

    return None


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
        utils.setup_signal_handlers()

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
