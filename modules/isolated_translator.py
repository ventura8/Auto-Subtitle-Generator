import sys
import os
import json
import traceback
import time

# Ensure the root directory is in sys.path for internal module imports
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from modules.models import ModelManager, OPTIMIZER, log  # noqa: E402
from modules import config, utils  # noqa: E402


def run_translation_worker(input_file, output_file, src_lang, tgt_lang, batch_size, lang_label, prefix_str):
    """Executes the translation job in isolation."""
    # 1. Load Data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    log(f"[Isolation] Loaded {len(data)} segments.", level="DEBUG")

    # 2. Init Model (Fresh Environment)
    config.load_config(OPTIMIZER, log)
    OPTIMIZER.detect_hardware(verbose=False)

    # If batch_size arg is provided and > 0, we can use it,
    # otherwise we use the optimizer's dynamic value.
    if batch_size <= 0:
        batch_size = OPTIMIZER.config["nllb_batch"]

    log(f"[Isolation] PID: {os.getpid()} | Batch Size: {batch_size} (Dynamic Scaling)")
    manager = ModelManager()
    translator = manager.get_nllb()

    # 3. Process in Batches
    translations = []
    total = len(data)
    total_dur = data[-1]["end"] if data else 0
    start_real = time.time()

    for i in range(0, total, batch_size):
        batch_items = data[i: i + batch_size]
        batch_texts = [item["text"] for item in batch_items]
        current_audio_time = batch_items[-1]["end"]

        try:
            # Run Translation
            res = translator.translate(batch_texts, src_lang, tgt_lang)
            translations.extend(res)

            # Update progress bar
            elapsed = time.time() - start_real
            # Speed is audio-seconds per real-second
            speed = current_audio_time / elapsed if elapsed > 0 else 0
            eta = (total_dur - current_audio_time) / speed if speed > 0 else 0
            ts_str = f"{utils.format_timestamp(current_audio_time)} / {utils.format_timestamp(total_dur)}"

            # Professional Format: Full Text + Timestamps
            if res:
                for item, src_text, tgt_text in zip(batch_items, batch_texts, res):
                    ts = utils.format_timestamp(item["start"])
                    log(f"  [{ts}] SRC: {src_text}")
                    log(f"  [{ts}] TGT: {tgt_text}")

            utils.print_progress_bar(
                current_audio_time, total_dur,
                prefix=prefix_str,
                timestamp_str=ts_str,
                speed=speed,
                eta=eta
            )

            # Aggressive Cleanup to prevent Shared RAM spillover
            # The driver might page out fragmentary memory if we don't clear it.
            if i + batch_size < total:  # Don't need to do it on the very last one
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            log(f"[Isolation] Batch Failed: {e}")
            raise e

    # 4. Save Output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)

    log("[Isolation] Success. Worker exiting.", level="DEBUG")


def _translate_batch_chunk(translator, chunk, src_code, tgt_code):
    """Refactored helper for inference."""
    try:
        return translator.translate(chunk, src_code, tgt_code)
    except Exception as batch_err:
        log(f"    [Batch Error] Chunk failed: {batch_err}", "ERROR")
        return ["Translation Error"] * len(chunk)


def _process_single_job(job, idx, total_jobs, translator):
    """Helper to process a single translation job within the batch."""
    # ... (Setup vars)
    lang = job["lang"]
    label = job.get("label", lang)
    tgt_code = job["tgt_code"]
    input_file = job["input"]
    output_file = job["output"]
    src_code = job.get("src_code")

    log(f"[Isolation] Job {idx + 1}/{total_jobs}: {label} ({tgt_code})")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reconstruct batch texts
        batch_texts = [item["text"] for item in data]
        batch_size = OPTIMIZER.config["nllb_batch"]
        translations = []

        total_dur = data[-1]["end"] if data else 0
        start_real = time.time()

        for i in range(0, len(batch_texts), batch_size):
            chunk = batch_texts[i: i + batch_size]

            # Helper call
            res = _translate_batch_chunk(translator, chunk, src_code, tgt_code)
            translations.extend(res)

            # Progress Logic
            current_idx = min(i + batch_size, len(data))
            current_audio_time = data[current_idx - 1]["end"]
            elapsed = time.time() - start_real
            speed = current_audio_time / elapsed if elapsed > 0 else 0
            eta = (total_dur - current_audio_time) / speed if speed > 0 else 0

            prefix_str = f"  [Translate {idx + 1}/{total_jobs}] {label} ({tgt_code})"
            utils.print_progress_bar(
                current_audio_time, total_dur,
                prefix=prefix_str,
                timestamp_str=f"{utils.format_timestamp(current_audio_time)} / {utils.format_timestamp(total_dur)}",
                speed=speed,
                eta=eta
            )

        # ... (Rest of function: Pad, Save, Sync)
        # Pad if mismatch
        if len(translations) != len(data):
            while len(translations) < len(data):
                translations.append("[Translation Error]")

        # Atomic Save
        temp_save_path = output_file + ".tmp"
        with open(temp_save_path, 'w', encoding='utf-8') as f:
            json.dump(translations, f, ensure_ascii=False)

        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(temp_save_path, output_file)

        # Sync Handshake
        wait_start = time.time()
        while os.path.exists(output_file):
            time.sleep(0.05)
            if time.time() - wait_start > 10:
                log(f"[Isolation] Warning: Parent timed out consuming {lang} output.", "WARNING")
                break

        # Cleanup Memory
        del translations
        del data
        del batch_texts

    except Exception as e:
        log(f"[Isolation] Job {lang} failed: {e}", "ERROR")
        return

    # Aggressive Cleanup
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_batch_translation_worker(manifest_path):
    """Executes multiple translation jobs with a single model load."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    jobs = manifest.get("jobs", [])
    if not jobs:
        log("[Isolation] No jobs in manifest. Exiting.")
        return

    # 1. Global Init (Load Model ONCE)
    log(f"[Isolation] Batch Mode: Processing {len(jobs)} targets...", level="INFO")
    config.load_config(OPTIMIZER, log)
    OPTIMIZER.detect_hardware(verbose=False)

    manager = ModelManager()
    translator = manager.get_nllb()

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
        sys.exit(1)

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

    run_translation_worker(
        input_file, output_file, src_lang, tgt_lang, batch_size, lang_label, prefix_str
    )


def main():
    try:
        utils.init_console()
        global torch
        import torch

        # Mode 1: Batch Mode (Manifest)
        if len(sys.argv) == 3 and sys.argv[1] == "--batch":
            manifest_path = sys.argv[2]
            run_batch_translation_worker(manifest_path)
            sys.exit(0)

        # Mode 2: Legacy Single Mode
        _run_legacy_mode()

    except Exception as e:
        log(f"[Isolation] FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
