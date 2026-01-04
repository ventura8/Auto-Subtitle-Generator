
import os
import sys
import time
import math
from modules import config, utils
from modules.utils import log
from modules.models import OPTIMIZER


def _get_separated_vocal_path(video_path):
    """Internal helper to determine vocal separation output path."""
    target_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # Audio-Separator naming: {base_name}_(Vocals)_...
    try:
        for f in os.listdir(target_dir):
            if f.startswith(base_name) and "(Vocals)" in f:
                return os.path.join(target_dir, f)
    except OSError:
        pass
    return None


def _process_separator_outputs(output_files, target_dir):
    """Handles renaming and moving separator output files."""
    vocal_file = None
    for f in output_files:
        src_path = os.path.abspath(f)
        base = os.path.basename(f)

        # Rename Instrumental -> Background
        if "(Instrumental)" in base:
            base = base.replace("(Instrumental)", "(Background)")

        dst_path = os.path.join(target_dir, base)
        if os.path.exists(src_path) and src_path != dst_path:
            if os.path.exists(dst_path):
                os.remove(dst_path)
            os.rename(src_path, dst_path)

        if "Vocals" in base:
            vocal_file = dst_path
    return vocal_file


def _detect_and_separate_vocals(video_path, model_mgr):
    """Handles vocal separation if configured."""
    if not config.USE_VOCAL_SEPARATION:
        return video_path
    try:
        # 1. Check for existing output
        existing_vocal = _get_separated_vocal_path(video_path)
        if existing_vocal:
            log(f"  [AI] Resuming with existing vocals: {os.path.basename(existing_vocal)}")
            return existing_vocal

        # 2. Run separation
        log("  [Task 0/4] Separating Vocals (BS-Roformer)...")
        audio_input_path = utils.extract_clean_audio(video_path)

        separator = model_mgr.get_separator()
        output_files = separator.separate(audio_input_path)

        vocal_file = _process_separator_outputs(output_files, os.path.dirname(video_path))

        if vocal_file and os.path.exists(vocal_file):
            log(f"  [Sep] Vocal track isolated: {os.path.basename(vocal_file)}")
            return vocal_file

    except Exception as e:
        log(f"  [Sep] Warning: Separation failed ({e}). Using original audio.", "WARNING")

    return video_path


def _filter_hallucinations(segments, hallucination_phrases):
    """Internal helper to filter out hallucinated segments."""
    filtered_segments = []
    hallucinated_count = 0

    for s in segments:
        text_clean = s.text.strip().lower().strip(".,!?;: ")
        is_hallucination = False
        for phrase in hallucination_phrases:
            if phrase in text_clean and len(text_clean) < len(phrase) + 5:
                is_hallucination = True
                break

        if is_hallucination:
            hallucinated_count += 1
            continue
        filtered_segments.append(s)

    return filtered_segments, hallucinated_count


def _process_transcription_segments(segments_gen, total_dur, start_time):
    """Internal helper to process segments and update progress."""
    segments = []
    for segment in segments_gen:
        segments.append(segment)
        elapsed = time.time() - start_time
        # Calculate speed (audio seconds per real second)
        speed = segment.end / elapsed if elapsed > 0 else 0
        eta = (total_dur - segment.end) / speed if speed > 0 and total_dur > segment.end else 0

        # Print segment text (Verbose UI)
        sys.stdout.write("\r\033[K")

        # fast-whisper segments usually have 'avg_logprob'
        # standard openai-whisper segments also have 'avg_logprob'
        prob = math.exp(segment.avg_logprob) if hasattr(segment, 'avg_logprob') else 1.0

        ts_start = utils.format_timestamp(segment.start)
        ts_end = utils.format_timestamp(segment.end)
        print(f"[{ts_start}->{ts_end}] ({prob:.0%}) {segment.text.strip()}")

        utils.print_progress_bar(
            segment.end, total_dur,
            prefix="  [Whisper] Transcribing",
            timestamp_str=f"{utils.format_timestamp(segment.end)} / {utils.format_timestamp(total_dur)}",
            speed=speed,
            eta=eta
        )
    return segments


def transcribe_video_audio(video_path, model_mgr, forced_lang=None, forced_prompt=None):
    """Runs Whisper transcription on the video (or vocal track)."""

    # 1. Prepare Audio
# ... (rest of function until segment gen)
    # Check for existing Vocals first if separation enabled
    transcribe_path = video_path
    if config.USE_VOCAL_SEPARATION:
        transcribe_path = _detect_and_separate_vocals(video_path, model_mgr)
        # Offload separator immediately after use
        model_mgr.offload_separator()

    if transcribe_path == video_path:
        # No separation or failed, extract standard audio
        transcribe_path = utils.extract_clean_audio(video_path)

    # 2. Transcribe
    log(f"  [Task 1/4] Transcribing '{os.path.basename(transcribe_path)}'...")
    whisper_model = model_mgr.get_whisper()

    current_prompt = forced_prompt if forced_prompt else config.INITIAL_PROMPT

    # Language handling
    lang_to_use = forced_lang if forced_lang else config.FORCED_LANGUAGE

    # Log configuration details as requested
    if lang_to_use:
        log(f"  [Whisper] Config: Forced Language='{lang_to_use}'")
    else:
        log("  [Whisper] Config: Language Auto-Detection Enabled")

    if current_prompt:
        log(f"  [Whisper] Config: Input Prompt='{current_prompt}'")
    else:
        log("  [Whisper] Config: No Input Prompt")

    start_time = time.time()

    try:
        # Tuned VAD Parameters for Sequential Mode
        # Ensures correct Language ID without cutting off start
        vad_params = dict(
            threshold=0.35,              # More sensitive (default 0.5)
            min_silence_duration_ms=500,  # Matches config
            speech_pad_ms=500            # Add padding to capture breath/starts
        )

        try:
            segments_gen, info = whisper_model.transcribe(
                transcribe_path,
                beam_size=OPTIMIZER.config["whisper_beam"],
                initial_prompt=current_prompt,
                vad_filter=True,             # ENABLED: Needed for Lang ID accuracy
                vad_parameters=vad_params,   # TUNED: Prevents start cut-off
                language=lang_to_use,
                condition_on_previous_text=True,
                no_speech_threshold=0.6
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log("  [Whisper] OOM detected. Clearing cache and retrying...", "WARNING")
                import torch
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                time.sleep(1)
                segments_gen, info = whisper_model.transcribe(
                    transcribe_path,
                    beam_size=max(1, OPTIMIZER.config["whisper_beam"] // 2),
                    initial_prompt=current_prompt,
                    vad_filter=True,
                    vad_parameters=vad_params,
                    language=lang_to_use,
                    condition_on_previous_text=True,
                    no_speech_threshold=0.6
                )
            else:
                raise e

        total_dur = info.duration
        log(f"  [Whisper] Detected Language: {info.language} (Probability: {info.language_probability:.2%})")

        # Extracted loop call
        segments = _process_transcription_segments(segments_gen, total_dur, start_time)

        # Ensure final state
        elapsed = time.time() - start_time
        utils.print_progress_bar(
            total_dur, total_dur,
            prefix="  [Whisper] Transcribing",
            elapsed=elapsed,
            speed=total_dur / elapsed if elapsed > 0 else 1.0
        )

        # --- HALLUCINATION FILTERING ---
        segments, hallucinated_count = _filter_hallucinations(segments, config.HALLUCINATION_PHRASES)

        if hallucinated_count > 0:
            log(f"  [Whisper] Filtered {hallucinated_count} hallucinated segments.", "WARNING")

        detected_lang = info.language
        prob = info.language_probability

        log(f"  [Whisper] Detected Language: {detected_lang} (Conf: {prob:.2f})")

        if prob < 0.4:
            log(f"  [Warning] Low language confidence ({prob:.2f}).", "WARNING")

        # Offload Whisper to free VRAM for NLLB
        model_mgr.offload_whisper()

        # CRITICAL: Sort segments to prevent out-of-order SRT corruption
        segments.sort(key=lambda s: s.start)

        return segments, detected_lang, transcribe_path

    except Exception as e:
        log(f"Transcription failed: {e}", "ERROR")
        raise e
