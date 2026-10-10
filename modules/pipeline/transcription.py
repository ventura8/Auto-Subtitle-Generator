"""Transcription pipeline and vocal-separation orchestration."""

import gc
import math
import os
import sys
import time
from collections import namedtuple
from typing import Any

import modules.asr.pipeline as asr_pipeline
from modules import utils, workdir
from modules.asr import audio_source, language_id, routing
from modules.asr.cues import DEFAULT_LIMITS
from modules.asr.languages import fold_romanian_diacritics, normalize_iso639_1
from modules.configuration import asr_settings, config
from modules.media import ffmpeg_utils
from modules.models import OPTIMIZER, Segment, apply_dynamic_asr_batch, asr_gpu_shortfall, log_vram
from modules.runtime.optional_imports import load_optional_torch
from modules.safe_io import create_private_dir, discard_temp_path, promote_temp_path, reject_symlink, reserve_temp_path
from modules.utils import log

torch: Any | None = load_optional_torch()

# A resumed vocal track must match the extracted audio length; anything shorter was cut off mid-write.
VOCAL_DURATION_TOLERANCE_SECONDS = 2.0
# Chunked separation: context added on each side of a chunk so the model's edge behaviour never
# lands on a chunk seam, and the shortest tail that is worth a chunk of its own.
CHUNK_PAD_SECONDS = 2.0
CHUNK_MIN_TAIL_SECONDS = 60.0
CHUNK_DURATION_TOLERANCE_SECONDS = 0.5
# Below this the language guess is reported as unreliable (Whisper path and the ASR language vote alike).
LOW_LANGUAGE_CONFIDENCE = 0.4

# One NVIDIA-path transcription: the engine asked for, the audio it reads and the 16 kHz
# transcode it may need (both inside the work directory), and the model manager.
_AsrJob = namedtuple("_AsrJob", ["engine", "audio_path", "transcode_path", "model_mgr"])
# The routed NVIDIA run: the engine chosen, the spoken language, the PCM source and its speech spans.
_NvidiaRun = namedtuple("_NvidiaRun", ["engine", "language", "source", "spans"])


def _split_video_path(video_path):
    """Return ``(folder, base_name)`` for a video path."""
    folder = os.path.dirname(video_path) or "."
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    return folder, base_name


def _get_separated_vocal_path(video_path):
    """Return an existing, valid isolated-vocals track from the work directory, or None.

    Audio-Separator names its output ``{base_name}_temp_(Vocals)_<model>.wav``.
    A track left truncated by a crash mid-write is discarded so it is never
    resumed from.
    """
    folder, base_name = _split_video_path(video_path)
    work_dir = workdir.work_dir_path(folder, base_name)
    source_wav = os.path.join(work_dir, f"{base_name}_temp.wav")
    for entry in _list_vocal_candidates(work_dir, f"{base_name}_"):
        candidate = os.path.join(work_dir, entry)
        if _is_valid_vocal_track(candidate, source_wav):
            return candidate
        log(f"  [Sep] Discarding incomplete vocal track: {entry}", "WARNING")
        _discard_file(candidate)
    return None


def _list_vocal_candidates(work_dir, separator_prefix):
    """Return sorted separator vocal-stem names found in the work directory."""
    try:
        entries = sorted(os.listdir(work_dir))
    except OSError:
        return []
    return [entry for entry in entries if entry.startswith(separator_prefix) and "(Vocals)" in entry]


def _is_valid_vocal_track(vocal_path, source_wav_path):
    """Return True when ``vocal_path`` is a regular file whose duration matches the source audio."""
    if os.path.islink(vocal_path) or not os.path.isfile(vocal_path):
        return False
    vocal_duration = _probe_duration(vocal_path)
    if vocal_duration <= 0:
        return False
    return _matches_source_duration(vocal_duration, _probe_duration(source_wav_path))


def _matches_source_duration(vocal_duration, source_duration):
    """Return True when the vocal length matches the source, or when the source cannot be measured."""
    return source_duration <= 0 or abs(vocal_duration - source_duration) <= VOCAL_DURATION_TOLERANCE_SECONDS


def _probe_duration(path):
    """Return the media duration in seconds, or 0.0 when the file is missing or cannot be probed."""
    if not os.path.isfile(path):
        return 0.0
    try:
        return utils.get_audio_duration(path)
    except (OSError, ValueError):
        return 0.0


def _discard_file(path):
    """Best-effort removal of a temp file."""
    try:
        os.remove(path)
    except OSError:
        pass


def _process_separator_outputs(output_files, scratch_dir, target_dir):
    """Move the vocal stem from the private scratch directory into the work directory; ignore other stems."""
    vocal_file = None
    for output_file in output_files:
        file_name = os.path.basename(output_file)
        if "Vocals" not in file_name:
            continue

        src_path = _resolve_separator_output_path(output_file, scratch_dir)
        if src_path is None:
            log(f"  [Sep] Ignoring separator output outside the scratch directory: {file_name}", "WARNING")
            continue

        dst_path = os.path.join(target_dir, file_name)
        _move_separator_output(src_path, dst_path)
        vocal_file = dst_path
    return vocal_file


def _resolve_separator_output_path(output_file, scratch_dir):
    """Resolve a separator output to a path inside ``scratch_dir``, or None when it escapes.

    Audio-Separator reports its stems either by bare name or by full path. A
    bare name is resolved against the scratch directory it was given, never
    against the current working directory: preferring a same-named file in the
    CWD would make the pipeline move an unrelated file it does not own.
    Anything resolving outside the scratch directory is rejected, so a
    traversal or absolute path in the reported name can never cause a move or
    a delete elsewhere.
    """
    candidate = output_file if os.path.isabs(output_file) else os.path.join(scratch_dir, output_file)
    return candidate if _is_within_directory(candidate, scratch_dir) else None


def _is_within_directory(candidate, directory):
    """Return True when ``candidate`` resolves inside ``directory``."""
    root = os.path.realpath(directory)
    try:
        return os.path.commonpath([root, os.path.realpath(candidate)]) == root
    except ValueError:
        # Different drives on Windows, or a mix of absolute and relative parts.
        return False


def _move_separator_output(src_path, dst_path):
    """Move a separator output file into the work directory, replacing stale outputs.

    The destination name is predictable, so a planted symlink there is refused.
    """
    if not os.path.exists(src_path) or src_path == dst_path:
        return
    reject_symlink(dst_path)
    os.replace(src_path, dst_path)


def _detect_and_separate_vocals(video_path, model_mgr):
    """Handles vocal separation if configured."""
    if not config.USE_VOCAL_SEPARATION:
        return video_path

    existing_vocal = _get_separated_vocal_path(video_path)
    if existing_vocal:
        log(f"  [AI] Resuming with existing vocals: {os.path.basename(existing_vocal)}")
        return existing_vocal

    try:
        vocal_file = _run_vocal_separation(video_path, model_mgr)
        if vocal_file:
            return vocal_file

    except (ImportError, RuntimeError, OSError) as e:
        log(f"  [Sep] Warning: Separation failed ({e}). Using original audio.", "WARNING")

    return video_path


def _run_vocal_separation(video_path, model_mgr):
    """Execute separator model and return isolated vocals path when successful.

    Audio-Separator writes stems with predictable names, so it is pointed at a
    private scratch directory inside the work directory; only the finished
    vocal stem is moved out, and the scratch directory is removed afterwards
    (also on failure), so a crash mid-write never leaves a half-written track
    behind under the resumable name.
    """
    log("  [Task 0/4] Separating Vocals (BS-Roformer)...")
    audio_input_path = utils.extract_clean_audio(video_path)
    folder, base_name = _split_video_path(video_path)
    work_dir = workdir.ensure_work_dir(folder, base_name)
    scratch_dir = create_private_dir(work_dir, f"{base_name}_vocals")
    try:
        job = {
            "separator": model_mgr.get_separator(output_dir=scratch_dir),
            "audio_path": audio_input_path,
            "scratch_dir": scratch_dir,
            "work_dir": work_dir,
            "base_name": base_name,
        }
        vocal_file = _separate(job)
    finally:
        workdir.remove_scratch_dir(scratch_dir)
    if vocal_file and os.path.exists(vocal_file):
        log(f"  [Sep] Vocal track isolated: {os.path.basename(vocal_file)}")
        return vocal_file
    return None


def _separate(job):
    """Separate the whole file in one pass, or in resumable chunks when it is long.

    Audio-Separator loads the entire input into RAM at 44.1 kHz stereo float32
    and writes a stem of the same shape, so a multi-hour file would need many
    gigabytes of RAM and produce a stem past the 4 GB RIFF limit. Anything
    longer than the configured chunk is therefore cut into windows, each
    separated on its own with a little context on both sides, downmixed to the
    16 kHz mono Whisper consumes, and joined at the end.
    """
    chunk_seconds = _separation_chunk_seconds()
    duration = _probe_duration(job["audio_path"])
    if chunk_seconds and duration > chunk_seconds:
        return _run_chunked_separation(job, duration, chunk_seconds)
    output_files = job["separator"].separate(job["audio_path"])
    return _process_separator_outputs(output_files, job["scratch_dir"], job["work_dir"])


def _separation_chunk_seconds():
    """Return the configured chunk length in seconds, or 0 when chunking is disabled."""
    minutes = getattr(config, "SEPARATION_CHUNK_MINUTES", 0) or 0
    return max(0, int(minutes)) * 60


def _run_chunked_separation(job, duration, chunk_seconds):
    """Separate ``duration`` seconds of audio window by window and join the vocal stems."""
    windows = _chunk_windows(duration, chunk_seconds)
    log(f"  [Sep] Long audio ({utils.format_timestamp(duration)}): separating in {len(windows)} chunks of up to {chunk_seconds // 60} min.")
    stems = [_separate_chunk(job, index, len(windows), window) for index, window in enumerate(windows)]
    final_path = os.path.join(job["work_dir"], f"{job['base_name']}_temp_(Vocals)_chunked.wav")
    _join_chunk_stems(stems, final_path, job)
    for stem in stems:
        _discard_file(stem)
    return final_path


def _chunk_windows(duration, chunk_seconds):
    """Tile ``duration`` into ``(start, length)`` windows; a short tail is folded into the last one."""
    windows = []
    start = 0.0
    while start < duration:
        windows.append((start, min(chunk_seconds, duration - start)))
        start += chunk_seconds
    if len(windows) > 1 and windows[-1][1] < CHUNK_MIN_TAIL_SECONDS:
        tail = windows.pop()
        last_start, last_length = windows[-1]
        windows[-1] = (last_start, last_length + tail[1])
    return windows


def _chunk_stem_path(job, index):
    """Return the resumable path of one finished chunk stem inside the work directory."""
    return os.path.join(job["work_dir"], f"{job['base_name']}_sepchunk_{index:03d}.wav")


def _separate_chunk(job, index, total, window):
    """Return the 16 kHz mono vocal stem for one window, reusing one finished by an earlier run."""
    start, length = window
    stem_path = _chunk_stem_path(job, index)
    if _is_finished_chunk(stem_path, length):
        log(f"  [Sep] Chunk {index + 1}/{total}: resuming finished stem.")
        return stem_path
    log(f"  [Sep] Chunk {index + 1}/{total}: {utils.format_timestamp(start)} -> {utils.format_timestamp(start + length)}")
    pad_before = min(CHUNK_PAD_SECONDS, start)
    chunk_input = os.path.join(job["scratch_dir"], f"chunk_{index:03d}.wav")
    try:
        ffmpeg_utils.write_audio_window(job["audio_path"], chunk_input, start - pad_before, length + pad_before + CHUNK_PAD_SECONDS)
        raw_stem = _separate_chunk_input(job, chunk_input, index)
        _write_chunk_stem(raw_stem, stem_path, pad_before, length, job["work_dir"])
    finally:
        _discard_file(chunk_input)
    return stem_path


def _separate_chunk_input(job, chunk_input, index):
    """Run the separator on one padded chunk and return its raw vocal stem inside the scratch directory."""
    output_files = job["separator"].separate(chunk_input)
    raw_stem = _process_separator_outputs(output_files, job["scratch_dir"], job["scratch_dir"])
    if raw_stem is None or not os.path.isfile(raw_stem):
        raise RuntimeError(f"Separator produced no vocal stem for chunk {index + 1}")
    return raw_stem


def _write_chunk_stem(raw_stem, stem_path, pad_before, length, work_dir):
    """Trim the padding off a raw chunk stem and land it atomically as 16 kHz mono."""
    reservation = reserve_temp_path(stem_path, scratch_dir=work_dir)
    try:
        ffmpeg_utils.write_audio_window(raw_stem, reservation.path, pad_before, length, mono_16k=True)
        promote_temp_path(reservation, stem_path)
    except BaseException:
        discard_temp_path(reservation)
        raise
    finally:
        _discard_file(raw_stem)


def _is_finished_chunk(stem_path, length):
    """True when a chunk stem from an earlier run exists as a regular file of the expected length."""
    if os.path.islink(stem_path) or not os.path.isfile(stem_path):
        return False
    return abs(_probe_duration(stem_path) - length) <= CHUNK_DURATION_TOLERANCE_SECONDS


def _join_chunk_stems(stems, final_path, job):
    """Concatenate the chunk stems into the resumable vocal track, atomically."""
    list_path = os.path.join(job["scratch_dir"], "stems.list")
    reservation = reserve_temp_path(final_path, scratch_dir=job["work_dir"])
    try:
        ffmpeg_utils.concat_audio_files(stems, reservation.path, list_path)
        promote_temp_path(reservation, final_path)
    except BaseException:
        discard_temp_path(reservation)
        raise
    finally:
        _discard_file(list_path)


def _filter_hallucinations(segments, hallucination_phrases):
    """Internal helper to filter out hallucinated segments."""
    filtered_segments = []
    hallucinated_count = 0

    for segment in segments:
        text_clean = segment.text.strip().lower().strip(".,!?;: ")
        if _is_hallucinated_text(text_clean, hallucination_phrases):
            hallucinated_count += 1
            continue
        filtered_segments.append(segment)

    return filtered_segments, hallucinated_count


def _is_hallucinated_text(text_clean, hallucination_phrases):
    """Return True when segment text matches known short hallucination patterns."""
    for phrase in hallucination_phrases:
        if phrase in text_clean and len(text_clean) < len(phrase) + 5:
            return True
    return False


def _process_transcription_segments(segments_gen, total_dur, start_time):
    """Internal helper to process segments and update progress."""
    segments = []
    for segment in segments_gen:
        segments.append(segment)
        elapsed = time.time() - start_time
        # Calculate speed (audio seconds per real second)
        speed = segment.end / elapsed if elapsed > 0 else 0
        eta = (total_dur - segment.end) / speed if speed > 0 and total_dur > segment.end else 0

        _print_transcription_segment(segment)
        _print_transcription_progress(segment.end, total_dur, speed, eta)
    return segments


def _print_transcription_segment(segment):
    """Print one transcription segment with timestamp and confidence."""
    sys.stdout.write("\r\033[K")
    prob = math.exp(segment.avg_logprob) if hasattr(segment, "avg_logprob") else 1.0
    ts_start = utils.format_timestamp(segment.start)
    ts_end = utils.format_timestamp(segment.end)
    print(f"[{ts_start}->{ts_end}] ({prob:.0%}) {segment.text.strip()}")


def _print_transcription_progress(current_end, total_dur, speed, eta):
    """Print transcription progress bar with timing context."""
    utils.print_progress_bar(
        current_end,
        total_dur,
        prefix="  [Whisper] Transcribing",
        timestamp_str=f"{utils.format_timestamp(current_end)} / {utils.format_timestamp(total_dur)}",
        speed=speed,
        eta=eta,
    )


def _prepare_audio(video_path, model_mgr):
    """Prepares audio for transcription (Separation or Extraction)."""
    transcribe_path = video_path
    if config.USE_VOCAL_SEPARATION:
        transcribe_path = _detect_and_separate_vocals(video_path, model_mgr)
        model_mgr.offload_separator()

    if transcribe_path == video_path:
        transcribe_path = utils.extract_clean_audio(video_path)
    return transcribe_path


def _run_whisper_transcribe_call(whisper_model, path, vad_params, transcription_options):
    """Execute one Whisper transcribe call for a specific beam size."""
    return whisper_model.transcribe(
        path,
        beam_size=transcription_options["beam_size"],
        initial_prompt=transcription_options["prompt"],
        vad_filter=True,
        vad_parameters=vad_params,
        language=transcription_options["language"],
        condition_on_previous_text=True,
        no_speech_threshold=0.6,
    )


def _perform_transcription(whisper_model, path, vad_params, transcription_options, start_time):
    """Execute Whisper transcription and consume segments with OOM retry support."""
    initial_beam = int(transcription_options["beam_size"])
    try:
        first_pass_options = {
            **transcription_options,
            "beam_size": initial_beam,
        }
        segments_gen, info = _run_whisper_transcribe_call(whisper_model, path, vad_params, first_pass_options)
        segments = _process_transcription_segments(segments_gen, info.duration, start_time)
        return segments, info
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            reduced_beam = max(1, initial_beam // 2)
            if reduced_beam >= initial_beam:
                raise
            log("  [Whisper] OOM detected. Clearing cache and retrying...", "WARNING")
            if torch is not None:
                torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1)
            retry_options = {
                **transcription_options,
                "beam_size": reduced_beam,
            }
            segments_gen, info = _run_whisper_transcribe_call(whisper_model, path, vad_params, retry_options)
            segments = _process_transcription_segments(segments_gen, info.duration, start_time)
            return segments, info
        raise


def _log_transcription_config(lang_to_use, current_prompt):
    """Log the language and prompt configuration used for transcription."""
    if lang_to_use:
        log(f"  [Whisper] Config: Forced Language='{lang_to_use}'")
    else:
        log("  [Whisper] Config: Language Auto-Detection Enabled")

    if current_prompt:
        log("  [Whisper] Config: Input Prompt=Enabled")
    else:
        log("  [Whisper] Config: No Input Prompt")


def _finalize_transcription(segments, info, start_time):
    """Finalize Whisper output, filter segments, and free model resources."""
    elapsed = time.time() - start_time
    utils.print_progress_bar(
        info.duration,
        info.duration,
        prefix="  [Whisper] Transcribing",
        elapsed=elapsed,
        speed=info.duration / elapsed if elapsed > 0 else 1.0,
    )

    filtered_segments, hallucinated_count = _filter_hallucinations(
        segments,
        config.HALLUCINATION_PHRASES,
    )
    if hallucinated_count > 0:
        log(f"  [Whisper] Filtered {hallucinated_count} hallucinated segments.", "WARNING")

    detected_lang = info.language
    probability = info.language_probability
    log(f"  [Whisper] Detected Language: {detected_lang} (Conf: {probability:.2f})")
    if probability < LOW_LANGUAGE_CONFIDENCE:
        log(f"  [Warning] Low language confidence ({probability:.2f}).", "WARNING")

    filtered_segments = _fold_romanian(filtered_segments, detected_lang)
    filtered_segments.sort(key=lambda segment: segment.start)
    return filtered_segments, detected_lang


def _fold_romanian(segments, language):
    """Fold Romanian cedilla ş/ţ to comma-below ș/ț, so every engine spells Romanian the same way."""
    if language != "ro":
        return segments
    return [_with_folded_text(segment) for segment in segments]


def _with_folded_text(segment):
    """Return ``segment``, or a copy with folded text when folding changes it."""
    folded = fold_romanian_diacritics(segment.text)
    return segment if folded == segment.text else Segment(segment.start, segment.end, folded)


def transcribe_video_audio(video_path, model_mgr, forced_lang=None, forced_prompt=None):
    """Transcribe the video (or its vocal track) with the configured ASR engine.

    Returns ``(segments, language, transcribe_path)``. Whisper, the default,
    runs exactly as before; Canary and Parakeet (``--asr`` / ``asr.engine``)
    are routed by the spoken language and fall back to Whisper with a WARNING
    whenever they cannot handle the file.
    """
    # 1. Prepare Audio
    transcribe_path = _prepare_audio(video_path, model_mgr)

    # 2. Transcribe
    log(f"  [Task 1/4] Transcribing '{os.path.basename(transcribe_path)}'...")
    try:
        segments, detected_lang = _run_selected_engine(video_path, transcribe_path, model_mgr, (forced_lang, forced_prompt))
        return segments, detected_lang, transcribe_path

    except (RuntimeError, OSError, ValueError) as e:
        log(f"Transcription failed: {e}", "ERROR")
        raise
    finally:
        model_mgr.offload_whisper()
        model_mgr.offload_asr()


def _transcription_options(forced_lang, forced_prompt):
    """Whisper options: the CLI language and prompt win over config.yaml."""
    return {
        "prompt": forced_prompt if forced_prompt else config.INITIAL_PROMPT,
        "language": forced_lang if forced_lang else config.FORCED_LANGUAGE,
        "beam_size": int(OPTIMIZER.config["whisper_beam"]),
    }


def _whisper_vad_params():
    """Silero settings for Whisper's own VAD pass."""
    return {
        "threshold": 0.35,
        "min_silence_duration_ms": config.VAD_MIN_SILENCE_MS,
        "speech_pad_ms": 500,
    }


def _run_selected_engine(video_path, audio_path, model_mgr, forced):
    """Run the requested engine; plain Whisper keeps its original path (no decode, no language vote)."""
    options = _transcription_options(*forced)
    engine = asr_settings.active_engine()
    if engine == "whisper" and (options["language"] or not asr_settings.force_detected_language()):
        return _transcribe_with_whisper(model_mgr, audio_path, options)
    return _transcribe_routed(_asr_job(engine, video_path, audio_path, model_mgr), options)


def _transcribe_with_whisper(model_mgr, path, options):
    """Transcribe ``path`` with Faster-Whisper; returns ``(segments, detected_language)``."""
    _log_transcription_config(options["language"], options["prompt"])
    start_time = time.time()
    segments, info = _perform_transcription(model_mgr.get_whisper(), path, _whisper_vad_params(), options, start_time)
    return _finalize_transcription(segments, info, start_time)


def _asr_job(engine, video_path, audio_path, model_mgr):
    """Describe one routed transcription; the audio must come from the work directory."""
    folder, base_name = _split_video_path(video_path)
    work_dir = workdir.work_dir_path(folder, base_name)
    if not _is_within_directory(audio_path, work_dir):
        raise ValueError(f"ASR audio must come from the work directory, not {audio_path}")
    return _AsrJob(engine, audio_path, os.path.join(work_dir, f"{base_name}_asr16k.wav"), model_mgr)


def _transcribe_routed(job, options):
    """Settle the language and engine, then transcribe; Whisper takes over whenever the NVIDIA path cannot."""
    language = _forced_language(options["language"])
    if language and _announce_route(job.engine, language).engine == "whisper":
        return _transcribe_with_whisper(job.model_mgr, job.audio_path, {**options, "language": language})
    source = audio_source.open_pcm_source(job.audio_path, job.transcode_path)
    try:
        segments, language = _transcribe_source(job, source, language)
    finally:
        source.close()
    if segments is not None:
        return segments, language
    # Whisper reuses the model the language vote loaded; an NVIDIA model must not share the GPU with it.
    job.model_mgr.offload_asr()
    return _transcribe_with_whisper(job.model_mgr, job.audio_path, {**options, "language": language})


def _forced_language(raw_language):
    """Normalise ``--lang`` / ``whisper.language`` to ISO 639-1; an unusable value is voted on instead."""
    if not raw_language:
        return None
    language = normalize_iso639_1(raw_language)
    if language is None:
        log(f"  [ASR] Forced language '{raw_language}' is not a two-letter ISO 639-1 code; detecting the language instead.", "WARNING")
    return language


def _announce_route(requested, language):
    """Resolve the engine for ``language``, logging why an NVIDIA engine fell back to Whisper."""
    route = routing.resolve(requested, language, asr_settings.routes())
    if route.reason:
        log(f"  [ASR] Using Whisper instead of {requested}: {route.reason}.", "WARNING")
    return route


def _transcribe_source(job, source, language):
    """Find speech, vote the language when unknown and run the routed engine; ``None`` segments mean Whisper."""
    vad = audio_source.asr_vad_settings(asr_settings.max_segment_seconds(), config.VAD_MIN_SILENCE_MS)
    spans = audio_source.iter_speech_spans(source, vad)
    language = language or _vote_language(job.model_mgr, source, spans)
    route = _announce_route(job.engine, language)
    if route.engine == "whisper":
        return None, language
    return _transcribe_with_nvidia(job, _NvidiaRun(route.engine, language, source, spans)), language


def _vote_language(model_mgr, source, spans):
    """Vote the spoken language with Whisper over windows spread across the speech; None without speech."""
    if not spans:
        log("  [ASR] No speech found; leaving the language to Whisper.")
        return None
    if not str(config.WHISPER_MODEL_SIZE).startswith("large"):
        log(f"  [ASR] Language ID runs on Whisper '{config.WHISPER_MODEL_SIZE}'; large-v3 tells related languages apart better.", "WARNING")
    log_vram("before language ID")
    return _announce_vote(language_id.detect_language(model_mgr.get_whisper(), source, spans))


def _announce_vote(vote):
    """Log the voted language and return its code, or None when no window could be scored."""
    if vote is None:
        log("  [ASR] Language ID could not score the audio; leaving the language to Whisper.", "WARNING")
        return None
    log(f"  [ASR] Detected Language: {vote.language} (Conf: {vote.probability:.2f}, top in {vote.share:.0%} of windows)")
    if vote.probability < LOW_LANGUAGE_CONFIDENCE:
        log(f"  [Warning] Low language confidence ({vote.probability:.2f}).", "WARNING")
    return vote.language


def _transcribe_with_nvidia(job, run):
    """Transcribe with Canary or Parakeet; ``None`` hands the file back to Whisper after a WARNING."""
    job.model_mgr.offload_whisper()
    log_vram("after language-ID offload")
    model = _load_nvidia_model(job, run.engine)
    if model is None:
        return None
    try:
        return _decode_with_model(model, run)
    except asr_pipeline.AsrEngineFailed as error:
        log(f"  [ASR] {_label(run.engine)} failed on this file ({error}); redoing it with Whisper in '{run.language}'.", "WARNING")
        return None


def _load_nvidia_model(job, engine):
    """Load the engine's model, or return None (WARNING logged) so Whisper is used instead."""
    # Under auto, Whisper on the GPU beats Canary pushed onto the CPU by a full card.
    shortfall = asr_gpu_shortfall(engine) if job.engine == "auto" else None
    if shortfall:
        log(f"  [ASR] Using Whisper instead of {_label(engine)}: {shortfall}.", "WARNING")
        return None
    try:
        return job.model_mgr.get_asr(engine)
    except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as error:
        log(f"  [ASR] Could not load {_label(engine)} ({error}); falling back to Whisper.", "WARNING")
        return None


def _decode_with_model(model, run):
    """Decode every speech span with the loaded model and return filtered, sorted segments."""
    batch = apply_dynamic_asr_batch(model, asr_settings.max_segment_seconds())
    log(f"  [{_label(run.engine)}] {len(run.spans)} speech spans in '{run.language}' on {model.device} (batch {batch}).")
    request = asr_pipeline.AsrRequest(run.language, batch, DEFAULT_LIMITS)
    result = asr_pipeline.transcribe(model, run.source, run.spans, request)
    asr_pipeline.check_decoded_speech(result)
    return _finish_nvidia_segments(result, _label(run.engine))


def _finish_nvidia_segments(result, label):
    """Turn cues into segments, apply the known-phrase filter and log one summary of everything dropped."""
    segments = [Segment(start, end, text) for start, end, text in result.cues]
    filtered, hallucinated = _filter_hallucinations(segments, config.HALLUCINATION_PHRASES)
    _log_dropped(label, {**result.dropped, "known phrase": hallucinated})
    log(f"  [{label}] {len(filtered)} cues from {result.decoded_seconds:.0f} s of {result.speech_seconds:.0f} s of speech.")
    filtered.sort(key=lambda segment: segment.start)
    return filtered


def _log_dropped(label, counts):
    """Log one WARNING summarising the spans and cues each filter removed."""
    summary = ", ".join(f"{count} {name}" for name, count in sorted(counts.items()) if count)
    if summary:
        log(f"  [{label}] Filtered: {summary}.", "WARNING")


def _label(engine):
    """Display name of an engine for log lines."""
    return engine.capitalize()
