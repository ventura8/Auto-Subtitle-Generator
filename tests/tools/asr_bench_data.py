"""Benchmark inputs for ``tests.tools.asr_benchmark``: FLEURS, VoxPopuli and synthetic audio.

Datasets come through ``hf_hub_download`` at pinned revisions into the Hugging
Face cache. FLEURS audio is read member by member from its tar archive into
memory (``extractfile``, never ``extractall``, so nothing from the archive
touches the disk). VoxPopuli is read from its parquet shard with pyarrow, which
lives in the optional ``bench`` Poetry group. Every synthetic WAV is written
under the benchmark's ``--out`` directory.
"""

import importlib
import io
import os
import tarfile
from collections import namedtuple
from typing import Any

SAMPLE_RATE = 16000

FLEURS_REPO = "google/fleurs"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
VOXPOPULI_REPO = "facebook/voxpopuli"
VOXPOPULI_REVISION = "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"
VOXPOPULI_COLUMNS = ("audio_id", "speaker_id", "raw_text", "audio")
PARQUET_BATCH_ROWS = 64

PROBE_SECONDS = 60.0
PROBE_SEED = 1234
NOISE_RMS = 0.05
TONE_HZ = 440.0
TONE_AMPLITUDE = 0.3

LONGFORM_SEED = 4321
GAP_SECONDS = (0.5, 3.0)
GAP_NOISE_RMS = 0.003
CODE_SWITCH_GAP_SECONDS = 0.5

# ``cluster`` is the FLEURS sentence id (several speakers read each sentence) or the VoxPopuli speaker.
Utterance = namedtuple("Utterance", ["uid", "cluster", "text", "audio"])
Corpus = namedtuple("Corpus", ["name", "language", "utterances"])
# ``truth`` holds (start, end, text) of every joined utterance, in seconds of the long file.
LongForm = namedtuple("LongForm", ["path", "truth", "seconds"])


def _module(name: str) -> Any:
    """Import an optional dependency only when the benchmark needs it."""
    return importlib.import_module(name)


def numpy() -> Any:
    """Return the lazily imported numpy module."""
    return _module("numpy")


def _hub_file(repo: str, filename: str, revision: str) -> str:
    """Download (or reuse from the HF cache) one dataset file at a pinned revision."""
    return _module("huggingface_hub").hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


def iso_code(lang_dir: str) -> str:
    """ISO 639-1 code of a FLEURS directory name (``ro_ro`` -> ``ro``)."""
    return lang_dir.split("_")[0].lower()


def _read_fleurs_tsv(path: str) -> list[list[str]]:
    """FLEURS rows: sentence id, file name, raw transcription, normalised text, ... (no header)."""
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n").split("\t") for line in handle if line.strip()]


def decode_wav(data: bytes) -> Any:
    """Decode in-memory WAV bytes to 16 kHz mono float32."""
    samples, rate = _module("soundfile").read(io.BytesIO(data), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {rate} Hz")
    return samples.mean(axis=1).astype("float32")


def _read_member(archive: Any, member: Any, wanted: set[str], found: dict[str, Any]) -> None:
    """Decode one wanted regular-file member straight from the archive stream."""
    name = os.path.basename(member.name)
    handle = archive.extractfile(member) if member.isfile() and name in wanted else None
    if handle is not None:
        found[name] = decode_wav(handle.read())


def _tar_audio(tar_path: str, wanted: set[str]) -> dict[str, Any]:
    """Read the wanted WAV members of a FLEURS archive into memory."""
    found: dict[str, Any] = {}
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive:
            _read_member(archive, member, wanted, found)
            if len(found) == len(wanted):
                break
    return found


def load_fleurs(lang_dir: str, limit: int | None) -> Corpus:
    """The first ``limit`` FLEURS test utterances of one language (all of them when ``limit`` is None)."""
    rows = _read_fleurs_tsv(_hub_file(FLEURS_REPO, f"data/{lang_dir}/test.tsv", FLEURS_REVISION))[:limit]
    audio = _tar_audio(_hub_file(FLEURS_REPO, f"data/{lang_dir}/audio/test.tar.gz", FLEURS_REVISION), {row[1] for row in rows})
    utterances = [Utterance(row[1], row[0], row[2], audio[row[1]]) for row in rows if row[1] in audio]
    return Corpus(f"fleurs/{lang_dir}", iso_code(lang_dir), utterances)


def _has_text(row: dict) -> bool:
    """VoxPopuli has rows without a transcript; they cannot be scored."""
    return bool(str(row.get("raw_text") or "").strip())


def _parquet_rows(path: str, limit: int | None) -> list[dict]:
    """Read transcribed rows batch by batch so ``limit`` bounds memory as well as work."""
    parquet = _module("pyarrow.parquet").ParquetFile(path)
    rows: list[dict] = []
    for batch in parquet.iter_batches(batch_size=PARQUET_BATCH_ROWS, columns=list(VOXPOPULI_COLUMNS)):
        rows.extend(filter(_has_text, batch.to_pylist()))
        if limit is not None and len(rows) >= limit:
            break
    return rows[:limit]


def load_voxpopuli(lang: str, limit: int | None) -> Corpus:
    """VoxPopuli test utterances of one language; speakers are the bootstrap clusters."""
    path = _hub_file(VOXPOPULI_REPO, f"{lang}/test-00000-of-00001.parquet", VOXPOPULI_REVISION)
    rows = _parquet_rows(path, limit)
    utterances = [Utterance(row["audio_id"], row["speaker_id"], row["raw_text"], decode_wav(row["audio"]["bytes"])) for row in rows]
    return Corpus(f"voxpopuli/{lang}", lang, utterances)


def take_seconds(utterances: list, seconds: float) -> list:
    """Leading utterances until their audio adds up to ``seconds``."""
    taken, total = [], 0.0
    for utterance in utterances:
        if total >= seconds:
            break
        taken.append(utterance)
        total += len(utterance.audio) / SAMPLE_RATE
    return taken


def one_per_sentence(utterances: list) -> list:
    """Keep the first reading of each FLEURS sentence, so joined audio never repeats a sentence back to back."""
    seen: set[str] = set()
    kept = []
    for utterance in utterances:
        if utterance.cluster not in seen:
            seen.add(utterance.cluster)
            kept.append(utterance)
    return kept


def write_wav(out_dir: str, name: str, audio: Any) -> str:
    """Write a 16 kHz mono PCM_16 WAV (the format of the pipeline's extracted audio) under ``out_dir/audio``."""
    folder = os.path.join(out_dir, "audio")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.wav")
    _module("soundfile").write(path, numpy().clip(audio, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")
    return path


def transcode_path(path: str) -> str:
    """Where ``open_pcm_source`` may put a transcode; never used for the 16 kHz mono files written here."""
    return f"{os.path.splitext(path)[0]}_asr16k.wav"


def noise(seconds: float, seed: int = 0) -> Any:
    """White noise at the probe level."""
    rng = numpy().random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SAMPLE_RATE)) * NOISE_RMS).astype("float32")


def _scaled(samples: Any) -> Any:
    """Scale noise to a fixed RMS so every probe is equally loud."""
    np = numpy()
    return (samples * (NOISE_RMS / np.sqrt(np.mean(np.square(samples))))).astype("float32")


def _silence(_rng: Any, count: int, _speech: list) -> Any:
    """Digital silence."""
    return numpy().zeros(count, dtype="float32")


def _white(rng: Any, count: int, _speech: list) -> Any:
    """White noise."""
    return _scaled(rng.standard_normal(count))


def _pink(rng: Any, count: int, _speech: list) -> Any:
    """Pink (1/f power) noise shaped in the frequency domain."""
    np = numpy()
    spectrum = np.fft.rfft(rng.standard_normal(count))
    frequencies = np.arange(len(spectrum), dtype="float64")
    frequencies[0] = 1.0
    return _scaled(np.fft.irfft(spectrum / np.sqrt(frequencies), count))


def _brown(rng: Any, count: int, _speech: list) -> Any:
    """Brown (random-walk) noise with its mean removed."""
    walk = numpy().cumsum(rng.standard_normal(count))
    return _scaled(walk - walk.mean())


def _tone(_rng: Any, count: int, _speech: list) -> Any:
    """A steady 440 Hz sine."""
    np = numpy()
    return (TONE_AMPLITUDE * np.sin(2 * np.pi * TONE_HZ * np.arange(count) / SAMPLE_RATE)).astype("float32")


def _reversed(_rng: Any, count: int, speech: list) -> Any:
    """Time-reversed speech: a speech-like spectrum without words."""
    np = numpy()
    joined = np.concatenate([utterance.audio for utterance in take_seconds(speech, PROBE_SECONDS)])
    return np.resize(joined[::-1], count).astype("float32")


PROBE_GENERATORS = {"silence": _silence, "white": _white, "pink": _pink, "brown": _brown, "tone": _tone, "reversed": _reversed}


def probe_audio(kind: str, speech: list) -> Any:
    """``PROBE_SECONDS`` of one non-speech probe; ``speech`` feeds the reversed-speech probe."""
    rng = numpy().random.default_rng(PROBE_SEED)
    return PROBE_GENERATORS[kind](rng, int(PROBE_SECONDS * SAMPLE_RATE), speech)


def _gap(rng: Any) -> Any:
    """0.5-3 s of silence or faint noise between joined utterances."""
    count = int(rng.uniform(*GAP_SECONDS) * SAMPLE_RATE)
    gap = rng.standard_normal(count) * GAP_NOISE_RMS if rng.random() < 0.5 else numpy().zeros(count)
    return gap.astype("float32")


def build_longform(utterances: list, out_dir: str, name: str) -> LongForm:
    """Join utterances with random gaps into one WAV whose utterance boundaries are known."""
    np = numpy()
    rng = np.random.default_rng(LONGFORM_SEED)
    pieces: list[Any] = []
    truth: list[tuple[float, float, str]] = []
    cursor = 0.0
    for utterance in utterances:
        gap = _gap(rng)
        start = cursor + len(gap) / SAMPLE_RATE
        cursor = start + len(utterance.audio) / SAMPLE_RATE
        pieces.extend((gap, utterance.audio))
        truth.append((start, cursor, utterance.text))
    audio = np.concatenate([*pieces, _gap(rng)])
    return LongForm(write_wav(out_dir, name, audio), truth, len(audio) / SAMPLE_RATE)


def code_switch_audio(english: list, romanian: list) -> tuple[Any, float]:
    """English utterances followed by Romanian ones, 0.5 s apart; also the length of the English intro."""
    np = numpy()
    gap = np.zeros(int(CODE_SWITCH_GAP_SECONDS * SAMPLE_RATE), dtype="float32")
    pieces = [piece for utterance in english + romanian for piece in (utterance.audio, gap)]
    english_seconds = sum(len(utterance.audio) + len(gap) for utterance in english) / SAMPLE_RATE
    return np.concatenate(pieces), english_seconds
