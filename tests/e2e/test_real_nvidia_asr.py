"""Real-library checks of the NVIDIA ASR engines (transformers Canary and Parakeet).

conftest.py replaces torch, transformers and faster_whisper with mocks inside the
pytest process, so every check here runs a fresh interpreter on the CPU
(``CUDA_VISIBLE_DEVICES=-1``):

- without any download: a tiny random-weight ``ParakeetForTDT`` proves that
  ``modules.asr.tdt_guard`` restores the per-frame symbol cap transformers skips
  (#49388), that the frame mask removes what padded batch rows emit past their
  own audio, and that a masked batched row equals the row decoded alone;
- with ``ASG_E2E_NVIDIA_ASR=1``: the pinned real checkpoints load through the
  production wrappers and transcribe a padded batch of noise clips.
"""

import os
import subprocess
import sys
import unittest
from importlib.machinery import PathFinder

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REAL_CHECKPOINTS_ENV = "ASG_E2E_NVIDIA_ASR"
TINY_MODEL_TIMEOUT_SECONDS = 600
# The first run downloads about 6.4 GB of safetensors and decodes on the CPU.
CHECKPOINT_TIMEOUT_SECONDS = 3600


def _installed(name):
    """True when ``name`` is importable from disk; ignores the conftest mocks in ``sys.modules``."""
    return PathFinder.find_spec(name) is not None


ML_INSTALLED = _installed("torch") and _installed("transformers")

TINY_TDT_SCRIPT = """
import torch
import transformers

from modules.asr import common, tdt_guard

VOCAB, BLANK, TOKEN, MAX_SYMBOLS = 20, 19, 5, 3


def build_model(duration_index):
    torch.manual_seed(0)
    config = transformers.ParakeetTDTConfig(
        vocab_size=VOCAB, blank_token_id=BLANK, decoder_hidden_size=32, num_decoder_layers=1, max_symbols_per_step=MAX_SYMBOLS,
        encoder_config={"hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 64,
                        "subsampling_conv_channels": 16, "num_mel_bins": 16},
    )
    model = transformers.ParakeetForTDT(config).eval()
    model.generation_config.decoder_start_token_id = BLANK
    with torch.no_grad():  # the joint always emits TOKEN with the given duration index
        model.joint.head.weight.zero_()
        model.joint.head.bias.fill_(-10.0)
        model.joint.head.bias[TOKEN] = 10.0
        model.joint.head.bias[VOCAB + duration_index] = 10.0
    common.flatten_lstm_weights(model)
    return model


def features(lengths):
    torch.manual_seed(1)
    feats = torch.randn(len(lengths), max(lengths), 16)
    mask = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
        feats[row, length:] = 0.0
    return feats, mask


def valid_frames(model, mask):
    return getattr(model, "_get_subsampling_output_length")(mask.sum(-1))


def run(model, feats, mask, max_new_tokens):
    processors = transformers.LogitsProcessorList([tdt_guard.MaskDurationLogits(VOCAB)])
    with torch.inference_mode():
        return model.generate(input_features=feats, attention_mask=mask, max_new_tokens=max_new_tokens, logits_processor=processors)


feats, mask = features([400])
stock = build_model(0)
frames = int(valid_frames(stock, mask)[0])
out = run(stock, feats, mask, 600)
assert out.sequences.shape[1] - 1 == 600 and int(out.durations.sum()) == 0, "stock TDT generate no longer loops; recheck #49388"
assert tdt_guard.loop_detected(out.durations[0], out.sequences[0], BLANK)

patched = build_model(0)
assert tdt_guard.install_symbol_cap(patched) is True, "install_symbol_cap no longer recognises transformers' TDT mixin"
budget = common.parakeet_step_budget(frames, patched.max_symbols_per_step)
out = run(patched, feats, mask, budget)
assert int(out.durations.sum()) >= frames and out.sequences.shape[1] - 1 <= budget, "capped decode did not walk every frame"
assert int((out.sequences[0, 1:] == TOKEN).sum()) <= MAX_SYMBOLS * frames
assert not tdt_guard.loop_detected(out.durations[0], out.sequences[0], BLANK)
assert torch.equal(run(patched, feats, mask, budget).sequences, out.sequences), "per-generate state leaked between calls"
assert "_update_model_kwargs_for_generation" not in vars(type(patched)), "the patch must stay on the instance"

model = build_model(1)  # one token per frame, never blank
tdt_guard.install_symbol_cap(model)
feats, mask = features([400, 160])
valid = valid_frames(model, mask)
out = run(model, feats, mask, common.parakeet_step_budget(int(valid.max()), model.max_symbols_per_step))
seqs, durs = out.sequences.cpu(), out.durations.cpu()
masked = tdt_guard.mask_frames_past_end(seqs, durs, valid, BLANK)
assert (masked == TOKEN).sum(-1).tolist() == valid.tolist(), "frame mask left tokens past a row's valid frames"
assert torch.equal(masked[0], seqs[0])
alone = run(model, feats[1:2, :160], mask[1:2, :160], common.parakeet_step_budget(int(valid[1]), model.max_symbols_per_step))
assert int((alone.sequences == TOKEN).sum()) == int((masked[1] == TOKEN).sum()), "masked batched row differs from the row alone"
print("TDT_GUARD_OK")
"""

LANGUAGE_TABLE_SCRIPT = """
from transformers.models.canary import processing_canary

from modules.asr.languages import NVIDIA_EU25

assert set(processing_canary.LANGUAGE_CODE_TO_NAME) == set(NVIDIA_EU25), sorted(set(processing_canary.LANGUAGE_CODE_TO_NAME) ^ NVIDIA_EU25)
print("LANGUAGE_TABLE_OK")
"""

CHECKPOINT_SCRIPT = """
import sys

import numpy as np

from modules.asr.canary import CanaryModel
from modules.asr.common import release_cuda_cache
from modules.asr.parakeet import ParakeetModel
from modules.configuration import asr_settings

engine = sys.argv[1]
wrapper = {"canary": CanaryModel, "parakeet": ParakeetModel}[engine]
model = wrapper(asr_settings.model_id(engine), asr_settings.model_revision(engine))
assert model.device == "cpu", model.device
noise = (np.random.default_rng(0).standard_normal(48000) * 0.05).astype(np.float32)
clips = model.transcribe_batch([noise, noise[:32000]], "en")  # 3 s and 2 s: a padded batch
assert len(clips) == 2, clips
for clip in clips:
    assert isinstance(clip.text, str) and isinstance(clip.degenerate, bool), clip
if engine == "parakeet":
    assert model.symbol_cap_installed is True, "the TDT symbol cap was not installed on the real checkpoint"
    assert all(stamp["start"] <= stamp["end"] for clip in clips for stamp in clip.stamps), clips
else:
    assert all(isinstance(clip.logprob, float) and clip.stamps is None for clip in clips), clips
model.release()
release_cuda_cache()
print("NVIDIA_ASR_SMOKE_OK", engine, [clip.text for clip in clips])
"""


def _run_script(code, arguments, timeout, strict_warnings):
    """Run ``code`` in a fresh CPU-only interpreter from the repository root."""
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    warning_flags = ["-W", "error"] if strict_warnings else []
    command = [sys.executable, *warning_flags, "-c", code, *arguments]
    try:
        return subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        raise AssertionError(f"NVIDIA ASR check timed out after {timeout} seconds") from error


@unittest.skipUnless(ML_INSTALLED, "torch and transformers are not installed (poetry install --with ml)")
class TestTdtGuardOnTinyModel(unittest.TestCase):
    """The Parakeet decode guards against the installed transformers, without a download."""

    def test_symbol_cap_and_frame_mask(self):
        # Warnings are errors here, like pytest's filterwarnings=error: a transformers change shows up at once.
        result = _run_script(TINY_TDT_SCRIPT, [], TINY_MODEL_TIMEOUT_SECONDS, strict_warnings=True)
        self.assertEqual(result.returncode, 0, f"Stderr: {result.stderr}")
        self.assertIn("TDT_GUARD_OK", result.stdout)

    def test_language_table_matches_canary_processor(self):
        result = _run_script(LANGUAGE_TABLE_SCRIPT, [], TINY_MODEL_TIMEOUT_SECONDS, strict_warnings=True)
        self.assertEqual(result.returncode, 0, f"Stderr: {result.stderr}")
        self.assertIn("LANGUAGE_TABLE_OK", result.stdout)


@unittest.skipUnless(
    ML_INSTALLED and os.environ.get(REAL_CHECKPOINTS_ENV) == "1",
    f"set {REAL_CHECKPOINTS_ENV}=1 to download and run the real Canary/Parakeet checkpoints",
)
class TestRealNvidiaCheckpoints(unittest.TestCase):
    """The pinned checkpoints through the production wrappers, on the CPU."""

    def _assert_smoke(self, engine):
        result = _run_script(CHECKPOINT_SCRIPT, [engine], CHECKPOINT_TIMEOUT_SECONDS, strict_warnings=False)
        self.assertEqual(result.returncode, 0, f"Stderr: {result.stderr}")
        self.assertIn(f"NVIDIA_ASR_SMOKE_OK {engine}", result.stdout)

    def test_canary_checkpoint_on_noise(self):
        self._assert_smoke("canary")

    def test_parakeet_checkpoint_on_noise(self):
        self._assert_smoke("parakeet")


if __name__ == "__main__":
    unittest.main()
