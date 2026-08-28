"""End-to-end pipeline test using real dependencies and synthetic media."""

import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

FFMPEG_TIMEOUT_SECONDS = 30
NLLB_TIMEOUT_SECONDS = 900


def _run_with_timeout(command, timeout, **kwargs):
    """Run a subprocess and fail the test clearly if it exceeds its timeout."""
    try:
        return subprocess.run(command, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as error:
        raise AssertionError(f"Command timed out after {timeout} seconds: {error.cmd}") from error


def _create_synthetic_test_video(output_video_path, duration_seconds=3):
    """Generate a valid synthetic MP4 video with a test audio tone."""
    from modules.media.ffmpeg_utils import get_ffmpeg_paths

    ffmpeg_bin, _ = get_ffmpeg_paths()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration_seconds}:size=320x240:rate=24",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=1000:duration={duration_seconds}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        output_video_path,
    ]
    _run_with_timeout(cmd, FFMPEG_TIMEOUT_SECONDS, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class TestRealPipelineE2E(unittest.TestCase):
    """Real dependencies end-to-end test verifying execution without mocks."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_path = os.path.join(self.temp_dir.name, "sample_test.mp4")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_real_audio_extraction_and_duration(self):
        from modules.media.ffmpeg_utils import extract_clean_audio, get_audio_duration

        _create_synthetic_test_video(self.video_path, duration_seconds=3)
        duration = get_audio_duration(self.video_path)
        self.assertGreater(duration, 2.0)

        wav_path = extract_clean_audio(self.video_path)
        self.assertTrue(os.path.exists(wav_path))
        self.assertGreater(os.path.getsize(wav_path), 1024)

    def test_real_hardware_detection(self):
        from modules.models import OPTIMIZER

        OPTIMIZER.detect_hardware(verbose=False)
        snapshot = OPTIMIZER.snapshot()
        self.assertIn("gpu_name", snapshot)
        self.assertIn("profile", snapshot)
        self.assertIn("cpu_cores", snapshot)
        self.assertGreaterEqual(snapshot["cpu_cores"], 1)

    def test_real_nllb_translator_cpu_execution(self):
        # In pytest, conftest.py mocks transformers for unit tests.
        # Run real NLLB translation in an unmocked subprocess using the distilled 600M model on CPU:
        code = """
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_TORCHAUDIO"] = "0"
from modules.configuration import config
config.NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"
from modules.translators.nllb import NLLBTranslator
translator = NLLBTranslator()
res = translator.translate(["Hello"], "eng_Latn", "spa_Latn")
assert len(res) == 1 and len(res[0].strip()) > 0
print("NLLB_TRANSLATE_OK")
"""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = os.environ.copy()
        env["PYTHONPATH"] = repo_root
        # NLLB may download and initialize a 600M model, so allow a bounded 15 minutes.
        result = _run_with_timeout(
            [sys.executable, "-c", code],
            NLLB_TIMEOUT_SECONDS,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(result.returncode, 0, f"Stderr: {result.stderr}")
        self.assertIn("NLLB_TRANSLATE_OK", result.stdout)

    def test_real_cli_dry_run_or_help(self):
        script_path = os.path.join(REPO_ROOT, "auto_subtitle.py")
        env = os.environ.copy()
        env["PYTHONPATH"] = REPO_ROOT
        result = _run_with_timeout(
            [sys.executable, script_path, "--help"],
            FFMPEG_TIMEOUT_SECONDS,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(result.returncode, 0, f"Stderr: {result.stderr}")
        self.assertIn("Auto Subtitle Generator", result.stdout)


if __name__ == "__main__":
    unittest.main()
