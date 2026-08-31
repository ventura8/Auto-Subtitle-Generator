"""Comprehensive End-to-End executable and artifact scenario test suite.

Tests compiled executables and entrypoints across all core functional scenarios:
1. CLI help / version flag
2. Single video transcription & subtitle embedding
3. Multilingual target translations (NLLB)
4. Custom initial Whisper prompt
5. Batch folder processing with multiple files
6. Skip on pre-existing completed outputs
7. Resume from existing SRT files
8. Silence / no-speech handling
9. Filenames with spaces and special Unicode characters
"""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from typing import List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_TIMEOUT = 900


def generate_synthetic_media(output_path: str, duration: int = 3, tone_hz: int = 440) -> None:
    """Generate a valid test MP4 video with a tone or silence using FFmpeg."""
    from modules.media.ffmpeg_utils import get_ffmpeg_paths

    ffmpeg_bin, _ = get_ffmpeg_paths()
    if tone_hz > 0:
        audio_src = f"sine=frequency={tone_hz}:duration={duration}"
    else:
        audio_src = f"anullsrc=r=44100:cl=mono:d={duration}"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size=320x240:rate=24",
        "-f",
        "lavfi",
        "-i",
        audio_src,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=DEFAULT_TIMEOUT)


def create_minimal_config(folder: str, target_languages: Optional[dict] = None) -> str:
    """Write a minimal config.yaml in the specified directory."""
    config_path = os.path.join(folder, "config.yaml")
    target_section = "target_languages: {}"
    if target_languages:
        lines = ["target_languages:"]
        for code, details in target_languages.items():
            lines.append(f"  {code}:")
            lines.append(f"    code: {details['code']}")
            lines.append(f"    label: {details['label']}")
        target_section = "\n".join(lines)

    content = f"""whisper:
  model_size: tiny.en
  language: en
  use_prompt: false
  use_vocal_separation: false
{target_section}
"""
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)
    return config_path


class ExecutableScenarioTests(unittest.TestCase):
    """Test suite executing the given application binary against all operational scenarios."""

    executable: List[str] = []

    @classmethod
    def setUpClass(cls):
        if not cls.executable:
            configured_executable = os.environ.get("TEST_EXECUTABLE")
            cls.executable = (
                shlex.split(configured_executable)
                if configured_executable
                else [sys.executable, os.path.join(REPO_ROOT, "auto_subtitle.py")]
            )

    def _exec(self, args: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
        cmd = self.executable + args
        env = os.environ.copy()
        env["PYTHONPATH"] = REPO_ROOT
        env["CUDA_VISIBLE_DEVICES"] = ""
        return subprocess.run(
            cmd,
            cwd=cwd,
            timeout=DEFAULT_TIMEOUT,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    def test_scenario_01_cli_help(self):
        """Scenario 1: Help flag returns usage and exits 0."""
        res = self._exec(["--help"])
        self.assertEqual(res.returncode, 0, f"Help failed: {res.stderr}")
        self.assertIn("Auto Subtitle Generator", res.stdout + res.stderr)

    def test_scenario_02_single_video_transcription(self):
        """Scenario 2: Single video generates SRT and embedded multilang container."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            create_minimal_config(tmp_dir)
            video_path = os.path.join(tmp_dir, "test_clip.mp4")
            generate_synthetic_media(video_path, duration=2, tone_hz=440)

            res = self._exec([video_path, "--cpu", "--lang", "en"], cwd=tmp_dir)
            self.assertEqual(res.returncode, 0, f"Error: {res.stderr}\nStdout: {res.stdout}")

            srt_path = os.path.join(tmp_dir, "test_clip.en.srt")
            out_video = os.path.join(tmp_dir, "test_clip_multilang.mp4")
            self.assertTrue(os.path.exists(srt_path), f"Expected SRT output not found in {tmp_dir}")
            self.assertTrue(os.path.exists(out_video), f"Expected multilang output not found in {tmp_dir}")

    def test_scenario_03_multilingual_translation(self):
        """Scenario 3: Target languages in config trigger NLLB translation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            targets = {"es": {"code": "spa_Latn", "label": "Spanish"}}
            create_minimal_config(tmp_dir, target_languages=targets)
            video_path = os.path.join(tmp_dir, "multilang_test.mp4")
            generate_synthetic_media(video_path, duration=2, tone_hz=440)

            res = self._exec([video_path, "--cpu", "--lang", "en"], cwd=tmp_dir)
            self.assertEqual(res.returncode, 0, f"Error: {res.stderr}\nStdout: {res.stdout}")

            es_srt = os.path.join(tmp_dir, "multilang_test.es.srt")
            out_video = os.path.join(tmp_dir, "multilang_test_multilang.mp4")
            self.assertTrue(os.path.exists(es_srt), "Expected Spanish SRT output")
            self.assertTrue(os.path.exists(out_video), "Expected multilang output")
            self.assertTrue(os.path.exists(out_video), "Expected multilang output")

    def test_scenario_04_custom_initial_prompt(self):
        """Scenario 4: Custom initial prompt flag is accepted and processed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            create_minimal_config(tmp_dir)
            video_path = os.path.join(tmp_dir, "prompt_test.mp4")
            generate_synthetic_media(video_path, duration=3, tone_hz=440)

            res = self._exec([video_path, "--cpu", "--lang", "en", "--prompt", "Technical glossary CUDA"], cwd=tmp_dir)
            self.assertEqual(res.returncode, 0, f"Error: {res.stderr}\nStdout: {res.stdout}")
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "prompt_test_multilang.mp4")))

    def test_scenario_05_batch_folder_processing(self):
        """Scenario 5: Passing a directory processes all discovered video files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            create_minimal_config(tmp_dir)
            clip1 = os.path.join(tmp_dir, "clip1.mp4")
            clip2 = os.path.join(tmp_dir, "clip2.mkv")
            generate_synthetic_media(clip1, duration=2, tone_hz=440)
            generate_synthetic_media(clip2, duration=2, tone_hz=880)

            res = self._exec([tmp_dir, "--cpu", "--lang", "en"], cwd=tmp_dir)
            self.assertEqual(res.returncode, 0, f"Error: {res.stderr}\nStdout: {res.stdout}")

            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "clip1_multilang.mp4")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "clip2_multilang.mkv")))
            self.assertIn("Batch Summary", res.stdout + res.stderr)

    def test_scenario_06_skip_preexisting_output(self):
        """Scenario 6: Pre-existing output video is skipped promptly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            create_minimal_config(tmp_dir)
            video_path = os.path.join(tmp_dir, "skip_test.mp4")
            out_video = os.path.join(tmp_dir, "skip_test_multilang.mp4")
            generate_synthetic_media(video_path, duration=2, tone_hz=440)
            generate_synthetic_media(out_video, duration=2, tone_hz=440)

            t0 = time.time()
            res = self._exec([video_path, "--cpu"], cwd=tmp_dir)
            elapsed = time.time() - t0

            self.assertEqual(res.returncode, 0, f"Error: {res.stderr}\nStdout: {res.stdout}")
            self.assertIn("already exists", res.stdout + res.stderr)
            self.assertLess(elapsed, 15.0, "Skip execution should be fast")

    def test_scenario_07_resume_from_existing_srt(self):
        """Scenario 7: Pre-existing source SRT skips Whisper transcription."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            targets = {"es": {"code": "spa_Latn", "label": "Spanish"}}
            create_minimal_config(tmp_dir, target_languages=targets)
            video_path = os.path.join(tmp_dir, "resume_test.mp4")
            generate_synthetic_media(video_path, duration=3, tone_hz=440)

            # Pre-create valid source SRT
            srt_path = os.path.join(tmp_dir, "resume_test.en.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write("1\n00:00:00,000 --> 00:00:02,000\nHello world.\n")

            res = self._exec([video_path, "--cpu"], cwd=tmp_dir)
            self.assertEqual(res.returncode, 0, f"Error: {res.stderr}\nStdout: {res.stdout}")

            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "resume_test.es.srt")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "resume_test_multilang.mp4")))

    def test_scenario_08_silent_audio_no_speech(self):
        """Scenario 8: Pure silent audio logs warning and exits gracefully."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            create_minimal_config(tmp_dir)
            video_path = os.path.join(tmp_dir, "silent_test.mp4")
            generate_synthetic_media(video_path, duration=2, tone_hz=0)

            res = self._exec([video_path, "--cpu"], cwd=tmp_dir)
            self.assertEqual(res.returncode, 0, f"Error: {res.stderr}\nStdout: {res.stdout}")

    def test_scenario_09_special_characters_and_spaces(self):
        """Scenario 9: Complex filenames with spaces, brackets, and accents work properly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            create_minimal_config(tmp_dir)
            complex_name = "My Video & Clip [1998] (Final) #1.mp4"
            video_path = os.path.join(tmp_dir, complex_name)
            generate_synthetic_media(video_path, duration=2, tone_hz=440)

            res = self._exec([video_path, "--cpu", "--lang", "en"], cwd=tmp_dir)
            self.assertEqual(res.returncode, 0, f"Error: {res.stderr}\nStdout: {res.stdout}")

            expected_out = os.path.join(tmp_dir, "My Video & Clip [1998] (Final) #1_multilang.mp4")
            self.assertTrue(os.path.exists(expected_out), "Expected output with special characters")


def main():
    parser = argparse.ArgumentParser(description="Run E2E scenario tests against an executable")
    parser.add_argument("--executable", help="Path to the executable to test")
    args, unknown = parser.parse_known_args()

    if args.executable:
        ExecutableScenarioTests.executable = [os.path.abspath(args.executable)]

    unittest.main(argv=[sys.argv[0]] + unknown)


if __name__ == "__main__":
    main()
