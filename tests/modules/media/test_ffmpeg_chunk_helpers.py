"""FFmpeg helpers behind chunked vocal separation, plus the RF64 guard on extraction."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from modules.media import ffmpeg_utils


def _popen_returning(returncode, stderr_text=""):
    process = MagicMock()
    process.returncode = returncode
    process.communicate.return_value = ("", stderr_text)
    process.__enter__.return_value = process
    process.__exit__.return_value = False
    return process


class TestRunFfmpegQuiet(unittest.TestCase):
    def test_success_registers_and_unregisters_the_process(self):
        process = _popen_returning(0)
        with (
            patch("modules.media.ffmpeg_utils.subprocess.Popen", return_value=process),
            patch("modules.media.ffmpeg_utils.register_subprocess") as register,
            patch("modules.media.ffmpeg_utils.unregister_subprocess") as unregister,
        ):
            ffmpeg_utils.run_ffmpeg_quiet(["ffmpeg", "-i", "x"])
        register.assert_called_once_with(process)
        unregister.assert_called_once_with(process)

    def test_failure_raises_with_stderr_tail(self):
        process = _popen_returning(1, "some noise\nActual error: bad input")
        with (
            patch("modules.media.ffmpeg_utils.subprocess.Popen", return_value=process),
            patch("modules.media.ffmpeg_utils.register_subprocess"),
            patch("modules.media.ffmpeg_utils.unregister_subprocess") as unregister,
        ):
            with self.assertRaises(RuntimeError) as raised:
                ffmpeg_utils.run_ffmpeg_quiet(["ffmpeg"])
        self.assertIn("Actual error: bad input", str(raised.exception))
        unregister.assert_called_once()


class TestWriteAudioWindow(unittest.TestCase):
    def test_native_format_cut(self):
        with patch("modules.media.ffmpeg_utils.run_ffmpeg_quiet") as run:
            ffmpeg_utils.write_audio_window("src.wav", "dst.wav", 58.0, 64.0)
        cmd = run.call_args[0][0]
        self.assertEqual(cmd[cmd.index("-ss") + 1], "58.000")
        self.assertEqual(cmd[cmd.index("-t") + 1], "64.000")
        self.assertEqual(cmd[cmd.index("-i") + 1], "src.wav")
        self.assertNotIn("-ac", cmd)
        self.assertNotIn("-ar", cmd)
        self.assertEqual(cmd[cmd.index("-c:a") + 1], "pcm_f32le")
        self.assertEqual(cmd[cmd.index("-rf64") + 1], "auto")
        self.assertEqual(cmd[-1], "dst.wav")

    def test_mono_16k_downmix(self):
        with patch("modules.media.ffmpeg_utils.run_ffmpeg_quiet") as run:
            ffmpeg_utils.write_audio_window("stem.wav", "out.wav", 2.0, 60.0, mono_16k=True)
        cmd = run.call_args[0][0]
        self.assertEqual(cmd[cmd.index("-ac") + 1], "1")
        self.assertEqual(cmd[cmd.index("-ar") + 1], "16000")
        # Seek/duration are input options so the decode is bounded to the window.
        self.assertLess(cmd.index("-ss"), cmd.index("-i"))


class TestConcatAudioFiles(unittest.TestCase):
    def test_writes_escaped_manifest_and_joins_with_stream_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            list_path = os.path.join(folder, "stems.list")
            sources = [os.path.join(folder, "a.wav"), os.path.join(folder, "it's.wav")]
            with patch("modules.media.ffmpeg_utils.run_ffmpeg_quiet") as run:
                ffmpeg_utils.concat_audio_files(sources, "final.wav", list_path)
            with open(list_path, encoding="utf-8") as handle:
                manifest = handle.read()
        self.assertIn(f"file '{sources[0]}'\n", manifest)
        # A single quote inside a path must be written as '\'' for the concat demuxer.
        self.assertIn("it'\\''s.wav'", manifest)
        cmd = run.call_args[0][0]
        self.assertEqual(cmd[cmd.index("-f") + 1], "concat")
        self.assertEqual(cmd[cmd.index("-safe") + 1], "0")
        self.assertEqual(cmd[cmd.index("-i") + 1], list_path)
        self.assertEqual(cmd[cmd.index("-c:a") + 1], "copy")
        self.assertEqual(cmd[cmd.index("-rf64") + 1], "auto")
        self.assertEqual(cmd[-1], "final.wav")

    def test_manifest_entries_are_absolute_for_relative_inputs(self):
        # FFmpeg resolves relative manifest entries against the manifest's directory,
        # so a relative input path (python auto_subtitle.py clip.mp4) must be absolutised.
        with tempfile.TemporaryDirectory() as folder:
            list_path = os.path.join(folder, "nested", "stems.list")
            os.makedirs(os.path.dirname(list_path))
            with patch("modules.media.ffmpeg_utils.run_ffmpeg_quiet"):
                ffmpeg_utils.concat_audio_files([os.path.join("clip.asg-temp", "chunk_000.wav")], "final.wav", list_path)
            with open(list_path, encoding="utf-8") as handle:
                manifest = handle.read()
        expected = os.path.abspath(os.path.join("clip.asg-temp", "chunk_000.wav"))
        self.assertEqual(manifest, f"file '{ffmpeg_utils._concat_quote(expected)}'\n")

    def test_concat_quote(self):
        self.assertEqual(ffmpeg_utils._concat_quote("plain.wav"), "plain.wav")
        self.assertEqual(ffmpeg_utils._concat_quote("a'b"), "a'\\''b")


class TestExtractionRf64Guard(unittest.TestCase):
    def test_extraction_command_enables_rf64_auto(self):
        # A 16 kHz mono float WAV passes the 4 GB RIFF limit at about 17 h; RF64 must be on.
        with tempfile.TemporaryDirectory() as folder:
            video = os.path.join(folder, "long.mp4")
            with open(video, "wb") as handle:
                handle.write(b"\x00" * 16)
            captured = {}

            def fake_run(cmd, _desc, _dur):
                captured["cmd"] = cmd
                with open(cmd[-1], "wb") as handle:
                    handle.write(b"\x00" * 4096)

            with (
                patch("modules.media.ffmpeg_utils.run_ffmpeg_progress", side_effect=fake_run),
                patch("modules.media.ffmpeg_utils.get_audio_duration", return_value=1.0),
                patch("modules.media.ffmpeg_utils.log"),
            ):
                ffmpeg_utils.extract_clean_audio(video)
        cmd = captured["cmd"]
        self.assertEqual(cmd[cmd.index("-rf64") + 1], "auto")
        self.assertLess(cmd.index("-rf64"), len(cmd) - 1, "rf64 is an output option before the target path")


if __name__ == "__main__":
    unittest.main()
