"""FFmpeg helpers behind chunked vocal separation, plus the RF64 guard on extraction."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from modules.media import ffmpeg_utils
from modules.safe_io import SymlinkRefusedError


def _popen_returning(returncode, stderr_text=""):
    process = MagicMock()
    process.returncode = returncode
    process.communicate.return_value = ("", stderr_text)
    process.__enter__.return_value = process
    process.__exit__.return_value = False
    return process


def _write_cmd_output(cmd, payload=b"WAV"):
    """Emulate FFmpeg by writing ``payload`` to the command's final output path."""
    with open(cmd[-1], "wb") as handle:
        handle.write(payload)


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
        with tempfile.TemporaryDirectory() as folder:
            dst = os.path.join(folder, "dst.wav")
            with patch("modules.media.ffmpeg_utils.run_ffmpeg_quiet", side_effect=_write_cmd_output) as run:
                ffmpeg_utils.write_audio_window("src.wav", dst, 58.0, 64.0)
            cmd = run.call_args[0][0]
            self.assertEqual(cmd[cmd.index("-ss") + 1], "58.000")
            self.assertEqual(cmd[cmd.index("-t") + 1], "64.000")
            self.assertEqual(cmd[cmd.index("-i") + 1], "src.wav")
            self.assertNotIn("-ac", cmd)
            self.assertNotIn("-ar", cmd)
            self.assertEqual(cmd[cmd.index("-c:a") + 1], "pcm_f32le")
            self.assertEqual(cmd[cmd.index("-rf64") + 1], "auto")
            # FFmpeg must write into a private scratch file, not the predictable destination.
            self.assertNotEqual(cmd[-1], dst)
            self.assertTrue(cmd[-1].startswith(folder + os.sep))
            self.assertTrue(os.path.isfile(dst))
            with open(dst, "rb") as handle:
                self.assertEqual(handle.read(), b"WAV")

    def test_mono_16k_downmix(self):
        with tempfile.TemporaryDirectory() as folder:
            out = os.path.join(folder, "out.wav")
            with patch("modules.media.ffmpeg_utils.run_ffmpeg_quiet", side_effect=_write_cmd_output) as run:
                ffmpeg_utils.write_audio_window("stem.wav", out, 2.0, 60.0, mono_16k=True)
            cmd = run.call_args[0][0]
            self.assertEqual(cmd[cmd.index("-ac") + 1], "1")
            self.assertEqual(cmd[cmd.index("-ar") + 1], "16000")
            # Seek/duration are input options so the decode is bounded to the window.
            self.assertLess(cmd.index("-ss"), cmd.index("-i"))
            self.assertTrue(os.path.isfile(out))

    def test_planted_symlink_destination_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            victim = os.path.join(folder, "victim.bin")
            with open(victim, "wb") as handle:
                handle.write(b"keep-me")
            link = os.path.join(folder, "chunk_000.wav")
            try:
                os.symlink(victim, link)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")
            with patch("modules.media.ffmpeg_utils.run_ffmpeg_quiet", side_effect=_write_cmd_output):
                with self.assertRaises(SymlinkRefusedError):
                    ffmpeg_utils.write_audio_window("src.wav", link, 0.0, 1.0)
            with open(victim, "rb") as handle:
                self.assertEqual(handle.read(), b"keep-me")

    def test_ffmpeg_failure_discards_scratch(self):
        with tempfile.TemporaryDirectory() as folder:
            dst = os.path.join(folder, "chunk_000.wav")
            with patch("modules.media.ffmpeg_utils.run_ffmpeg_quiet", side_effect=RuntimeError("ffmpeg died")):
                with self.assertRaises(RuntimeError):
                    ffmpeg_utils.write_audio_window("src.wav", dst, 0.0, 1.0)
            self.assertFalse(os.path.exists(dst))
            leftovers = [name for name in os.listdir(folder) if name.startswith(".asg-tmp-")]
            self.assertEqual(leftovers, [])


class TestConcatAudioFiles(unittest.TestCase):
    def test_writes_escaped_manifest_and_joins_with_stream_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            list_path = os.path.join(folder, "stems.list")
            final_path = os.path.join(folder, "final.wav")
            sources = [os.path.join(folder, "a.wav"), os.path.join(folder, "it's.wav")]
            with patch("modules.media.ffmpeg_utils.run_ffmpeg_quiet", side_effect=_write_cmd_output) as run:
                ffmpeg_utils.concat_audio_files(sources, final_path, list_path)
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
            self.assertNotEqual(cmd[-1], final_path)
            self.assertTrue(os.path.isfile(final_path))

    def test_manifest_entries_are_absolute_for_relative_inputs(self):
        # FFmpeg resolves relative manifest entries against the manifest's directory,
        # so a relative input path (python auto_subtitle.py clip.mp4) must be absolutised.
        with tempfile.TemporaryDirectory() as folder:
            nested = os.path.join(folder, "nested")
            os.makedirs(nested)
            list_path = os.path.join(nested, "stems.list")
            final_path = os.path.join(folder, "final.wav")
            with patch("modules.media.ffmpeg_utils.run_ffmpeg_quiet", side_effect=_write_cmd_output):
                ffmpeg_utils.concat_audio_files([os.path.join("clip.asg-temp", "chunk_000.wav")], final_path, list_path)
            with open(list_path, encoding="utf-8") as handle:
                manifest = handle.read()
        expected = os.path.abspath(os.path.join("clip.asg-temp", "chunk_000.wav"))
        self.assertEqual(manifest, f"file '{ffmpeg_utils._concat_quote(expected)}'\n")

    def test_planted_stems_list_symlink_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            victim = os.path.join(folder, "authorized_keys")
            with open(victim, "w", encoding="utf-8") as handle:
                handle.write("ssh-ed25519 AAAA victim-key\n")
            list_path = os.path.join(folder, "stems.list")
            final_path = os.path.join(folder, "final.wav")
            try:
                os.symlink(victim, list_path)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")
            with patch("modules.media.ffmpeg_utils.run_ffmpeg_quiet", side_effect=_write_cmd_output):
                with self.assertRaises(SymlinkRefusedError):
                    ffmpeg_utils.concat_audio_files([os.path.join(folder, "a.wav")], final_path, list_path)
            with open(victim, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "ssh-ed25519 AAAA victim-key\n")
            self.assertFalse(os.path.exists(final_path))

    def test_planted_concat_target_symlink_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            victim = os.path.join(folder, "victim.bin")
            with open(victim, "wb") as handle:
                handle.write(b"keep-me")
            list_path = os.path.join(folder, "stems.list")
            final_path = os.path.join(folder, "final.wav")
            try:
                os.symlink(victim, final_path)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")
            with patch("modules.media.ffmpeg_utils.run_ffmpeg_quiet", side_effect=_write_cmd_output):
                with self.assertRaises(SymlinkRefusedError):
                    ffmpeg_utils.concat_audio_files([os.path.join(folder, "a.wav")], final_path, list_path)
            with open(victim, "rb") as handle:
                self.assertEqual(handle.read(), b"keep-me")
            # Manifest may have been written; the join target must stay a link to the untouched victim.
            self.assertTrue(os.path.islink(final_path))

    def test_concat_ffmpeg_failure_discards_join_scratch(self):
        with tempfile.TemporaryDirectory() as folder:
            list_path = os.path.join(folder, "stems.list")
            final_path = os.path.join(folder, "final.wav")
            with patch("modules.media.ffmpeg_utils.run_ffmpeg_quiet", side_effect=RuntimeError("concat died")):
                with self.assertRaises(RuntimeError):
                    ffmpeg_utils.concat_audio_files([os.path.join(folder, "a.wav")], final_path, list_path)
            self.assertTrue(os.path.isfile(list_path))
            self.assertFalse(os.path.exists(final_path))
            leftovers = [name for name in os.listdir(folder) if name.startswith(".asg-tmp-")]
            self.assertEqual(leftovers, [])

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

            def fake_run(cmd, _desc, _dur, pass_fds=()):
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
