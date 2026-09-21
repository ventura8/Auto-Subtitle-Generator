"""End-of-video temp hygiene: nothing left behind on completion, everything kept for resume on failure."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from modules import safe_io, workdir
from modules.media import ffmpeg_utils
from modules.models import Segment
from modules.pipeline import isolated_translator, translation
from modules.subtitles import srt_io


class _WorkDirCase(unittest.TestCase):
    def setUp(self):
        global auto_subtitle
        import auto_subtitle

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = self._tmp.name
        self.video = os.path.join(self.folder, "movie.mp4")
        with open(self.video, "wb") as handle:
            handle.write(b"\x00" * 16)
        self.work_dir = workdir.work_dir_path(self.folder, "movie")

    def _entries(self):
        return sorted(entry for entry in os.listdir(self.folder) if entry != "movie.mp4")

    def _write(self, path, payload="x"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload)
        return path


class TestFinishTempHygiene(_WorkDirCase):
    def _populate_work_dir(self):
        workdir.ensure_work_dir(self.folder, "movie")
        self._write(os.path.join(self.work_dir, "movie_temp.wav"))
        self._write(os.path.join(self.work_dir, "movie.source_lang.txt"), "ro")
        scratch = safe_io.create_private_dir(self.work_dir, "movie.en")
        self._write(os.path.join(scratch, "output-1.srt"))

    def test_success_purges_work_dir_and_legacy_sidecars(self):
        self._populate_work_dir()
        # Sidecars an older release left beside the video.
        self._write(os.path.join(self.folder, "movie_temp.wav"))
        self._write(os.path.join(self.folder, "movie.source_lang.txt"), "ro")
        self._write(os.path.join(self.folder, "movie.manifest.json"), "{}")
        # Deliverables and unrelated user files must survive.
        self._write(os.path.join(self.folder, "movie.en.srt"), "1\n00:00:00,000 --> 00:00:01,000\nHi\n")
        self._write(os.path.join(self.folder, "notes.txt"), "keep")

        with patch("auto_subtitle.log") as mock_log:
            auto_subtitle._finish_temp_hygiene(self.folder, "movie", "movie.mp4", ([Segment(0, 1, "Hi")], "en", "out"))

        self.assertEqual(self._entries(), ["movie.en.srt", "notes.txt"])
        mock_log.assert_any_call("  [Temp] Work directory removed; no temporary files left behind.", "DEBUG")

    def test_no_speech_is_terminal_and_purges(self):
        self._populate_work_dir()
        with patch("auto_subtitle.log"):
            auto_subtitle._finish_temp_hygiene(self.folder, "movie", "movie.mp4", ([], None, None))
        self.assertEqual(self._entries(), [])

    def test_failure_keeps_work_dir_for_resume(self):
        self._populate_work_dir()
        with patch("auto_subtitle.log") as mock_log:
            auto_subtitle._finish_temp_hygiene(self.folder, "movie", "movie.mp4", (None, None, None))
        self.assertEqual(self._entries(), ["movie.asg-temp"])
        self.assertTrue(os.path.isfile(os.path.join(self.work_dir, "movie_temp.wav")))
        mock_log.assert_any_call(f"  [Temp] Keeping {self.work_dir} so the next run can resume.", "INFO")

    def test_failure_without_work_dir_is_silent(self):
        with patch("auto_subtitle.log") as mock_log:
            auto_subtitle._finish_temp_hygiene(self.folder, "movie", "movie.mp4", (None, None, None))
        mock_log.assert_not_called()

    def test_skip_of_finished_video_purges_leftover_work_dir(self):
        self._populate_work_dir()
        output = self._write(os.path.join(self.folder, "movie_multilang.mp4"))
        with patch("auto_subtitle.config.load_config", return_value=True), patch("auto_subtitle.log"):
            result = auto_subtitle.process_video(self.video, MagicMock())
        self.assertEqual(result, (None, None, output))
        self.assertEqual(self._entries(), ["movie_multilang.mp4"])

    def test_interrupt_during_pipeline_keeps_work_dir(self):
        self._populate_work_dir()
        with (
            patch("auto_subtitle.config.load_config", return_value=True),
            patch("auto_subtitle._process_video_pipeline", side_effect=KeyboardInterrupt),
            patch("auto_subtitle.log"),
        ):
            with self.assertRaises(KeyboardInterrupt):
                auto_subtitle.process_video(self.video, MagicMock())
        self.assertEqual(self._entries(), ["movie.asg-temp"])


class TestWorkDirBoundToInput(_WorkDirCase):
    def _run_pipeline_to_interrupt(self):
        with (
            patch("auto_subtitle.config.load_config", return_value=True),
            patch("auto_subtitle._process_video_pipeline", side_effect=KeyboardInterrupt),
            patch("auto_subtitle.log"),
        ):
            with self.assertRaises(KeyboardInterrupt):
                auto_subtitle.process_video(self.video, MagicMock())

    def test_unchanged_input_keeps_resume_state(self):
        self._run_pipeline_to_interrupt()  # first run creates the stamp
        stem = self._write(os.path.join(self.work_dir, "movie_temp.wav"))
        self._run_pipeline_to_interrupt()
        self.assertTrue(os.path.isfile(stem))
        self.assertTrue(os.path.isfile(os.path.join(self.work_dir, workdir.SOURCE_STAMP_NAME)))

    def test_replaced_input_discards_resume_state(self):
        self._run_pipeline_to_interrupt()
        stem = self._write(os.path.join(self.work_dir, "movie_temp.wav"))
        with open(self.video, "wb") as handle:  # same name, different file
            handle.write(b"\x01" * 32)
        self._run_pipeline_to_interrupt()
        self.assertFalse(os.path.exists(stem))
        self.assertEqual(self._entries(), ["movie.asg-temp"])

    def test_unstamped_work_dir_from_an_unknown_input_is_discarded(self):
        stem = self._write(os.path.join(self.work_dir, "movie_temp.wav"))
        self._run_pipeline_to_interrupt()
        self.assertFalse(os.path.exists(stem))

    def _pipeline_context_after_bind(self):
        with (
            patch("auto_subtitle.config.load_config", return_value=True),
            patch("auto_subtitle._process_video_pipeline", return_value=([], None, None)) as pipeline,
            patch("auto_subtitle.log"),
        ):
            auto_subtitle.process_video(self.video, MagicMock())
        return pipeline.call_args[0][2]

    def test_pipeline_learns_whether_the_input_changed(self):
        self.assertFalse(self._pipeline_context_after_bind()["input_changed"])
        self._run_pipeline_to_interrupt()  # leaves a stamped work dir behind
        self.assertFalse(self._pipeline_context_after_bind()["input_changed"])
        self._run_pipeline_to_interrupt()
        with open(self.video, "wb") as handle:
            handle.write(b"\x01" * 32)
        self.assertTrue(self._pipeline_context_after_bind()["input_changed"])

    def test_changed_input_ignores_srt_files_beside_the_video(self):
        # Deliverables from the previous file must not be resumed from either.
        self._write(os.path.join(self.folder, "movie.en.srt"), "1\n00:00:00,000 --> 00:00:01,000\nold\n")
        context = auto_subtitle._build_transcription_context(self.folder, "movie", self.video, input_changed=True)
        with (
            patch("auto_subtitle._check_resume") as mock_resume,
            patch("auto_subtitle.transcribe_video_audio", return_value=([], "en", None)) as mock_transcribe,
            patch("auto_subtitle.log") as mock_log,
        ):
            auto_subtitle._obtain_segments(context, MagicMock(), None, None)
        mock_resume.assert_not_called()
        mock_transcribe.assert_called_once()
        self.assertIn("changed since the last run", mock_log.call_args[0][0])
        pipeline_context = {"folder": self.folder, "base_name": "movie", "input_changed": True}
        with patch("auto_subtitle.translate_segments") as mock_translate, patch("auto_subtitle._clear_cuda_cache_if_available"):
            auto_subtitle._run_translation_step([], "ro", MagicMock(), pipeline_context)
        self.assertFalse(mock_translate.call_args[0][3]["reuse_outputs"])


class TestSourceLanguageSidecar(_WorkDirCase):
    def test_source_language_round_trips_through_work_dir(self):
        with patch("auto_subtitle.log"):
            auto_subtitle._write_recorded_source_language(self.folder, "movie", "ro")
        self.assertEqual(self._entries(), ["movie.asg-temp"])
        self.assertEqual(sorted(os.listdir(self.work_dir)), ["movie.source_lang.txt"])
        self.assertEqual(auto_subtitle._read_recorded_source_language(self.folder, "movie"), "ro")

    def test_missing_work_dir_reads_as_unknown(self):
        self.assertIsNone(auto_subtitle._read_recorded_source_language(self.folder, "movie"))

    def test_write_failure_is_logged_not_raised(self):
        self._write(self.work_dir, "occupied")  # a file where the directory should be
        with patch("auto_subtitle.log") as mock_log:
            auto_subtitle._write_recorded_source_language(self.folder, "movie", "ro")
        self.assertIn("Could not record source language", mock_log.call_args[0][0])


class TestSourceSrtScratch(_WorkDirCase):
    def test_source_srt_scratch_lives_in_work_dir(self):
        seen = {}

        def spy_writer(path, encoding="utf-8", scratch_dir=None):
            seen["scratch_dir"] = scratch_dir
            return safe_io.atomic_text_writer(path, encoding=encoding, scratch_dir=scratch_dir)

        with patch("modules.subtitles.srt_io.atomic_text_writer", side_effect=spy_writer), patch("auto_subtitle.log"):
            path = auto_subtitle._prepare_source_srt_path(self.folder, "movie", "en", "audio.wav", [Segment(0.0, 1.0, "Hi")])

        self.assertEqual(path, os.path.join(self.folder, "movie.en.srt"))
        self.assertEqual(seen["scratch_dir"], self.work_dir)
        self.assertEqual(self._entries(), ["movie.asg-temp", "movie.en.srt"])
        self.assertEqual(sorted(os.listdir(self.work_dir)), ["movie.source_lang.txt"])


class TestEmbedScratch(_WorkDirCase):
    def _run_embed(self, run_side_effect):
        with (
            patch("auto_subtitle.utils.run_ffmpeg_progress", side_effect=run_side_effect),
            patch("auto_subtitle.utils.get_audio_duration", return_value=1.0),
            patch("auto_subtitle.log"),
        ):
            return auto_subtitle.embed_subtitles(self.video, [(os.path.join(self.folder, "movie.en.srt"), "en", "English")], "en")

    def test_mux_scratch_is_in_work_dir_and_promoted(self):
        def fake_ffmpeg(cmd, _desc, _dur, pass_fds=()):
            self.assertTrue(cmd[-1].startswith(os.path.join(self.work_dir, safe_io.SCRATCH_PREFIX)))
            with open(cmd[-1], "wb") as handle:
                handle.write(b"muxed")

        output = self._run_embed(fake_ffmpeg)
        self.assertEqual(output, os.path.join(self.folder, "movie_multilang.mp4"))
        self.assertTrue(os.path.isfile(output))
        self.assertEqual(sorted(os.listdir(self.work_dir)), [])

    def test_interrupted_mux_discards_scratch(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run_embed(KeyboardInterrupt)
        self.assertEqual(sorted(os.listdir(self.work_dir)), [])
        self.assertEqual(self._entries(), ["movie.asg-temp"])

    def test_failed_mux_discards_scratch(self):
        self.assertIsNone(self._run_embed(RuntimeError("ffmpeg died")))
        self.assertEqual(sorted(os.listdir(self.work_dir)), [])


class TestExtractCleanAudioWorkDir(_WorkDirCase):
    def test_extracted_audio_lands_in_work_dir(self):
        def fake_ffmpeg(cmd, _desc, _dur, pass_fds=()):
            self.assertTrue(cmd[-1].startswith(os.path.join(self.work_dir, safe_io.SCRATCH_PREFIX)))
            with open(cmd[-1], "wb") as handle:
                handle.write(b"\x00" * 4096)

        with (
            patch("modules.media.ffmpeg_utils.run_ffmpeg_progress", side_effect=fake_ffmpeg),
            patch("modules.media.ffmpeg_utils.get_audio_duration", return_value=1.0),
            patch("modules.media.ffmpeg_utils.log"),
        ):
            wav = ffmpeg_utils.extract_clean_audio(self.video)

        self.assertEqual(wav, os.path.join(self.work_dir, "movie_temp.wav"))
        self.assertTrue(os.path.isfile(wav))
        self.assertEqual(sorted(os.listdir(self.work_dir)), ["movie_temp.wav"])
        self.assertEqual(self._entries(), ["movie.asg-temp"])

    def test_failed_extraction_leaves_empty_work_dir(self):
        with (
            patch("modules.media.ffmpeg_utils.run_ffmpeg_progress", side_effect=RuntimeError("no ffmpeg")),
            patch("modules.media.ffmpeg_utils.get_audio_duration", return_value=1.0),
            patch("modules.media.ffmpeg_utils.log"),
        ):
            with self.assertRaises(RuntimeError):
                ffmpeg_utils.extract_clean_audio(self.video)
        self.assertEqual(sorted(os.listdir(self.work_dir)), [])


class TestTranslationWorkDir(_WorkDirCase):
    def _context(self, src_lang="ro", missing=("fr",)):
        return {
            "missing_langs": list(missing),
            "source_data": [{"text": "Salut", "start": 0.0, "end": 1.0}],
            "src_code": "ron_Latn",
            "src_lang": src_lang,
            "folder": self.folder,
            "base_name": "movie",
            "segments": [Segment(0.0, 1.0, "Salut")],
        }

    def test_manifest_and_worker_files_live_in_work_dir(self):
        with patch("modules.configuration.config.TARGET_LANGUAGES", {"en": {"code": "eng_Latn"}, "fr": {"code": "fra_Latn"}}):
            manifest_path, temp_files = translation._create_translation_manifest(self._context(missing=("en", "fr")))

        self.assertEqual(os.path.dirname(manifest_path), self.work_dir)
        for temp_file in temp_files:
            self.assertEqual(os.path.dirname(temp_file), self.work_dir, temp_file)
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(os.path.dirname(manifest["pivot"]["output"]), self.work_dir)
        self.assertEqual(os.path.dirname(manifest["pivot"]["en_output"]), self.work_dir)
        for job in manifest["jobs"]:
            self.assertEqual(os.path.dirname(job["output"]), self.work_dir)
            self.assertEqual(os.path.dirname(job["input"]), self.work_dir)
        self.assertEqual(self._entries(), ["movie.asg-temp"])

    def test_completed_output_writes_srt_beside_video_with_scratch_in_work_dir(self):
        workdir.ensure_work_dir(self.folder, "movie")
        output = self._write(translation._temp_output_path(self.folder, "movie", "fr"), json.dumps(["Bonjour"]))
        pending = {"fr"}
        with patch("modules.pipeline.translation.log"):
            remaining = translation._scan_pending_outputs(pending, self.folder, "movie", [Segment(0.0, 1.0, "Salut")])
        self.assertEqual(remaining, set())
        self.assertFalse(os.path.exists(output))
        self.assertEqual(self._entries(), ["movie.asg-temp", "movie.fr.srt"])
        self.assertEqual(sorted(os.listdir(self.work_dir)), [])

    def test_failed_worker_keeps_temp_files(self):
        workdir.ensure_work_dir(self.folder, "movie")
        pivot = self._write(os.path.join(self.work_dir, "movie.pivot_pivoted.json"), "[]")
        context = {**self._context(), "manifest_path": "m.json", "temp_files": [pivot]}
        proc = MagicMock()
        proc.returncode = 1
        proc.poll.return_value = 1
        with (
            patch("modules.pipeline.translation.subprocess.Popen") as mock_popen,
            patch("modules.pipeline.translation.log") as mock_log,
        ):
            mock_popen.return_value.__enter__.return_value = proc
            with self.assertRaises(RuntimeError):
                translation._run_worker_process(context)
        self.assertTrue(os.path.isfile(pivot))
        mock_log.assert_any_call(f"  [Temp] Translation did not finish; keeping resumable files in {self.work_dir}", "WARNING")

    def test_successful_worker_removes_temp_files(self):
        workdir.ensure_work_dir(self.folder, "movie")
        pivot = self._write(os.path.join(self.work_dir, "movie.pivot_pivoted.json"), "[]")
        context = {**self._context(), "manifest_path": "m.json", "temp_files": [pivot]}
        proc = MagicMock()
        proc.returncode = 0
        proc.poll.return_value = 0
        with (
            patch("modules.pipeline.translation.subprocess.Popen") as mock_popen,
            patch("modules.pipeline.translation._poll_translation_results", return_value=set()),
            patch("modules.pipeline.translation.log"),
        ):
            mock_popen.return_value.__enter__.return_value = proc
            translation._run_worker_process(context)
        self.assertFalse(os.path.exists(pivot))


class TestPivotReuseValidation(_WorkDirCase):
    def _pivot_job(self, source, pivot):
        workdir.ensure_work_dir(self.folder, "movie")
        input_path = self._write(os.path.join(self.work_dir, "movie.common_input.json"), json.dumps(source))
        output_path = os.path.join(self.work_dir, "movie.pivot_pivoted.json")
        if pivot is not None:
            self._write(output_path, pivot if isinstance(pivot, str) else json.dumps(pivot))
        return {"input": input_path, "output": output_path, "emit_en_output": False}

    def test_matching_pivot_is_reused(self):
        source = [{"text": "Salut", "start": 0.0, "end": 1.0}]
        job = self._pivot_job(source, [{"text": "Hello", "start": 0.0, "end": 1.0}])
        with patch("modules.pipeline.isolated_translator.log"):
            self.assertTrue(isolated_translator._reuse_existing_pivot_output_if_available(job))
        self.assertTrue(os.path.isfile(job["output"]))

    def test_stale_pivot_with_different_timings_is_discarded(self):
        source = [{"text": "Salut", "start": 0.0, "end": 1.0}, {"text": "Pa", "start": 1.0, "end": 2.0}]
        job = self._pivot_job(source, [{"text": "Hello", "start": 0.0, "end": 1.0}, {"text": "Bye", "start": 5.0, "end": 6.0}])
        with patch("modules.pipeline.isolated_translator.log") as mock_log:
            self.assertFalse(isolated_translator._reuse_existing_pivot_output_if_available(job))
        self.assertFalse(os.path.exists(job["output"]))
        self.assertIn("does not match", mock_log.call_args[0][0])

    def test_pivot_with_wrong_length_or_corrupt_json_is_discarded(self):
        source = [{"text": "Salut", "start": 0.0, "end": 1.0}]
        short = self._pivot_job(source, [])
        self.assertIsNone(isolated_translator._load_pivot_output_matching_input(short["output"], short["input"]))
        corrupt = self._pivot_job(source, "{not json")
        self.assertIsNone(isolated_translator._load_pivot_output_matching_input(corrupt["output"], corrupt["input"]))
        not_list = self._pivot_job(source, {"text": "x"})
        self.assertIsNone(isolated_translator._load_pivot_output_matching_input(not_list["output"], not_list["input"]))
        no_text = self._pivot_job(source, [{"start": 0.0, "end": 1.0}])
        self.assertIsNone(isolated_translator._load_pivot_output_matching_input(no_text["output"], no_text["input"]))

    def test_malformed_pivot_or_source_shapes_are_discarded_not_raised(self):
        # Corrupt-but-valid JSON of the wrong shape must regenerate the pivot, never crash the worker.
        source = [{"text": "Salut", "start": 0.0, "end": 1.0}]
        pivot = [{"text": "Hello", "start": 0.0, "end": 1.0}]
        for bad_source in (42, {"text": "x"}, [1]):
            job = self._pivot_job(bad_source, pivot)
            self.assertIsNone(isolated_translator._load_pivot_output_matching_input(job["output"], job["input"]))
        non_string_text = self._pivot_job(source, [{"text": None, "start": 0.0, "end": 1.0}])
        self.assertIsNone(isolated_translator._load_pivot_output_matching_input(non_string_text["output"], non_string_text["input"]))

    def test_missing_pivot_is_not_reused(self):
        job = self._pivot_job([{"text": "Salut", "start": 0.0, "end": 1.0}], None)
        self.assertFalse(isolated_translator._reuse_existing_pivot_output_if_available(job))


class TestSafeIoScratchDir(_WorkDirCase):
    def test_atomic_text_writer_uses_scratch_dir_and_promotes_across_directories(self):
        workdir.ensure_work_dir(self.folder, "movie")
        destination = os.path.join(self.folder, "movie.en.srt")
        with safe_io.atomic_text_writer(destination, scratch_dir=self.work_dir) as handle:
            scratch_entries = os.listdir(self.work_dir)
            self.assertEqual(len(scratch_entries), 1)
            self.assertTrue(scratch_entries[0].startswith(safe_io.SCRATCH_PREFIX))
            self.assertEqual([e for e in os.listdir(self.folder) if e.startswith(safe_io.SCRATCH_PREFIX)], [])
            handle.write("content")
        with open(destination, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "content")
        self.assertEqual(os.listdir(self.work_dir), [])

    def test_reserve_temp_path_scratch_dir_discard(self):
        workdir.ensure_work_dir(self.folder, "movie")
        reservation = safe_io.reserve_temp_path(os.path.join(self.folder, "movie_multilang.mp4"), scratch_dir=self.work_dir)
        self.assertTrue(reservation.path.startswith(self.work_dir))
        safe_io.discard_temp_path(reservation)
        self.assertEqual(os.listdir(self.work_dir), [])

    def test_srt_writers_forward_scratch_dir(self):
        workdir.ensure_work_dir(self.folder, "movie")
        seen = []

        def spy_writer(path, encoding="utf-8", scratch_dir=None):
            seen.append(scratch_dir)
            return safe_io.atomic_text_writer(path, encoding=encoding, scratch_dir=scratch_dir)

        with patch("modules.subtitles.srt_io.atomic_text_writer", side_effect=spy_writer):
            srt_io.save_srt([Segment(0.0, 1.0, "Hi")], os.path.join(self.folder, "movie.en.srt"), scratch_dir=self.work_dir)
            translated_path = os.path.join(self.folder, "movie.fr.srt")
            srt_io.save_translated_srt([Segment(0.0, 1.0, "Hi")], ["Salut"], translated_path, scratch_dir=self.work_dir)
        self.assertEqual(seen, [self.work_dir, self.work_dir])


if __name__ == "__main__":
    unittest.main()
