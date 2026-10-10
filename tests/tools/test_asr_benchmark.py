"""Unit tests for the pure helpers of the ASR benchmark harness (no models, numpy or audio libraries)."""

import io
import json
import os
import tarfile
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tests.tools import asr_bench_data as data
from tests.tools import asr_bench_engines as engines
from tests.tools import asr_bench_report as report
from tests.tools import asr_bench_scoring as scoring
from tests.tools import asr_benchmark


def _utterance(uid, cluster, samples):
    """An utterance whose audio is a plain list of ``samples`` zeros."""
    return data.Utterance(uid, cluster, f"text {uid}", [0.0] * samples)


class TestScoring(unittest.TestCase):
    """Batching, timing and long-form scoring."""

    def test_batches_are_length_sorted_and_cover_every_index(self):
        utterances = [_utterance("a", "1", 10), _utterance("b", "2", 30), _utterance("c", "3", 20)]
        self.assertEqual(scoring.batches(utterances, 2), [[1, 2], [0]])

    def test_decode_corpus_restores_order_and_times_each_call(self):
        utterances = [_utterance("a", "1", 16000), _utterance("b", "2", 32000)]
        texts, timings = scoring.decode_corpus(lambda clips: [str(len(clip)) for clip in clips], utterances, 1)
        self.assertEqual(texts, ["16000", "32000"])
        self.assertEqual([timing[0] for timing in timings], [2.0, 1.0])

    def test_rtf_skips_the_warm_up_call(self):
        self.assertAlmostEqual(scoring.rtf([(1.0, 9.0), (2.0, 1.0), (2.0, 1.0)]), 0.5)
        self.assertAlmostEqual(scoring.rtf([(4.0, 2.0)]), 0.5)
        self.assertIsNone(scoring.rtf([]))

    def test_longform_metrics_count_dropped_spans_and_median_onset(self):
        truth = [(1.0, 3.0, "bună ziua"), (5.0, 7.0, "ce faci"), (9.0, 10.0, "pa")]
        cues = [(1.2, 2.9, "bună ziua"), (5.6, 6.8, "ce faci")]
        metrics = scoring.longform_metrics(truth, cues)
        self.assertEqual(metrics["dropped_spans"], 1)
        self.assertAlmostEqual(metrics["median_onset_error_s"], 0.4)
        self.assertAlmostEqual(metrics["wer"], 0.2)

    def test_longform_metrics_without_cues(self):
        metrics = scoring.longform_metrics([(0.0, 1.0, "da")], [])
        self.assertEqual(metrics["dropped_spans"], 1)
        self.assertIsNone(metrics["median_onset_error_s"])

    def test_add_deltas_compares_with_whisper_and_drops_vectors(self):
        whisper = {"engine": "whisper", "corpus": "c", "wer": 0.5, "_errors": [(1, 2), (1, 2)], "_clusters": ["1", "2"]}
        canary = {"engine": "canary", "corpus": "c", "wer": 0.0, "_errors": [(0, 2), (0, 2)], "_clusters": ["1", "2"]}
        with patch.object(scoring.asr_metrics, "paired_cluster_bootstrap", return_value=(-0.6, -0.4)) as bootstrap:
            scoring.add_deltas([whisper, canary])
        bootstrap.assert_called_once_with([0, 0], [1, 1], [2, 2], ["1", "2"])
        self.assertEqual(canary["delta_wer_vs_whisper_ci95"], [-0.6, -0.4])
        self.assertAlmostEqual(canary["delta_wer_vs_whisper"], -0.5)
        self.assertNotIn("delta_wer_vs_whisper", whisper)
        self.assertNotIn("_errors", canary)


class TestData(unittest.TestCase):
    """Dataset parsing and utterance selection."""

    def test_iso_code_and_selection_helpers(self):
        utterances = [_utterance("a", "1", 16000), _utterance("b", "1", 16000), _utterance("c", "2", 16000)]
        self.assertEqual(data.iso_code("ro_ro"), "ro")
        self.assertEqual([u.uid for u in data.take_seconds(utterances, 1.5)], ["a", "b"])
        self.assertEqual([u.uid for u in data.one_per_sentence(utterances)], ["a", "c"])
        self.assertEqual(data.transcode_path(os.path.join("out", "x.wav")), os.path.join("out", "x_asr16k.wav"))

    def test_tar_members_are_read_in_memory_only(self):
        with tempfile.TemporaryDirectory() as folder:
            tar_path = os.path.join(folder, "test.tar.gz")
            with tarfile.open(tar_path, "w:gz") as archive:
                for name in ("test/a.wav", "test/b.wav"):
                    info = tarfile.TarInfo(name)
                    info.size = 4
                    archive.addfile(info, io.BytesIO(b"RIFF"))
            with patch.object(data, "decode_wav", side_effect=lambda payload: payload) as decode:
                found = data._tar_audio(tar_path, {"b.wav"})
            self.assertEqual(found, {"b.wav": b"RIFF"})
            decode.assert_called_once_with(b"RIFF")
            self.assertEqual(sorted(os.listdir(folder)), ["test.tar.gz"])

    def test_voxpopuli_rows_skip_missing_transcripts_and_stop_at_the_limit(self):
        batch = MagicMock()
        batch.to_pylist.return_value = [{"raw_text": "unu"}, {"raw_text": "  "}, {"raw_text": None}, {"raw_text": "doi"}]
        parquet = MagicMock()
        parquet.ParquetFile.return_value.iter_batches.return_value = [batch, batch]
        with patch.object(data, "_module", return_value=parquet):
            rows = data._parquet_rows("shard.parquet", 1)
        self.assertEqual(rows, [{"raw_text": "unu"}])


class TestEngines(unittest.TestCase):
    """Engine-independent helpers."""

    def test_supported_languages_and_batch_sizes(self):
        manager = MagicMock()
        self.assertTrue(engines.supports("whisper", "ja"))
        self.assertFalse(engines.supports("canary", "ja"))
        self.assertEqual(engines.batch_size(manager, "whisper", 15.0, None), 1)
        self.assertEqual(engines.batch_size(manager, "canary", 15.0, 6), 6)
        with patch.object(engines, "apply_dynamic_asr_batch", return_value=12) as dynamic:
            self.assertEqual(engines.batch_size(manager, "parakeet", 15.0, None), 12)
        dynamic.assert_called_once_with(manager.get_asr.return_value, 15.0)

    def test_nvidia_smi_parsing(self):
        self.assertEqual(engines._row_mib("123, python, 3954 MiB"), 3954)
        self.assertEqual(engines._row_mib("123, python, [N/A]"), 0)
        self.assertEqual(engines._to_mib(["2048"]), 2048)
        self.assertIsNone(engines._to_mib([]))
        self.assertEqual(engines._max(None, 5), 5)
        self.assertEqual(engines._max(7, None), 7)
        self.assertEqual(engines._smi_lines(None, "--help"), [])


class TestReportAndCli(unittest.TestCase):
    """Markdown rendering and command-line parsing."""

    def test_table_formats_rates_and_missing_values(self):
        rendered = report.table(
            [{"engine": "canary", "wer": 0.0661, "rtf": None}], report.BEAM_COLUMNS[:1] + (("WER", "wer", report._pct),)
        )
        self.assertIn("| – | 6.61% |", rendered)

    def test_render_markdown_includes_selected_sections(self):
        meta = {"started": "now", "host": "h", "hardware": {}, "torch": "t", "transformers": "x", "faster_whisper": "f", "git_commit": "c"}
        code_switch = {"passed": True, "vote": "ro", "share": 1.0, "lid_seconds": 2.0, "total_seconds": 330.0}
        code_switch.update({"english_seconds": 31.0, "first_window_language": "en"})
        rendered = report.render_markdown({"meta": meta, "probes": [{"probe": "white", "engine": "canary"}], "code_switch": code_switch})
        self.assertIn("## Non-speech probes", rendered)
        self.assertIn("PASS: the spread vote picked `ro`", rendered)
        self.assertNotIn("## Utterance-level accuracy", rendered)

    def test_write_report_turns_non_finite_values_into_null(self):
        meta = {"started": "now", "host": "h", "hardware": {}, "torch": "t", "transformers": "x", "faster_whisper": "f", "git_commit": "c"}
        rows = [{"engine": "canary", "wer": float("nan"), "rtf": float("inf"), "ci": (float("-inf"), 0.5)}]
        with tempfile.TemporaryDirectory() as folder:
            json_path, _md_path = report.write_report({"meta": meta, "utterance": rows}, folder)
            with open(json_path, encoding="utf-8") as handle:
                saved = json.load(handle)
        self.assertEqual(saved["utterance"], [{"engine": "canary", "wer": None, "rtf": None, "ci": [None, 0.5]}])

    def test_parse_args_defaults_and_bare_sweeps(self):
        args = asr_benchmark.parse_args(["--segment-sweep", "--canary-beams", "--longform"])
        self.assertEqual(args.engines, ["whisper", "canary", "parakeet"])
        self.assertEqual(args.langs, ["ro_ro"])
        self.assertEqual(args.segment_sweep, [8.0, 12.0, 15.0, 20.0])
        self.assertEqual(args.canary_beams, [1, 4])
        self.assertEqual(args.longform, 30.0)

    def test_parse_args_rejects_unknown_engines(self):
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            asr_benchmark.parse_args(["--engines", "whisper,nemo"])

    def test_sections_follow_the_flags(self):
        args = asr_benchmark.parse_args(["--langs", "", "--probes", "--code-switch"])
        selected = [key for key, enabled, _runner in asr_benchmark.SECTIONS if enabled(args)]
        self.assertEqual(selected, ["probes", "code_switch"])


if __name__ == "__main__":
    unittest.main()
