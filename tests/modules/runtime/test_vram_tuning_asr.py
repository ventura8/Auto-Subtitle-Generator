"""ASR (Canary / Parakeet) VRAM fit checks and batch sizing."""

import unittest
from unittest.mock import patch

from modules.runtime import vram_tuning as vt

CANARY = "nvidia/canary-1b-v2"
PARAKEET = "nvidia/parakeet-tdt-0.6b-v3"


class TestAsrTables(unittest.TestCase):
    def test_weights_round_up_the_measured_bf16_footprint(self):
        # Measured on an RTX 3080 Laptop: 1.83 / 1.18 GB.
        self.assertGreaterEqual(vt.ASR_WEIGHT_GB[CANARY], 1.83)
        self.assertGreaterEqual(vt.ASR_WEIGHT_GB[PARAKEET], 1.18)

    def test_every_profile_has_a_cap_and_caps_shrink_with_the_profile(self):
        caps = [vt.ASR_BATCH_CAPS[name] for name in ("ULTRA", "HIGH", "MID", "LOW", "CPU_ONLY")]
        self.assertEqual(caps, sorted(caps, reverse=True))
        self.assertEqual(set(vt.ASR_PER_SECOND_GB), set(vt.ASR_WEIGHT_GB))


class TestAsrBatchCap(unittest.TestCase):
    def test_caps_per_profile(self):
        expectations = {"ULTRA": 32, "HIGH": 16, "MID": 8, "LOW": 4, "CPU_ONLY": 4}
        for profile, expected in expectations.items():
            with self.subTest(profile=profile):
                self.assertEqual(vt.asr_batch_cap(profile), expected)

    def test_unknown_profile_gets_the_smallest_cap(self):
        self.assertEqual(vt.asr_batch_cap("WEIRD"), vt.ASR_BATCH_CAP_DEFAULT)
        self.assertEqual(vt.asr_batch_cap(None), 4)


class TestAsrFitsGpu(unittest.TestCase):
    def test_weights_plus_activation_floor_must_fit(self):
        self.assertTrue(vt.asr_fits_gpu(CANARY, 2.9))
        self.assertFalse(vt.asr_fits_gpu(CANARY, 2.8))
        self.assertTrue(vt.asr_fits_gpu(PARAKEET, 2.2))
        self.assertFalse(vt.asr_fits_gpu(PARAKEET, 2.1))

    def test_unknown_model_is_charged_as_the_larger_model(self):
        self.assertFalse(vt.asr_fits_gpu("someone/other-asr", 2.8))
        self.assertTrue(vt.asr_fits_gpu("someone/other-asr", 2.9))

    def test_no_free_memory_never_fits(self):
        self.assertFalse(vt.asr_fits_gpu(PARAKEET, 0.0))


class TestAsrBatchSize(unittest.TestCase):
    def test_sized_from_free_memory_and_segment_length(self):
        # 2 GB * 0.8 / (0.010 GB/s * 15 s) = 10.67 -> 10
        self.assertEqual(vt.asr_batch_size(2.0, CANARY, 15.0, 32), 10)
        # 2 GB * 0.8 / (0.006 GB/s * 15 s) = 17.78 -> 17
        self.assertEqual(vt.asr_batch_size(2.0, PARAKEET, 15.0, 32), 17)

    def test_respects_the_cap(self):
        self.assertEqual(vt.asr_batch_size(40.0, CANARY, 15.0, 16), 16)

    def test_respects_the_floor(self):
        self.assertEqual(vt.asr_batch_size(0.0, CANARY, 15.0, 16), 1)
        self.assertEqual(vt.asr_batch_size(-3.0, CANARY, 15.0, 16, floor=2), 2)

    def test_unknown_model_uses_the_default_cost(self):
        self.assertEqual(vt.asr_batch_size(2.0, "someone/other-asr", 15.0, 32), vt.asr_batch_size(2.0, CANARY, 15.0, 32))

    def test_short_segments_are_charged_at_least_one_second(self):
        self.assertEqual(vt.asr_batch_size(0.2, CANARY, 0.0, 32), 16)

    def test_zero_cost_returns_the_cap(self):
        with patch.dict(vt.ASR_PER_SECOND_GB, {CANARY: 0.0}):
            self.assertEqual(vt.asr_batch_size(1.0, CANARY, 15.0, 12), 12)


class TestDescribeAsrBudget(unittest.TestCase):
    def test_known_model(self):
        line = vt.describe_asr_budget(8, 5.25, CANARY, 16, 16)
        self.assertEqual(
            line,
            "[VRAM] ASR nvidia/canary-1b-v2: 8 GB card, 5.2 GB free after load (1.9 GB weights) -> batch 16 (profile cap 16)",
        )

    def test_unknown_model(self):
        line = vt.describe_asr_budget(8, 5.0, "someone/other-asr", 4, 8)
        self.assertIn("unknown weights", line)
        self.assertIn("batch 4 (profile cap 8)", line)


if __name__ == "__main__":
    unittest.main()
