from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "adaptive_sampling.py"
)
SPEC = importlib.util.spec_from_file_location("wbt_adaptive_sampling_tests", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
sampling = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sampling
SPEC.loader.exec_module(sampling)


class ProbabilityTest(unittest.TestCase):
    def test_probability_is_finite_masked_mixed_and_under_sampling_aware(self) -> None:
        result = sampling.build_probability(
            torch.tensor([0.0, 0.0, 100.0]),
            torch.tensor([True, True, True]),
            torch.tensor([True, True, False]),
            torch.tensor([100, 0, 0]),
            uniform_mix=0.2,
            temperature=1.0,
            under_sampling_weight=2.0,
            probability_cap=0.9,
            score_clip=10,
        )
        probability = result.probability
        self.assertTrue(torch.all(torch.isfinite(probability)))
        self.assertTrue(torch.all(probability >= 0))
        self.assertAlmostEqual(probability.sum().item(), 1.0)
        self.assertEqual(probability[2].item(), 0.0)
        self.assertGreater(probability[1], probability[0])
        self.assertGreaterEqual(probability[0], 0.1)

    def test_nonfinite_score_falls_back_to_uniform(self) -> None:
        result = sampling.build_probability(
            torch.tensor([1.0, float("nan")]),
            torch.tensor([True, True]),
            torch.tensor([True, True]),
            torch.zeros(2),
            uniform_mix=0.15,
            temperature=1,
            under_sampling_weight=0,
            probability_cap=1,
            score_clip=10,
        )
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, "nonfinite_score")
        self.assertEqual(result.probability.tolist(), [0.5, 0.5])

    def test_water_filling_cap_is_respected_after_normalization(self) -> None:
        capped = sampling.capped_simplex(
            torch.tensor([0.95, 0.03, 0.02]), torch.ones(3, dtype=torch.bool), 0.6
        )
        self.assertAlmostEqual(capped.sum().item(), 1.0)
        self.assertLessEqual(capped.max().item(), 0.6 + 1e-12)
        self.assertAlmostEqual(capped[0].item(), 0.6)
        self.assertAlmostEqual(capped[1].item(), 0.24)
        self.assertAlmostEqual(capped[2].item(), 0.16)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            sampling.capped_simplex(torch.ones(2), torch.ones(2, dtype=torch.bool), 0.4)

    def test_grouped_probability_masks_reject_and_empty_motion(self) -> None:
        probability, nonempty, _ = sampling.grouped_probability(
            torch.tensor([1.0, 9.0, 3.0, 7.0]),
            torch.ones(4, dtype=torch.bool),
            torch.tensor([True, False, True, False]),
            torch.zeros(4),
            torch.tensor([0, 0, 1, 2]),
            num_groups=3,
            uniform_mix=0.15,
            temperature=1,
            under_sampling_weight=0,
            probability_cap=1,
            score_clip=10,
        )
        self.assertEqual(nonempty.tolist(), [True, True, False])
        self.assertEqual(probability.tolist(), [1.0, 0.0, 1.0, 0.0])

    def test_forced_uniform_layer_does_not_report_score_fallback(self) -> None:
        probability, _, fallback_count = sampling.grouped_probability(
            torch.zeros(4),
            torch.zeros(4, dtype=torch.bool),
            torch.ones(4, dtype=torch.bool),
            torch.zeros(4),
            torch.tensor([0, 0, 1, 1]),
            num_groups=2,
            uniform_mix=0.15,
            temperature=1,
            under_sampling_weight=0,
            probability_cap=1,
            score_clip=10,
            force_uniform=True,
        )
        self.assertEqual(probability.tolist(), [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(fallback_count, 0)

    def test_grouped_cap_fails_fast_when_normalization_is_impossible(self) -> None:
        with self.assertRaisesRegex(ValueError, "infeasible"):
            sampling.grouped_probability(
                torch.tensor([1.0]),
                torch.tensor([True]),
                torch.tensor([True]),
                torch.zeros(1),
                torch.tensor([0]),
                num_groups=1,
                uniform_mix=0.15,
                temperature=1,
                under_sampling_weight=0,
                probability_cap=0.8,
                score_clip=10,
            )

    def test_signed_local_scores_preserve_below_median_differences(self) -> None:
        scores = torch.tensor([-2.0, -1.0, 0.0, 1.0])
        probability, _, _ = sampling.grouped_probability(
            scores,
            torch.ones(4, dtype=torch.bool),
            torch.ones(4, dtype=torch.bool),
            torch.zeros(4),
            torch.zeros(4, dtype=torch.long),
            num_groups=1,
            uniform_mix=0.0,
            temperature=1.0,
            under_sampling_weight=0.0,
            probability_cap=1.0,
            score_clip=10.0,
        )
        self.assertTrue(
            torch.allclose(probability, torch.softmax(scores.to(torch.float64), dim=0))
        )


class HierarchicalSamplerTest(unittest.TestCase):
    def _sampler(self, motion_mode: str, segment_mode: str, *, warmup: int = 0, interval: int = 1):
        return sampling.HierarchicalAdaptiveSampler(
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 5, 0, 5]),
            torch.tensor([5, 10, 5, 10]),
            torch.tensor([10, 10]),
            motion_eligible_mask=torch.tensor([True, True]),
            segment_eligible_mask=torch.tensor([True, True, True, True]),
            motion_mode=motion_mode,
            segment_mode=segment_mode,
            warmup_iterations=warmup,
            probability_update_interval=interval,
            uniform_mix=0.1,
            temperature=1,
            under_sampling_weight=0,
            motion_probability_cap=0.9,
            segment_probability_cap=0.9,
            score_clip=10,
            sampler_seed=42,
            config_hash="fixture",
        )

    def _update(self, sampler, iteration: int = 0):
        return sampler.update_probabilities(
            iteration,
            motion_score=torch.tensor([0.0, 4.0]),
            motion_score_valid=torch.tensor([True, True]),
            segment_score=torch.tensor([0.0, 4.0, 0.0, 4.0]),
            segment_score_valid=torch.ones(4, dtype=torch.bool),
            motion_sample_count=torch.zeros(2),
            segment_sample_count=torch.zeros(4),
        )

    def test_m2_m3_m4_change_only_the_configured_layers(self) -> None:
        m2 = self._sampler("raw_error", "uniform")
        self._update(m2)
        self.assertGreater(m2.motion_probability[1], m2.motion_probability[0])
        self.assertEqual(m2.segment_probability.tolist(), [0.5, 0.5, 0.5, 0.5])

        m3 = self._sampler("uniform", "raw_error")
        self._update(m3)
        self.assertEqual(m3.motion_probability.tolist(), [0.5, 0.5])
        self.assertGreater(m3.segment_probability[1], m3.segment_probability[0])
        self.assertGreater(m3.segment_probability[3], m3.segment_probability[2])

        m4 = self._sampler("raw_error", "raw_error")
        self._update(m4)
        self.assertGreater(m4.motion_probability[1], m4.motion_probability[0])
        self.assertGreater(m4.segment_probability[1], m4.segment_probability[0])

    def test_warmup_and_update_interval(self) -> None:
        sampler = self._sampler("raw_error", "raw_error", warmup=3, interval=2)
        self.assertFalse(self._update(sampler, 0))
        self.assertEqual(sampler.motion_probability.tolist(), [0.5, 0.5])
        self.assertTrue(self._update(sampler, 3))
        first = sampler.motion_probability.clone()
        self.assertFalse(self._update(sampler, 4))
        self.assertTrue(torch.equal(first, sampler.motion_probability))
        self.assertTrue(self._update(sampler, 5))

    def test_resume_at_warmup_boundary_does_not_refresh_twice(self) -> None:
        sampler = self._sampler("raw_error", "raw_error", warmup=3, interval=2)
        self.assertTrue(self._update(sampler, 3))
        restored = self._sampler("raw_error", "raw_error", warmup=3, interval=2)
        restored.load_state_dict(sampler.state_dict())
        self.assertFalse(self._update(restored, 3))
        self.assertFalse(self._update(restored, 4))
        self.assertTrue(self._update(restored, 5))

    def test_dedicated_rng_state_round_trip_and_global_rng_is_untouched(self) -> None:
        sampler = self._sampler("raw_error", "raw_error")
        self._update(sampler)
        torch.manual_seed(123)
        expected_global = torch.rand(5)
        torch.manual_seed(123)
        state = sampler.state_dict()
        first = sampler.sample(50)
        actual_global = torch.rand(5)
        self.assertTrue(torch.equal(expected_global, actual_global))

        restored = self._sampler("raw_error", "raw_error")
        restored.load_state_dict(state)
        second = restored.sample(50)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))

    def test_checkpoint_rejects_probability_above_effective_cap(self) -> None:
        sampler = self._sampler("raw_error", "raw_error")
        self._update(sampler)
        state = sampler.state_dict()
        state["motion_probability"] = torch.tensor([0.95, 0.05], dtype=torch.float64)
        restored = self._sampler("raw_error", "raw_error")
        with self.assertRaisesRegex(ValueError, "configured cap"):
            restored.load_state_dict(state)

    def test_global_bin_mode_samples_across_motions_and_honors_mask(self) -> None:
        sampler = self._sampler("uniform", "global_bin_raw_error")
        self._update(sampler)
        self.assertGreater(sampler.global_segment_probability[3], sampler.global_segment_probability[0])
        motion_ids, segment_ids, starts = sampler.sample(100)
        self.assertIsNotNone(segment_ids)
        self.assertTrue(torch.all(sampler.segment_motion_ids[segment_ids] == motion_ids))
        self.assertTrue(torch.all(starts >= sampler.segment_start_frames[segment_ids]))


if __name__ == "__main__":
    unittest.main()
