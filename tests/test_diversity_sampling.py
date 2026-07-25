from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = (
    PROJECT_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
)
if str(UTILS_PATH) not in sys.path:
    sys.path.insert(0, str(UTILS_PATH))


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, UTILS_PATH / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adaptive = _load_module("wbt_adaptive_sampling_for_diversity_tests", "adaptive_sampling.py")
diversity = _load_module("wbt_diversity_sampling_tests", "diversity_sampling.py")


class ClusterBudgetTest(unittest.TestCase):
    def test_floor_size_exponent_and_empty_cluster(self) -> None:
        probability = diversity.build_cluster_target_probability(
            torch.tensor([1, 0, 9]),
            minimum_budget_fraction_of_uniform=0.5,
            cluster_size_exponent=0.5,
        )
        self.assertTrue(torch.all(torch.isfinite(probability)))
        self.assertAlmostEqual(float(probability.sum()), 1.0)
        self.assertEqual(float(probability[1]), 0.0)
        self.assertGreaterEqual(float(probability[0]), 0.25)
        self.assertTrue(
            torch.allclose(
                probability,
                torch.tensor([0.375, 0.0, 0.625], dtype=torch.float64),
            )
        )

    def test_exponent_zero_one_and_single_eligible_cluster(self) -> None:
        uniform_remainder = diversity.cluster_target_probability(
            torch.tensor([1, 9]),
            minimum_budget_fraction_of_uniform=0.2,
            cluster_size_exponent=0.0,
        )
        size_proportional_remainder = diversity.cluster_target_probability(
            torch.tensor([1, 9]),
            minimum_budget_fraction_of_uniform=0.2,
            cluster_size_exponent=1.0,
        )
        self.assertTrue(
            torch.allclose(
                uniform_remainder,
                torch.tensor([0.5, 0.5], dtype=torch.float64),
            )
        )
        self.assertLess(
            float(size_proportional_remainder[0]),
            float(uniform_remainder[0]),
        )
        single = diversity.cluster_target_probability(
            torch.tensor([0, 7, 0]),
            minimum_budget_fraction_of_uniform=0.5,
            cluster_size_exponent=0.5,
        )
        self.assertEqual(single.tolist(), [0.0, 1.0, 0.0])

    def test_invalid_budget_inputs_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one cluster"):
            diversity.build_cluster_target_probability(
                torch.zeros(3),
                minimum_budget_fraction_of_uniform=0.5,
                cluster_size_exponent=0.5,
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            diversity.build_cluster_target_probability(
                torch.ones(3),
                minimum_budget_fraction_of_uniform=1.1,
                cluster_size_exponent=0.5,
            )


class DiversityConstrainedSamplerTest(unittest.TestCase):
    @staticmethod
    def _layout() -> tuple[torch.Tensor, ...]:
        # Neither cluster IDs in motion order nor motion IDs in segment order
        # are contiguous.  This exercises the packed conditional sampler.
        segment_motion_ids = torch.tensor(
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
        )
        segment_start_frames = torch.tensor([0] * 6 + [5] * 6)
        segment_end_frames = torch.tensor([5] * 6 + [10] * 6)
        motion_lengths = torch.tensor([10] * 6)
        motion_cluster_ids = torch.tensor([3, 0, 3, 2, 0, 2])
        return (
            segment_motion_ids,
            segment_start_frames,
            segment_end_frames,
            motion_lengths,
            motion_cluster_ids,
        )

    def _sampler(
        self,
        *,
        seed: int = 42,
        clusters: torch.Tensor | None = None,
        num_clusters: int | None = None,
        motion_cap: float = 0.9,
        motion_eligible: torch.Tensor | None = None,
        segment_eligible: torch.Tensor | None = None,
        warmup: int = 0,
        interval: int = 1,
        motion_mode: str = "learning_gap",
        segment_mode: str = "relative_learning_gap",
    ):
        (
            segment_motion_ids,
            segment_start_frames,
            segment_end_frames,
            motion_lengths,
            default_clusters,
        ) = self._layout()
        clusters = default_clusters if clusters is None else clusters
        return diversity.DiversityConstrainedSampler(
            segment_motion_ids,
            segment_start_frames,
            segment_end_frames,
            motion_lengths,
            motion_cluster_ids=clusters,
            num_clusters=(
                int(clusters.max().item()) + 1
                if num_clusters is None
                else num_clusters
            ),
            motion_eligible_mask=(
                torch.ones(6, dtype=torch.bool)
                if motion_eligible is None
                else motion_eligible
            ),
            segment_eligible_mask=(
                torch.ones(12, dtype=torch.bool)
                if segment_eligible is None
                else segment_eligible
            ),
            motion_mode=motion_mode,
            segment_mode=segment_mode,
            warmup_iterations=warmup,
            probability_update_interval=interval,
            uniform_mix=0.1,
            temperature=1.0,
            under_sampling_weight=0.5,
            motion_probability_cap=motion_cap,
            segment_probability_cap=0.9,
            score_clip=10.0,
            sampler_seed=seed,
            config_hash="fixture",
            minimum_budget_fraction_of_uniform=0.5,
            cluster_size_exponent=0.5,
            cluster_metadata_hash="metadata",
            cluster_profile_sha256="profile",
        )

    @staticmethod
    def _update(
        sampler,
        *,
        iteration: int = 0,
        motion_score: torch.Tensor | None = None,
        segment_score: torch.Tensor | None = None,
    ) -> bool:
        motion_score = (
            torch.tensor([6.0, 0.0, 4.0, 0.0, 2.0, 0.0])
            if motion_score is None
            else motion_score
        )
        segment_score = (
            torch.tensor([0.0] * 6 + [3.0] * 6)
            if segment_score is None
            else segment_score
        )
        return sampler.update_probabilities(
            iteration,
            motion_score=motion_score,
            motion_score_valid=torch.ones(6, dtype=torch.bool),
            segment_score=segment_score,
            segment_score_valid=torch.ones(12, dtype=torch.bool),
            motion_sample_count=torch.zeros(6),
            segment_sample_count=torch.zeros(12),
        )

    def test_factorization_each_layer_and_gap_independent_cluster_budget(self) -> None:
        sampler = self._sampler()
        before = sampler.cluster_probability.clone()
        self.assertTrue(self._update(sampler))
        self.assertTrue(torch.equal(before, sampler.cluster_probability))

        expected_global = (
            sampler.cluster_probability[sampler.motion_cluster_ids]
            * sampler.motion_probability_conditional
        )
        self.assertTrue(
            torch.allclose(sampler.motion_probability, expected_global)
        )
        self.assertAlmostEqual(float(sampler.motion_probability.sum()), 1.0)

        conditional_motion_sums = torch.zeros(
            sampler.num_clusters, dtype=torch.float64
        )
        conditional_motion_sums.scatter_add_(
            0,
            sampler.motion_cluster_ids,
            sampler.motion_probability_conditional,
        )
        self.assertTrue(
            torch.allclose(
                conditional_motion_sums[sampler.eligible_cluster_mask],
                torch.ones(
                    int(sampler.eligible_cluster_mask.sum()),
                    dtype=torch.float64,
                ),
            )
        )
        segment_sums = torch.zeros(sampler.num_motions, dtype=torch.float64)
        segment_sums.scatter_add_(
            0, sampler.segment_motion_ids, sampler.segment_probability
        )
        self.assertTrue(
            torch.allclose(segment_sums, torch.ones_like(segment_sums))
        )
        self.assertAlmostEqual(
            float(sampler.segment_probability_global.sum()), 1.0
        )

        other = self._sampler()
        self._update(
            other,
            motion_score=torch.tensor([-9.0, 8.0, -7.0, 6.0, -5.0, 4.0]),
        )
        self.assertTrue(
            torch.equal(sampler.cluster_probability, other.cluster_probability)
        )
        self.assertFalse(
            torch.allclose(
                sampler.motion_probability_conditional,
                other.motion_probability_conditional,
            )
        )

    def test_motion_and_segment_gap_control_only_their_conditional_layer(self) -> None:
        sampler = self._sampler()
        self._update(sampler)
        # Motions 0 and 2 share cluster 3; motion 0 has the larger G_motion.
        self.assertGreater(
            sampler.motion_probability_conditional[0],
            sampler.motion_probability_conditional[2],
        )
        # Each motion's second segment has the larger signed G_local.
        for motion_id in range(6):
            first = motion_id
            second = motion_id + 6
            self.assertGreater(
                sampler.segment_probability[second],
                sampler.segment_probability[first],
            )

    def test_small_cluster_cap_relaxes_per_cluster_only(self) -> None:
        clusters = torch.tensor([0, 1, 1, 1, 1, 1])
        sampler = self._sampler(
            clusters=clusters, num_clusters=2, motion_cap=0.2
        )
        self.assertEqual(
            sampler.effective_motion_cap_by_cluster.tolist(), [1.0, 0.2]
        )
        self.assertEqual(sampler.conditional_motion_cap_relax_count, 1)
        self._update(sampler)
        self.assertEqual(float(sampler.motion_probability_conditional[0]), 1.0)
        self.assertLessEqual(
            float(sampler.motion_probability_conditional[1:].max()),
            0.2 + 1.0e-9,
        )

    def test_rejected_and_empty_motion_have_zero_probability(self) -> None:
        motion_eligible = torch.ones(6, dtype=torch.bool)
        segment_eligible = torch.ones(12, dtype=torch.bool)
        segment_eligible[[2, 8]] = False  # Motion 2 becomes empty.
        motion_eligible[4] = False
        segment_eligible[[4, 10]] = False
        sampler = self._sampler(
            motion_eligible=motion_eligible,
            segment_eligible=segment_eligible,
        )
        self.assertTrue(sampler.empty_motion_mask[2])
        self._update(sampler)
        self.assertEqual(float(sampler.motion_probability[2]), 0.0)
        self.assertEqual(float(sampler.motion_probability[4]), 0.0)
        self.assertEqual(float(sampler.segment_probability[2]), 0.0)
        self.assertEqual(float(sampler.segment_probability[8]), 0.0)

    def test_noncontiguous_conditional_ordering_is_correct(self) -> None:
        sampler = self._sampler()
        self._update(sampler)
        motion_ids, segment_ids, starts = sampler.sample(10_000)
        self.assertIsNotNone(segment_ids)
        sampled_clusters = sampler.motion_cluster_ids[motion_ids]
        self.assertTrue(torch.all(sampler.eligible_cluster_mask[sampled_clusters]))
        self.assertTrue(
            torch.all(sampler.segment_motion_ids[segment_ids] == motion_ids)
        )
        self.assertTrue(
            torch.all(starts >= sampler.segment_start_frames[segment_ids])
        )
        self.assertTrue(
            torch.all(
                starts
                < torch.minimum(
                    sampler.segment_end_frames[segment_ids],
                    sampler.motion_lengths[motion_ids] - 1,
                )
            )
        )

    def test_diversity_uniform_mode_still_draws_factorized_segment(self) -> None:
        sampler = self._sampler(
            motion_mode="uniform",
            segment_mode="uniform",
        )
        motion_ids, segment_ids, starts = sampler.sample(10_000)
        self.assertIsNotNone(segment_ids)
        self.assertTrue(
            torch.all(sampler.segment_motion_ids[segment_ids] == motion_ids)
        )
        self.assertTrue(
            torch.all(starts >= sampler.segment_start_frames[segment_ids])
        )
        self.assertTrue(
            torch.all(
                starts
                < torch.minimum(
                    sampler.segment_end_frames[segment_ids],
                    sampler.motion_lengths[motion_ids] - 1,
                )
            )
        )
        sampled_second_segment = segment_ids >= 6
        self.assertAlmostEqual(
            float(sampled_second_segment.to(torch.float64).mean()),
            0.5,
            delta=0.025,
        )

    def test_rng_counts_and_assignments_resume_exactly(self) -> None:
        sampler = self._sampler(seed=123)
        self._update(sampler)
        sampler.sample(37)
        state = sampler.state_dict()

        torch.manual_seed(987)
        expected_global = torch.rand(5)
        torch.manual_seed(987)
        expected = sampler.sample(500)
        actual_global = torch.rand(5)
        self.assertTrue(torch.equal(expected_global, actual_global))

        restored = self._sampler(seed=123)
        restored.load_state_dict(state)
        actual = restored.sample(500)
        self.assertTrue(
            all(torch.equal(left, right) for left, right in zip(expected, actual))
        )
        self.assertTrue(
            torch.equal(sampler.cluster_sample_count, restored.cluster_sample_count)
        )
        self.assertTrue(
            torch.equal(
                sampler.observed_cluster_share,
                restored.observed_cluster_share,
            )
        )

    def test_long_run_covers_small_clusters_and_matches_target(self) -> None:
        clusters = torch.tensor([0, 1, 1, 1, 1, 1])
        sampler = self._sampler(clusters=clusters, num_clusters=3)
        self._update(sampler)
        sampler.sample(100_000)
        observed = sampler.observed_cluster_share
        self.assertGreater(int(sampler.cluster_sample_count[0]), 0)
        self.assertEqual(int(sampler.cluster_sample_count[2]), 0)
        self.assertTrue(
            torch.allclose(
                observed,
                sampler.cluster_probability,
                atol=0.01,
                rtol=0.0,
            )
        )
        metrics = sampler.metrics()
        self.assertEqual(metrics["cluster_coverage"], 1.0)
        self.assertLess(metrics["target_observed_l1"], 0.02)
        self.assertGreater(metrics["cluster_entropy"], 0.0)
        self.assertTrue(
            torch.isfinite(
                torch.tensor(metrics["target_observed_js_divergence"])
            )
        )

    def test_warmup_update_schedule_matches_adaptive_sampler(self) -> None:
        sampler = self._sampler(warmup=3, interval=2)
        initial_motion = sampler.motion_probability.clone()
        self.assertFalse(self._update(sampler, iteration=0))
        self.assertTrue(torch.equal(initial_motion, sampler.motion_probability))
        self.assertTrue(self._update(sampler, iteration=3))
        updated = sampler.motion_probability.clone()
        self.assertFalse(self._update(sampler, iteration=4))
        self.assertTrue(torch.equal(updated, sampler.motion_probability))
        self.assertTrue(self._update(sampler, iteration=5))

    def test_one_cluster_probabilities_equal_m6(self) -> None:
        (
            segment_motion_ids,
            segment_start_frames,
            segment_end_frames,
            motion_lengths,
            _,
        ) = self._layout()
        common = dict(
            motion_eligible_mask=torch.ones(6, dtype=torch.bool),
            segment_eligible_mask=torch.ones(12, dtype=torch.bool),
            motion_mode="learning_gap",
            segment_mode="relative_learning_gap",
            warmup_iterations=0,
            probability_update_interval=1,
            uniform_mix=0.1,
            temperature=1.0,
            under_sampling_weight=0.5,
            motion_probability_cap=0.9,
            segment_probability_cap=0.9,
            score_clip=10.0,
            sampler_seed=42,
            config_hash="fixture",
        )
        m6 = adaptive.HierarchicalAdaptiveSampler(
            segment_motion_ids,
            segment_start_frames,
            segment_end_frames,
            motion_lengths,
            **common,
        )
        m7 = diversity.DiversityConstrainedSampler(
            segment_motion_ids,
            segment_start_frames,
            segment_end_frames,
            motion_lengths,
            motion_cluster_ids=torch.zeros(6, dtype=torch.long),
            num_clusters=1,
            **common,
        )
        update = dict(
            motion_score=torch.tensor([6.0, 0.0, 4.0, 0.0, 2.0, 0.0]),
            motion_score_valid=torch.ones(6, dtype=torch.bool),
            segment_score=torch.tensor([0.0] * 6 + [3.0] * 6),
            segment_score_valid=torch.ones(12, dtype=torch.bool),
            motion_sample_count=torch.zeros(6),
            segment_sample_count=torch.zeros(12),
        )
        self.assertTrue(m6.update_probabilities(0, **update))
        self.assertTrue(m7.update_probabilities(0, **update))
        self.assertTrue(
            torch.allclose(m7.motion_probability, m6.motion_probability)
        )
        self.assertTrue(
            torch.allclose(m7.segment_probability, m6.segment_probability)
        )


if __name__ == "__main__":
    unittest.main()
