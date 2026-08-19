from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = PROJECT_ROOT / "scripts" / "rsl_rl" / "preflight_joint_gap_probability.py"


def _load_preflight_module():
    spec = importlib.util.spec_from_file_location("wbt_joint_gap_probability_preflight_test", PREFLIGHT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {PREFLIGHT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class JointGapProbabilityPreflightTest(unittest.TestCase):
    def _write_checkpoint(self, path: Path, preflight, *, corrected: bool = True) -> None:
        modules = preflight._load_utils_modules(PROJECT_ROOT, "wbt_joint_gap_preflight_fixture")
        segment_motion_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        segment_start_frames = torch.tensor([0, 5, 0, 5], dtype=torch.long)
        segment_end_frames = torch.tensor([5, 10, 5, 10], dtype=torch.long)
        motion_lengths = torch.tensor([20, 20], dtype=torch.long)
        motion_cluster_ids = torch.tensor([0, 1], dtype=torch.long)
        motion_eligible = torch.ones(2, dtype=torch.bool)
        segment_eligible = torch.ones(4, dtype=torch.bool)
        sampler = modules["diversity_sampling"].DiversityConstrainedSampler(
            segment_motion_ids,
            segment_start_frames,
            segment_end_frames,
            motion_lengths,
            motion_cluster_ids=motion_cluster_ids,
            motion_eligible_mask=motion_eligible,
            segment_eligible_mask=segment_eligible,
            motion_mode="raw_error",
            segment_mode="raw_error_joint_gap",
            warmup_iterations=1000,
            probability_update_interval=50,
            uniform_mix=0.0,
            temperature=1.0,
            under_sampling_weight=0.0,
            motion_probability_cap=1.0,
            segment_probability_cap=1.0,
            score_clip=10.0,
            sampler_seed=42,
            config_hash="fixture",
            num_clusters=2,
            minimum_budget_fraction_of_uniform=0.5,
            cluster_size_exponent=0.5,
            budget_mode="sqrt_size_with_floor",
            cluster_metadata_hash="cluster",
            cluster_profile_sha256="profile",
            cluster_schema_version="fixture",
            device="cpu",
        )
        raw_segment = torch.tensor([0.2, 1.0, 0.2, 1.0], dtype=torch.float64)
        correction = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64) if corrected else torch.zeros(4, dtype=torch.float64)
        corrected_segment = raw_segment * (1.0 + 0.025 * correction)
        checkpoint = {
            "iter": 498,
            "infos": {
                "sampling_state": {
                    "research_config": {
                        "method_name": "M7-JGap",
                        "online_learning": {
                            "warmup_iterations": 1000,
                            "probability_update_interval": 50,
                            "uniform_mix": 0.0,
                            "temperature": 1.0,
                            "under_sampling_weight": 0.0,
                            "motion_probability_cap": 1.0,
                            "segment_probability_cap": 1.0,
                            "score_clip": 10.0,
                            "sampler_seed": 42,
                            "joint_gap": {"lambda_joint": 0.025},
                        },
                    },
                    "online_learning": {
                        "bin_calibration": None,
                        "gap_result": None,
                        "sampler": sampler.state_dict(),
                        "statistics": {
                            "motion_sample_count": torch.tensor([10, 10], dtype=torch.long),
                            "segment_sample_count": torch.tensor([10, 10, 10, 10], dtype=torch.long),
                        },
                        "motion_error_result": {
                            "error": torch.tensor([0.4, 0.4], dtype=torch.float64),
                            "valid": torch.ones(2, dtype=torch.bool),
                        },
                        "segment_error_result": {
                            "error": raw_segment,
                            "valid": torch.ones(4, dtype=torch.bool),
                        },
                        "joint_gap": {
                            "joint_gap_result": {
                                "corrected_priority": corrected_segment,
                                "correction": correction,
                                "segment_valid": torch.ones(4, dtype=torch.bool),
                            }
                        },
                    },
                }
            },
        }
        torch.save(checkpoint, path)

    def test_preflight_passes_when_jgap_changes_only_segment_distribution(self) -> None:
        preflight = _load_preflight_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model_498.pt"
            self._write_checkpoint(checkpoint, preflight, corrected=True)
            report = preflight.run_preflight(
                checkpoint=checkpoint,
                output_dir=root / "out",
                project_root=PROJECT_ROOT,
            )
            self.assertEqual(report["status"], "PASS")
            self.assertGreater(report["raw_rebuild_probability_updates"], 0)
            self.assertGreater(report["jgap_rebuild_probability_updates"], 0)
            self.assertEqual(report["max_abs_diff_cluster_probability"], 0.0)
            self.assertEqual(report["max_abs_diff_motion_probability_conditional"], 0.0)
            self.assertGreater(report["max_abs_diff_segment_probability_conditional"], 0.0)
            self.assertNotEqual(report["raw_segment_probability_hash"], report["jgap_segment_probability_hash"])

    def test_preflight_blocks_when_segment_distribution_is_unchanged(self) -> None:
        preflight = _load_preflight_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model_498.pt"
            self._write_checkpoint(checkpoint, preflight, corrected=False)
            report = preflight.run_preflight(
                checkpoint=checkpoint,
                output_dir=root / "out",
                project_root=PROJECT_ROOT,
            )
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(report["changed_conditional_segments"], 0)


if __name__ == "__main__":
    unittest.main()
