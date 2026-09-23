from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPLAY_PATH = PROJECT_ROOT / "scripts" / "rsl_rl" / "replay_joint_gap_lambda_candidates.py"


def _load_replay_module():
    spec = importlib.util.spec_from_file_location("wbt_joint_gap_lambda_replay_test", REPLAY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {REPLAY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class JointGapLambdaReplayTest(unittest.TestCase):
    @staticmethod
    def _strip_timing(payload):
        timing_suffixes = ("_ms",)
        if isinstance(payload, list):
            return [JointGapLambdaReplayTest._strip_timing(item) for item in payload]
        if isinstance(payload, dict):
            return {
                key: JointGapLambdaReplayTest._strip_timing(value)
                for key, value in payload.items()
                if key != "performance" and not any(str(key).endswith(suffix) for suffix in timing_suffixes)
            }
        return payload

    def _fixture(self, root: Path, replay):
        modules = replay._load_utils_modules(PROJECT_ROOT, "wbt_joint_gap_lambda_fixture")
        run_dir = root / "run"
        snapshots_dir = root / "snapshots"
        output_a = root / "out_a"
        run_dir.mkdir()
        snapshots_dir.mkdir()
        manifest = root / "train.txt"
        quality = root / "quality.npz"
        difficulty = root / "difficulty.npz"
        cluster = root / "cluster.npz"
        manifest.write_text("a.npz\nb.npz\nc.npz\n", encoding="utf-8")
        np.savez_compressed(quality, schema_version="fixture")
        np.savez_compressed(difficulty, schema_version="fixture")
        np.savez_compressed(cluster, schema_version="fixture")

        segment_motion_ids = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.long)
        segment_start_frames = torch.tensor([0, 5, 0, 5, 10, 0], dtype=torch.long)
        segment_end_frames = torch.tensor([5, 10, 5, 10, 15, 5], dtype=torch.long)
        motion_lengths = torch.tensor([20, 20, 20], dtype=torch.long)
        motion_cluster_ids = torch.tensor([0, 1, 1], dtype=torch.long)
        motion_eligible = torch.ones(3, dtype=torch.bool)
        segment_eligible = torch.ones(6, dtype=torch.bool)
        motion_score = torch.tensor([0.3, 0.5, 0.4], dtype=torch.float64)
        segment_score = torch.tensor([0.2, 0.9, 0.1, 0.8, 0.4, 0.7], dtype=torch.float64)
        motion_count = torch.tensor([20, 21, 22], dtype=torch.float64)
        segment_count = torch.tensor([40, 41, 42, 43, 44, 45], dtype=torch.float64)
        sampler = modules["diversity_sampling"].DiversityConstrainedSampler(
            segment_motion_ids,
            segment_start_frames,
            segment_end_frames,
            motion_lengths,
            motion_cluster_ids=motion_cluster_ids,
            motion_eligible_mask=motion_eligible,
            segment_eligible_mask=segment_eligible,
            motion_mode="raw_error",
            segment_mode="raw_error",
            warmup_iterations=0,
            probability_update_interval=1,
            uniform_mix=0.15,
            temperature=1.0,
            under_sampling_weight=0.25,
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
        self.assertTrue(
            sampler.update_probabilities(
                100,
                motion_score=motion_score,
                motion_score_valid=torch.ones(3, dtype=torch.bool),
                segment_score=segment_score,
                segment_score_valid=torch.ones(6, dtype=torch.bool),
                motion_sample_count=motion_count,
                segment_sample_count=segment_count,
            )
        )
        checkpoint_path = run_dir / "model_100.pt"
        checkpoint = {
            "infos": {
                "sampling_state": {
                    "research_config": {
                        "online_learning": {
                            "warmup_iterations": 0,
                            "probability_update_interval": 1,
                            "uniform_mix": 0.15,
                            "temperature": 1.0,
                            "under_sampling_weight": 0.25,
                            "motion_probability_cap": 1.0,
                            "segment_probability_cap": 1.0,
                            "score_clip": 10.0,
                            "sampler_seed": 42,
                            "min_segment_observations": 1,
                            "min_bin_valid_segments": 1,
                            "sigma_floor": 0.1,
                            "gap_clip": 5.0,
                            "bin_observation_weighted": False,
                        }
                    },
                    "online_learning": {
                        "sampler": sampler.state_dict(),
                        "statistics": {
                            "motion_sample_count": motion_count.to(torch.long),
                            "segment_sample_count": segment_count.to(torch.long),
                        },
                        "segment_error_result": {
                            "error": segment_score,
                            "valid": torch.ones(6, dtype=torch.bool),
                        },
                        "motion_error_result": {
                            "error": motion_score,
                            "valid": torch.ones(3, dtype=torch.bool),
                        },
                    },
                }
            }
        }
        torch.save(checkpoint, checkpoint_path)
        checkpoint_sha = replay._sha256_file(checkpoint_path)
        joint_names = np.asarray([f"joint_{index}" for index in range(6)], dtype=str)
        joint_error = np.asarray(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [1.5, 1.4, 1.3, 1.2, 1.1, 1.0],
                [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                [1.7, 1.6, 1.5, 1.4, 1.3, 1.2],
                [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
                [1.9, 1.8, 1.7, 1.6, 1.5, 1.4],
            ],
            dtype=np.float32,
        )
        snapshot_path = snapshots_dir / "m7raw_ckpt_100_joint_snapshot.npz"
        np.savez_compressed(
            snapshot_path,
            schema_version="wbt.joint_gap.proxy_snapshot.v1",
            proxy_statistic="per_segment_mean_absolute_joint_error",
            proxy_limitation="unit-test frozen-policy proxy",
            checkpoint_iteration=100,
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha,
            train_manifest_sha256=replay._sha256_file(manifest),
            segment_joint_error=joint_error,
            segment_observation_count=np.asarray([40, 40, 40, 40, 40, 40], dtype=np.int64),
            eligible_mask=np.ones(6, dtype=bool),
            observed_mask=np.ones(6, dtype=bool),
            difficulty_bin=np.asarray([0, 0, 0, 0, 0, 0], dtype=np.int64),
            motion_cluster_id=motion_cluster_ids.numpy(),
            motion_segment_offsets=np.asarray([0, 2, 5, 6], dtype=np.int64),
            motion_id=segment_motion_ids.numpy(),
            local_segment_id=np.asarray([0, 1, 0, 1, 2, 0], dtype=np.int64),
            start_frame=segment_start_frames.numpy(),
            end_frame_exclusive=segment_end_frames.numpy(),
            joint_names=joint_names,
            joint_mapping_hash="fixture-hash",
            category=np.asarray(["cat_a", "cat_b", "cat_c"], dtype=str),
            source_group=np.asarray(["src_a", "src_b", "src_c"], dtype=str),
        )
        config = replay.ReplayConfig(
            project_root=PROJECT_ROOT,
            run_dir=run_dir,
            snapshots_dir=snapshots_dir,
            output_dir=output_a,
            checkpoints="100",
            checkpoint_pattern="model_*.pt",
            train_manifest=manifest,
            quality_metadata=quality,
            difficulty_metadata=difficulty,
            cluster_metadata=cluster,
            write_freeze_doc=False,
        )
        return config

    def test_fixed_candidates_identity_and_deterministic_selection(self) -> None:
        replay = _load_replay_module()
        self.assertEqual(replay.LAMBDA_CANDIDATES, (0.0, 0.025, 0.050, 0.100))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._fixture(root, replay)
            first = replay.run_replay(config)
            second_config = replay.ReplayConfig(**{**config.__dict__, "output_dir": root / "out_b"})
            second = replay.run_replay(second_config)
            self.assertEqual(first["selection"], second["selection"])
            self.assertEqual(self._strip_timing(first["candidate_rows"]), self._strip_timing(second["candidate_rows"]))
            zero_row = next(row for row in first["candidate_rows"] if row["lambda"] == "0.000")
            self.assertEqual(float(zero_row["global_tv"]), 0.0)
            self.assertEqual(int(zero_row["cluster_exact"]), 1)
            self.assertEqual(int(zero_row["motion_exact"]), 1)
            selected = first["selection"]["selected_lambda"]
            self.assertIn(selected, (0.025, 0.050, 0.100))
            manifest_a = json.loads((root / "out_a" / "lambda_replay_manifest.json").read_text())
            manifest_b = json.loads((root / "out_b" / "lambda_replay_manifest.json").read_text())
            self.assertEqual(self._strip_timing(manifest_a), self._strip_timing(manifest_b))


if __name__ == "__main__":
    unittest.main()
