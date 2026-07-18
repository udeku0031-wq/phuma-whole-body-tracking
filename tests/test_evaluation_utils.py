from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "rsl_rl"))

import evaluation_utils as utils  # noqa: E402


class EvaluationUtilsTest(unittest.TestCase):
    def test_parse_checkpoint_iteration(self):
        self.assertEqual(utils.parse_checkpoint_iteration("model_10000.pt"), 10000)
        self.assertEqual(utils.parse_checkpoint_iteration(Path("/x/model_33999.pt")), 33999)
        self.assertIsNone(utils.parse_checkpoint_iteration("policy.onnx"))

    def test_select_nearest_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for name in ("model_9999.pt", "model_20500.pt", "model_30000.pt", "model_33999.pt"):
                (run_dir / name).write_text("")
            selected = utils.select_checkpoints(utils.list_checkpoints(run_dir), "10000,20000,30000,final")
            self.assertEqual([path.name for path in selected], ["model_9999.pt", "model_20500.pt", "model_30000.pt", "model_33999.pt"])

    def test_best_checkpoint_rules_choose_macro_then_body_then_earlier(self):
        rows = [
            {
                "checkpoint": "model_10000.pt",
                "iteration": 10000,
                "macro_success_rate": "0.800",
                "micro_success_rate": "0.820",
                "mean_completion_ratio": "0.900",
                "mean_body_position_error_m": "0.120",
            },
            {
                "checkpoint": "model_20000.pt",
                "iteration": 20000,
                "macro_success_rate": "0.801",
                "micro_success_rate": "0.821",
                "mean_completion_ratio": "0.900",
                "mean_body_position_error_m": "0.100",
            },
            {
                "checkpoint": "model_30000.pt",
                "iteration": 30000,
                "macro_success_rate": "0.790",
                "micro_success_rate": "0.990",
                "mean_completion_ratio": "0.990",
                "mean_body_position_error_m": "0.050",
            },
        ]
        self.assertEqual(utils.choose_best_checkpoint(rows)["checkpoint"], "model_20000.pt")

    def test_test_manifest_helpers(self):
        self.assertTrue(utils.is_final_test_manifest("PHUMA_wbt_motions/manifests/splits_v1/test.txt"))
        self.assertTrue(utils.should_reject_final_test("test.txt", confirmed=False))
        self.assertFalse(utils.should_reject_final_test("test.txt", confirmed=True))
        self.assertFalse(utils.is_final_test_manifest("validation.txt"))

    def test_completion_ratio_clamped(self):
        self.assertEqual(utils.clamp_completion_ratio(12, 10), 1.0)
        self.assertEqual(utils.clamp_completion_ratio(-1, 10), 0.0)
        self.assertEqual(utils.clamp_completion_ratio(5, 10), 0.5)

    def test_duplicate_and_missing_detection(self):
        expected = ["a.npz", "b.npz", "c.npz"]
        rows = [{"motion_path": "a.npz"}, {"motion_path": "a.npz"}, {"motion_path": "extra.npz"}]
        result = utils.validate_results_exactly_once(rows, expected)
        self.assertFalse(result["ok"])
        self.assertEqual(result["duplicates"], ["a.npz"])
        self.assertEqual(result["missing"], ["b.npz", "c.npz"])
        self.assertEqual(result["unexpected"], ["extra.npz"])

    def test_resume_skip_set_from_existing_rows(self):
        rows = [{"motion_path": "a.npz"}, {"motion_path": "c.npz"}]
        expected = ["a.npz", "b.npz", "c.npz"]
        completed = {row["motion_path"] for row in rows}
        pending = [path for path in expected if path not in completed]
        self.assertEqual(pending, ["b.npz"])

    def test_category_macro_success_rate(self):
        rows = [
            self._row("a", "cat_a", "g1", 1, 1.0),
            self._row("b", "cat_a", "g1", 0, 0.5),
            self._row("c", "cat_b", "g2", 1, 1.0),
        ]
        summary = utils.summarize_rows(rows, checkpoint="model_1.pt")
        self.assertAlmostEqual(summary["micro_success_rate"], 2 / 3)
        self.assertAlmostEqual(summary["macro_success_rate"], 0.75)

    def test_source_group_aggregation(self):
        rows = [
            self._row("a", "cat_a", "group_1", 1, 1.0),
            self._row("b", "cat_a", "group_1", 0, 0.5),
            self._row("c", "cat_b", "group_2", 1, 1.0),
        ]
        grouped = utils.group_summary(rows, "source_group")
        by_name = {row["source_group"]: row for row in grouped}
        self.assertEqual(by_name["group_1"]["num_motions"], 2)
        self.assertEqual(by_name["group_1"]["num_success"], 1)
        self.assertEqual(by_name["group_1"]["success_rate"], "0.500000")

    def test_joint_rms(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion.npz"
            path.write_text("")
            row = utils.make_result_row(
                motion_path=path,
                category="cat",
                source_group="group",
                num_frames=10,
                completed_steps=9,
                body_error_sum=1.0,
                joint_error_sum=29.0,
                metric_count=1,
                success=True,
                termination_reason="completed",
                checkpoint="model_1.pt",
                project_root=Path(tmp),
                joint_count=29,
            )
        self.assertAlmostEqual(float(row["joint_position_error_rms_rad"]), 29.0 / math.sqrt(29), places=6)

    def test_summary_complete_requires_manifest_integrity(self):
        summary = {
            "checkpoint": "model_20000.pt",
            "num_motions": 10,
            "manifest": "validation.txt",
            "manifest_integrity": {
                "ok": True,
                "expected_count": 10,
                "observed_count": 10,
            },
        }
        self.assertTrue(
            utils.summary_is_complete(
                summary,
                expected_manifest="validation.txt",
                expected_checkpoint="model_20000.pt",
            )
        )

        summary["manifest_integrity"]["observed_count"] = 9
        self.assertFalse(
            utils.summary_is_complete(
                summary,
                expected_manifest="validation.txt",
                expected_checkpoint="model_20000.pt",
            )
        )

    def test_summary_complete_checks_checkpoint_name(self):
        summary = {
            "checkpoint": "model_33999.pt",
            "num_motions": 10,
            "manifest": "validation.txt",
            "manifest_integrity": {
                "ok": True,
                "expected_count": 10,
                "observed_count": 10,
            },
        }
        self.assertFalse(
            utils.summary_is_complete(
                summary,
                expected_manifest="validation.txt",
                expected_checkpoint="model_20000.pt",
            )
        )

    @staticmethod
    def _row(path: str, category: str, source_group: str, success: int, completion: float) -> dict[str, object]:
        return {
            "motion_path": path,
            "category": category,
            "source_group": source_group,
            "num_frames": 10,
            "completed_frames": int(completion * 10),
            "episode_steps": int(completion * 10),
            "success": success,
            "completion_ratio": f"{completion:.6f}",
            "body_position_error_m": "0.100000",
            "joint_position_error_l2_rad": "1.000000",
            "joint_position_error_rms_rad": "0.185695",
            "termination_reason": "completed" if success else "bad_motion_body_pos",
            "checkpoint": "model_1.pt",
        }


if __name__ == "__main__":
    unittest.main()
