from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import wandb_stitch_runs as stitch  # noqa: E402


class WandbStitchRunsTest(unittest.TestCase):
    def test_parse_run_ref_from_id_path_and_url(self):
        self.assertEqual(
            stitch.parse_run_ref("abc123", "entity", "project"),
            stitch.RunRef("entity", "project", "abc123"),
        )
        self.assertEqual(
            stitch.parse_run_ref("other/proj/run1", "entity", "project"),
            stitch.RunRef("other", "proj", "run1"),
        )
        self.assertEqual(
            stitch.parse_run_ref("https://wandb.ai/e/p/runs/r9?nw=x", "entity", "project"),
            stitch.RunRef("e", "p", "r9"),
        )

    def test_auto_step_mode_preserves_non_overlapping_runs(self):
        refs = [stitch.RunRef("e", "p", "a"), stitch.RunRef("e", "p", "b")]
        histories = [
            [{"_step": 10, "Loss/x": 1.0}, {"_step": 11, "Loss/x": 0.9}],
            [{"_step": 12, "Loss/x": 0.8}, {"_step": 13, "Loss/x": 0.7}],
        ]
        rows, summaries = stitch.stitch_histories(
            refs,
            histories,
            metrics=["Loss/x"],
            step_key="_step",
            step_mode="auto",
        )
        self.assertEqual([row["_step"] for row in rows], [10, 11, 12, 13])
        self.assertEqual(summaries[1]["offset"], 0)

    def test_auto_step_mode_offsets_reset_run(self):
        refs = [stitch.RunRef("e", "p", "a"), stitch.RunRef("e", "p", "b")]
        histories = [
            [{"_step": 10, "Loss/x": 1.0}, {"_step": 11, "Loss/x": 0.9}],
            [{"_step": 0, "Loss/x": 0.8}, {"_step": 1, "Loss/x": 0.7}],
        ]
        rows, summaries = stitch.stitch_histories(
            refs,
            histories,
            metrics=["Loss/x"],
            step_key="_step",
            step_mode="auto",
        )
        self.assertEqual([row["_step"] for row in rows], [10, 11, 12, 13])
        self.assertEqual(summaries[1]["offset"], 12)

    def test_auto_step_mode_preserves_partially_overlapping_continuation(self):
        refs = [stitch.RunRef("e", "p", "a"), stitch.RunRef("e", "p", "b")]
        histories = [
            [{"_step": 13000, "Loss/x": 1.0}, {"_step": 18027, "Loss/x": 0.9}],
            [
                {"_step": 0},
                {"_step": 18000, "Loss/x": 0.8},
                {"_step": 33998, "Loss/x": 0.7},
            ],
        ]
        rows, summaries = stitch.stitch_histories(
            refs,
            histories,
            metrics=["Loss/x"],
            step_key="_step",
            step_mode="auto",
        )
        self.assertEqual(rows[0]["_step"], 13000)
        self.assertEqual(rows[-1]["_step"], 33998)
        self.assertEqual(summaries[1]["offset"], 0)
        self.assertEqual(summaries[1]["raw_first_step"], 18000)

    def test_metric_inference_filters_internal_keys_and_prefixes(self):
        histories = [
            [
                {"_step": 1, "Loss/x": 1.0, "Metrics/y": 2.0, "other": 3.0, "_runtime": 4.0},
                {"_step": 2, "Loss/x": "bad"},
            ]
        ]
        metrics = stitch.infer_metrics(histories, explicit_metrics=None, prefixes=("Loss/", "Metrics/"))
        self.assertEqual(metrics, ["Loss/x", "Metrics/y"])


if __name__ == "__main__":
    unittest.main()
