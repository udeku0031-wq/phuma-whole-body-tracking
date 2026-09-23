from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPLAY_PATH = PROJECT_ROOT / "scripts" / "rsl_rl" / "replay_joint_gap_identity.py"


def _load_replay_module():
    spec = importlib.util.spec_from_file_location("wbt_joint_gap_identity_replay_test", REPLAY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {REPLAY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class JointGapIdentityReplayTest(unittest.TestCase):
    def test_lambda_zero_and_cold_state_are_raw_identity(self) -> None:
        replay = _load_replay_module()
        reference = replay._load_utils_modules(PROJECT_ROOT, "wbt_identity_unit_reference", include_joint_gap=True)
        current = replay._load_utils_modules(PROJECT_ROOT, "wbt_identity_unit_current", include_joint_gap=True)
        identity = replay._run_identity(
            reference,
            current,
            num_assignments=1024,
            num_motions=128,
            num_segments=511,
        )
        report = {"identity": identity}
        self.assertTrue(replay._pass_status(report))
        self.assertEqual(identity["priority"]["new_b_corrected_vs_raw"]["max_abs_diff"], 0.0)
        self.assertEqual(identity["priority"]["cold_lambda_positive_corrected_vs_raw"]["max_abs_diff"], 0.0)
        self.assertTrue(identity["trace"]["new_b_vs_old"]["before_rng_identical"])
        self.assertTrue(identity["trace"]["new_b_vs_old"]["after_rng_identical"])


if __name__ == "__main__":
    unittest.main()
