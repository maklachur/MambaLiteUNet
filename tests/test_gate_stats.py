"""Gate history summaries for metrics.json."""

import unittest

from qamseg.train_ssl import _gate_stats_from_history
from qamseg.cli import parse_train_args


class GateStatsTests(unittest.TestCase):
    def test_defaults_match_mqe_q_scale(self):
        args, _, _ = parse_train_args(["--ssl"])
        self.assertAlmostEqual(args.ssl_tau0, 0.35)
        self.assertAlmostEqual(args.ssl_tau1, 0.55)

    def test_summary_frac_open_and_unsup_key(self):
        history = [
            {"step": 0, "gate_keep_fraction": 1.0, "q_mean": 0.7, "loss_unsup": 0.5, "tau": 0.35},
            {"step": 1, "gate_keep_fraction": 0.0, "q_mean": 0.5, "loss_unsup": 0.0, "tau": 0.55},
            {"step": 2, "gate_keep_fraction": 0.5, "q_mean": 0.6, "loss_unsup": 0.2, "tau": 0.55},
        ]
        stats = _gate_stats_from_history(history)
        self.assertAlmostEqual(stats["mean_gate_keep"], (1.0 + 0.0 + 0.5) / 3)
        self.assertAlmostEqual(stats["frac_gate_open"], 2 / 3)
        self.assertAlmostEqual(stats["frac_steps_gate_open"], 2 / 3)
        self.assertEqual(stats["first_zero_keep_step"], 1)
        self.assertAlmostEqual(stats["mean_unsup"], (0.5 + 0.0 + 0.2) / 3)
        self.assertAlmostEqual(stats["mean_q"], (0.7 + 0.5 + 0.6) / 3)


if __name__ == "__main__":
    unittest.main()
