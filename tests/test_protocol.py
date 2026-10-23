"""MedIA job matrix and mean ± std aggregation."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from qamseg.aggregate import aggregate_records, load_metrics_tree, render_csv, render_markdown
from qamseg.protocol import (
    ABLATION_NAMES,
    DEFAULT_RECIPE_NAMES,
    iter_jobs,
    job_argv,
    parse_recipe_names,
)
from require_mamba_cuda import skip_unless_mamba_cuda


class MatrixTests(unittest.TestCase):
    def test_default_matrix_covers_seeds_fractions_and_ablation(self):
        jobs = iter_jobs()
        recipes = {job.recipe for job in jobs}
        self.assertTrue(set(ABLATION_NAMES).issubset(recipes))
        self.assertTrue({"mt", "uamt", "pl", "supervised", "qam-qw"}.issubset(recipes))
        self.assertEqual({job.seed for job in jobs}, {0, 1, 2})
        self.assertEqual({job.fraction_token for job in jobs if job.recipe == "mt"}, {"5", "10", "20"})
        full = [job for job in jobs if job.fraction_token == "full"]
        self.assertTrue(full)
        self.assertTrue(all(job.recipe == "supervised" for job in full))
        self.assertEqual(len(full), 3)

    def test_ablation_argv_uses_mqe_reweight_without_a_third_network(self):
        jobs = iter_jobs(seeds=(0,), fractions=(10,), recipe_names=("qam-gated-baq-qw", "qam-qw"))
        by_name = {job.recipe: job for job in jobs}
        stacked = job_argv(by_name["qam-gated-baq-qw"], steps=2, batch_size=2, device="cuda", backbone="mamba", image_size=32, out_dir="out/a")
        pure = job_argv(by_name["qam-qw"], steps=2, batch_size=2, device="cuda", backbone="mamba", image_size=32, out_dir="out/b")
        self.assertEqual(stacked[stacked.index("--ssl-backbone") + 1], "mamba")
        self.assertEqual(stacked[stacked.index("--ssl-device") + 1], "cuda")
        self.assertIn("--ssl-baq", stacked)
        self.assertIn("--quality-reweight", stacked)
        self.assertEqual(stacked[stacked.index("--quality-reweight") + 1], "sample")
        self.assertEqual(stacked[stacked.index("--ssl-gate") + 1], "gated")
        self.assertEqual(pure[pure.index("--ssl-gate") + 1], "ungated")
        self.assertEqual(pure[pure.index("--quality-reweight") + 1], "sample")
        blob = " ".join(stacked + pure).lower()
        for banned in ("g_phi", "g_φ", "yolo", "clip", "segment anything"):
            self.assertNotIn(banned, blob)

    def test_unknown_recipe_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_recipe_names("not-a-method")
        self.assertEqual(parse_recipe_names("default"), DEFAULT_RECIPE_NAMES)


class AggregateTests(unittest.TestCase):
    def test_three_seeds_report_mean_and_sample_std(self):
        records = []
        for seed, dice in ((0, 0.50), (1, 0.60), (2, 0.70)):
            records.append(
                {
                    "recipe": "qam-qw",
                    "group": "ablation",
                    "fraction_token": "10",
                    "backbone": "mamba",
                    "data": "synthetic",
                    "eval_split": "synthetic_labeled",
                    "metrics": {
                        "dice": dice,
                        "miou": dice / 2,
                        "hd95_mean": 4.0 + seed,
                        "assd_mean": 1.0 + seed,
                    },
                }
            )
        rows = aggregate_records(records)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row["dice_mean"], 0.6)
        self.assertAlmostEqual(row["dice_std"], 0.1)
        self.assertAlmostEqual(row["hd95_mean"], 5.0)
        self.assertAlmostEqual(row["assd_mean"], 2.0)
        self.assertIn("0.6000 ± 0.1000", row["dice"])
        self.assertIn("±", render_markdown(rows))
        self.assertIn("dice_mean", render_csv(rows).splitlines()[0])

    def test_loader_and_script_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed, dice in ((0, 0.2), (1, 0.4)):
                folder = root / f"seed{seed}" / "mt" / "fraction_5"
                folder.mkdir(parents=True)
                payload = {
                    "format": "qamseg-media-metrics-v1",
                    "recipe": "mt",
                    "group": "baseline",
                    "fraction_token": "5",
                    "backbone": "mamba",
                    "data": "synthetic",
                    "eval_split": "synthetic_labeled",
                    "seed": seed,
                    "metrics": {"dice": dice, "miou": 0.1, "hd95_mean": None, "assd": "n/a"},
                }
                (folder / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_metrics_tree(root)
            self.assertEqual(len(loaded), 2)
            completed = subprocess.run(
                [sys.executable, "scripts/aggregate_media_results.py", str(root), "--out-dir", str(root / "summary")],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            text = (root / "summary" / "summary.md").read_text(encoding="utf-8")
            self.assertIn("mt", text)
            self.assertIn("0.3000 ± 0.1414", text)
            self.assertNotIn("ISIC", text)


class ProtocolScriptTests(unittest.TestCase):
    def test_dry_run_lists_the_ablation_and_does_not_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_media_protocol.py",
                    "--dry-run",
                    "--seeds",
                    "0,1,2",
                    "--fractions",
                    "5,10,20",
                    "--out-dir",
                    tmp,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("qam-gated-baq-qw", completed.stdout)
            self.assertIn("--ssl-backbone mamba", completed.stdout)
            self.assertIn("--ssl-device cuda", completed.stdout)
            self.assertNotIn("TinySegStudent", completed.stdout)
            self.assertIn("Dry run only", completed.stdout)
            manifest = json.loads((Path(tmp) / "protocol_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["seeds"], [0, 1, 2])
            self.assertEqual(manifest["fractions"], [5, 10, 20])
            self.assertIn("qam-ungated", manifest["ablation"])
            self.assertGreaterEqual(len(manifest["jobs"]), 3)
            self.assertTrue(any(job["fraction"] == "full" for job in manifest["jobs"]))

    def test_real_run_without_arrays_exits_2(self):
        completed = subprocess.run(
            [sys.executable, "scripts/run_media_protocol.py", "--steps", "1", "--seeds", "0", "--fractions", "10", "--recipes", "mt"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("does not download", completed.stdout)

    def test_smoke_without_cuda_exits_before_training(self):
        import torch

        if torch.cuda.is_available():
            self.skipTest("CUDA is available; the missing-CUDA exit is not this runner")
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_media_protocol.py",
                    "--smoke",
                    "--seeds",
                    "0",
                    "--fractions",
                    "10",
                    "--recipes",
                    "mt",
                    "--out-dir",
                    tmp,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        blob = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 2, blob)
        self.assertIn("CUDA is not available", blob)
        self.assertIn("No other student is substituted", blob)
        self.assertNotIn("gate_keep_fraction:", completed.stdout)

    @skip_unless_mamba_cuda
    def test_smoke_writes_metrics_for_reweight_and_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_media_protocol.py",
                    "--smoke",
                    "--seeds",
                    "0",
                    "--fractions",
                    "10",
                    "--recipes",
                    "qam-qw,qam-gated-baq-qw,uamt,supervised",
                    "--out-dir",
                    tmp,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            files = list(Path(tmp).rglob("metrics.json"))
            self.assertGreaterEqual(len(files), 4)
            payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
            self.assertTrue(any(item["quality_reweight"] == "sample" for item in payloads))
            self.assertTrue(all(item["data"] == "synthetic" for item in payloads))
            self.assertTrue(all(item["qar"] == "skipped" for item in payloads))
            summary = subprocess.run(
                [sys.executable, "scripts/aggregate_media_results.py", tmp],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(summary.returncode, 0, summary.stderr)
            self.assertIn("qam-qw", (Path(tmp) / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
