"""MQE reweight, baseline masks, and the decision to skip QAR."""

import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import numpy as np
import torch

from qamseg.cli import parse_train_args, resolve_ssl_method
from qamseg.data import load_eval_arrays, load_labeled_arrays
from qamseg.mqe import QualityEstimate, estimate_quality
from qamseg.qar import QAR_STATUS, quality_aware_regularization
from qamseg.qgc import GateConfig, gate_weights, quality_gated_consistency_loss
from qamseg.reweight import ReweightConfig, qw_alpha, unlabeled_consistency_loss
from require_mamba_cuda import skip_unless_mamba_cuda
from utils import assd_binary, hd95_binary


def _estimate(q, pixel):
    return QualityEstimate(q=torch.as_tensor(q, dtype=torch.float32), pixel_weight=pixel)


class LegacyPathTests(unittest.TestCase):
    def test_off_matches_qgc(self):
        torch.manual_seed(0)
        student = torch.rand(2, 1, 8, 8)
        teacher = torch.rand(2, 1, 8, 8)
        quality = estimate_quality(teacher, student, student)
        gate = GateConfig(mode="gated", level="both", tau=0.4)
        legacy = quality_gated_consistency_loss(student, teacher, quality, gate)
        wrapped = unlabeled_consistency_loss(
            student,
            teacher,
            quality,
            gate,
            ReweightConfig(mode="off", lambda_qw=0.25),
            step=3,
            total_steps=10,
        )
        self.assertTrue(torch.allclose(legacy.loss, wrapped.loss))
        self.assertEqual(wrapped.qw_alpha, 0.0)

    def test_ungated_weights_are_one(self):
        pixel = torch.rand(2, 1, 4, 4)
        weights = gate_weights(
            _estimate(torch.tensor([0.2, 0.9]), pixel),
            GateConfig(mode="ungated", level="both", tau=0.8),
        )
        self.assertTrue(torch.allclose(weights, torch.ones_like(pixel)))


class SampleReweightTests(unittest.TestCase):
    def _pair(self):
        student = torch.zeros(2, 1, 4, 4)
        teacher = torch.zeros(2, 1, 4, 4)
        student[0] = 0.9
        teacher[0] = 0.9
        student[1] = 0.9
        teacher[1] = 0.1
        pixel = torch.ones(2, 1, 4, 4)
        return student, teacher, pixel

    def test_ramp_start_ignores_q(self):
        student, teacher, pixel = self._pair()
        gate = GateConfig(mode="ungated")
        config = ReweightConfig(mode="sample", ramp="linear", lambda_qw=1.0)
        low = unlabeled_consistency_loss(
            student, teacher, _estimate([0.0, 0.0], pixel), gate, config, step=0, total_steps=10
        )
        high = unlabeled_consistency_loss(
            student, teacher, _estimate([1.0, 1.0], pixel), gate, config, step=0, total_steps=10
        )
        self.assertAlmostEqual(low.qw_alpha, 0.0, places=6)
        self.assertTrue(torch.allclose(low.loss, high.loss))

    def test_finished_ramp_prefers_the_accurate_sample(self):
        student, teacher, pixel = self._pair()
        gate = GateConfig(mode="ungated")
        config = ReweightConfig(mode="sample", ramp="none", lambda_qw=1.0)
        prefer_good = unlabeled_consistency_loss(
            student, teacher, _estimate([1.0, 0.0], pixel), gate, config, step=0, total_steps=4
        )
        prefer_bad = unlabeled_consistency_loss(
            student, teacher, _estimate([0.0, 1.0], pixel), gate, config, step=0, total_steps=4
        )
        self.assertAlmostEqual(prefer_good.qw_alpha, 1.0, places=6)
        self.assertLess(prefer_good.loss.item(), prefer_bad.loss.item())
        self.assertAlmostEqual(float(prefer_good.sample_weight[0]), 1.0, places=5)
        self.assertAlmostEqual(float(prefer_good.sample_weight[1]), 0.0, places=5)

    def test_gated_sample_reweight_does_not_square_q(self):
        student = torch.full((1, 1, 4, 4), 0.8)
        teacher = torch.full((1, 1, 4, 4), 0.8)
        pixel = torch.ones(1, 1, 4, 4)
        result = unlabeled_consistency_loss(
            student,
            teacher,
            _estimate([0.5], pixel),
            GateConfig(mode="gated", level="both", tau=0.0, gamma=1.0),
            ReweightConfig(mode="sample", ramp="none"),
            step=0,
            total_steps=1,
        )
        self.assertTrue(torch.allclose(result.weights, torch.ones_like(pixel)))
        self.assertAlmostEqual(float(result.sample_weight), 0.5, places=5)

    def test_quality_score_is_detached(self):
        student = torch.full((1, 1, 4, 4), 0.7, requires_grad=True)
        teacher = torch.full((1, 1, 4, 4), 0.2)
        q = torch.tensor([0.4], requires_grad=True)
        pixel = torch.ones(1, 1, 4, 4)
        result = unlabeled_consistency_loss(
            student,
            teacher,
            QualityEstimate(q=q, pixel_weight=pixel),
            GateConfig(mode="ungated"),
            ReweightConfig(mode="sample", ramp="none"),
            step=1,
            total_steps=1,
        )
        result.loss.backward()
        self.assertIsNone(q.grad)
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_all_zero_sample_weights_are_finite(self):
        student = torch.rand(2, 1, 4, 4, requires_grad=True)
        teacher = torch.rand(2, 1, 4, 4)
        result = unlabeled_consistency_loss(
            student,
            teacher,
            _estimate([0.0, 0.0], torch.ones(2, 1, 4, 4)),
            GateConfig(mode="ungated"),
            ReweightConfig(mode="sample", ramp="none"),
        )
        self.assertEqual(result.loss.item(), 0.0)
        result.loss.backward()
        self.assertTrue(torch.all(student.grad == 0))


class PixelAndBaselineTests(unittest.TestCase):
    def test_pixel_reweight_ignores_the_wrong_half(self):
        student = torch.full((1, 1, 4, 4), 0.95, requires_grad=True)
        teacher = torch.zeros(1, 1, 4, 4)
        teacher[..., :, :2] = 1
        kept = torch.zeros(1, 1, 4, 4)
        kept[..., :, :2] = 1
        dropped = 1 - kept
        config = ReweightConfig(mode="pixel", ramp="none")
        gate = GateConfig(mode="ungated")
        loss_kept = unlabeled_consistency_loss(
            student, teacher, _estimate([1.0], kept), gate, config
        ).loss
        loss_dropped = unlabeled_consistency_loss(
            student, teacher, _estimate([1.0], dropped), gate, config
        ).loss
        self.assertLess(loss_kept.item(), loss_dropped.item())
        loss_kept.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_entropy_mask_drops_the_uncertain_pixel(self):
        teacher = torch.full((1, 1, 1, 2), 0.99)
        teacher[..., 1] = 0.5
        student = torch.full_like(teacher, 0.99)
        result = unlabeled_consistency_loss(
            student,
            teacher,
            _estimate([1.0], torch.ones_like(teacher)),
            GateConfig(mode="ungated"),
            ReweightConfig(mode="off", uncertainty="entropy", uncertainty_tau=0.75),
        )
        self.assertEqual(float(result.weights[0, 0, 0, 1]), 0.0)
        self.assertGreater(float(result.weights[0, 0, 0, 0]), 0.5)

    def test_confidence_threshold_drops_the_unsure_pixel(self):
        teacher = torch.tensor([[[[0.99, 0.60]]]])
        student = torch.ones_like(teacher)
        result = unlabeled_consistency_loss(
            student,
            teacher,
            _estimate([1.0], torch.ones_like(teacher)),
            GateConfig(mode="ungated"),
            ReweightConfig(mode="off", pl_threshold=0.95),
        )
        self.assertEqual(float(result.weights[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(result.weights[0, 0, 0, 1]), 0.0)
        self.assertTrue(torch.isfinite(result.loss))


class RampTests(unittest.TestCase):
    def test_endpoints(self):
        self.assertAlmostEqual(qw_alpha(0, 100, "gaussian"), math.exp(-5.0), places=6)
        self.assertAlmostEqual(qw_alpha(100, 100, "gaussian"), 1.0, places=6)
        self.assertAlmostEqual(qw_alpha(0, 10, "linear"), 0.0, places=6)
        self.assertAlmostEqual(qw_alpha(10, 10, "linear"), 1.0, places=6)
        self.assertEqual(qw_alpha(0, 10, "none"), 1.0)


class QarTests(unittest.TestCase):
    def test_qar_is_skipped(self):
        self.assertEqual(QAR_STATUS, "skipped")
        with self.assertRaises(RuntimeError) as ctx:
            quality_aware_regularization()
        self.assertIn("g_phi", str(ctx.exception))


class SurfaceTests(unittest.TestCase):
    def test_identical_masks_have_zero_assd_and_hd95(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        self.assertEqual(assd_binary(mask, mask), 0.0)
        self.assertEqual(hd95_binary(mask, mask), 0.0)

    def test_one_empty_mask_is_undefined(self):
        full = np.ones((4, 4), dtype=bool)
        empty = np.zeros((4, 4), dtype=bool)
        self.assertTrue(math.isnan(assd_binary(full, empty)))


class CliPresetTests(unittest.TestCase):
    def test_defaults_keep_qgc(self):
        args, _unknown, reserved = parse_train_args(["--ssl"])
        self.assertEqual(args.quality_reweight, "off")
        self.assertEqual(args.ssl_method, "qam")
        self.assertEqual(args.ssl_gate, "gated")
        self.assertEqual(args.ssl_lambda_qw, 1.0)
        self.assertEqual(args.ssl_qw_ramp, "gaussian")
        self.assertEqual(reserved, [])

    def test_mt_preset_clears_mqe_reweight(self):
        args, _unknown, _reserved = parse_train_args(
            ["--ssl", "--ssl-method", "mt", "--quality-reweight", "sample", "--ssl-baq"]
        )
        resolved = resolve_ssl_method(args)
        self.assertEqual(resolved.ssl_gate, "ungated")
        self.assertEqual(resolved.quality_reweight, "off")
        self.assertFalse(resolved.ssl_baq)
        self.assertEqual(resolved.ssl_uncertainty, "off")

    def test_uamt_and_pl_and_supervised_presets(self):
        uamt = resolve_ssl_method(parse_train_args(["--ssl-method", "uamt"])[0])
        self.assertEqual(uamt.ssl_uncertainty, "entropy")
        self.assertEqual(uamt.quality_reweight, "off")
        pl = resolve_ssl_method(parse_train_args(["--ssl-method", "pl"])[0])
        self.assertAlmostEqual(pl.ssl_pl_threshold, 0.95)
        custom = resolve_ssl_method(parse_train_args(["--ssl-method", "pl", "--ssl-pl-threshold", "0.8"])[0])
        self.assertAlmostEqual(custom.ssl_pl_threshold, 0.8)
        supervised = resolve_ssl_method(parse_train_args(["--ssl-method", "supervised"])[0])
        self.assertEqual(supervised.ssl_lambda_max, 0.0)
        self.assertEqual(supervised.ssl_gate, "ungated")


class SubprocessReweightTests(unittest.TestCase):
    def _run(self, extra):
        completed = subprocess.run(
            [
                sys.executable,
                "train.py",
                "--ssl",
                "--ssl-steps",
                "1",
                "--ssl-image-size",
                "32",
                "--ssl-batch-size",
                "2",
                *extra,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        self.assertIn("gate_keep_fraction:", completed.stdout)
        self.assertIn("QAR: skipped", completed.stdout)
        return completed.stdout

    @skip_unless_mamba_cuda
    def test_sample_and_pixel_paths_smoke(self):
        sample = self._run(["--quality-reweight", "sample", "--ssl-gate", "ungated"])
        self.assertIn("reweight: sample", sample)
        pixel = self._run(["--quality-reweight", "pixel", "--ssl-qw-ramp", "linear"])
        self.assertIn("reweight: pixel", pixel)

    @skip_unless_mamba_cuda
    def test_baseline_methods_smoke(self):
        for method in ("mt", "uamt", "pl", "supervised"):
            text = self._run(["--ssl-method", method])
            self.assertIn(f"SSL method: {method}", text)
            self.assertIn("reweight: off", text)

    @skip_unless_mamba_cuda
    def test_default_ssl_still_exits_zero(self):
        text = self._run([])
        self.assertIn("reweight: off", text)
        self.assertIn("Student: MambaLiteUNet", text)
        self.assertNotIn("TinySegStudent", text)

    def test_held_out_eval_name_is_allowed_only_for_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            images = np.zeros((2, 16, 16, 3), dtype=np.uint8)
            masks = np.zeros((2, 16, 16), dtype=np.uint8)
            images[:, 4:12, 4:12] = 200
            masks[:, 4:12, 4:12] = 255
            image_path = os.path.join(tmp, "data_test.npy")
            mask_path = os.path.join(tmp, "mask_test.npy")
            np.save(image_path, images)
            np.save(mask_path, masks)
            with self.assertRaises(ValueError):
                load_labeled_arrays(image_path, mask_path)
            loaded, loaded_masks = load_eval_arrays(image_path, mask_path)
            self.assertEqual(tuple(loaded.shape), (2, 3, 16, 16))
            self.assertEqual(tuple(loaded_masks.shape), (2, 1, 16, 16))


if __name__ == "__main__":
    unittest.main()
