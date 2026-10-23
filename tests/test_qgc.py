"""QGC gating, empty-gate safety, and the curriculum helpers."""

import math
import unittest

import torch

from qamseg.mqe import QualityEstimate, estimate_quality
from qamseg.qgc import (
    GateConfig,
    gate_weights,
    lambda_u_at_step,
    quality_gated_consistency_loss,
    tau_at_step,
)


def _estimate(q, pixel):
    return QualityEstimate(q=q, pixel_weight=pixel)


class GateBehaviorTests(unittest.TestCase):
    def test_soft_image_weights_prefer_the_matching_sample(self):
        student = torch.full((2, 1, 4, 4), 0.9)
        teacher = torch.zeros(2, 1, 4, 4)
        teacher[0] = 0.9
        pixel = torch.ones(2, 1, 4, 4)
        config = GateConfig(mode="soft", level="image", gamma=1.0, hard_targets=True)
        loss_good = quality_gated_consistency_loss(
            student,
            teacher,
            _estimate(torch.tensor([1.0, 0.0]), pixel),
            config,
        ).loss
        loss_bad = quality_gated_consistency_loss(
            student,
            teacher,
            _estimate(torch.tensor([0.0, 1.0]), pixel),
            config,
        ).loss
        self.assertLess(loss_good.item(), 0.5)
        self.assertGreater(loss_bad.item(), loss_good.item() + 0.5)

    def test_soft_pixel_weights_ignore_the_wrong_half(self):
        student = torch.full((1, 1, 4, 4), 0.95)
        teacher = torch.zeros(1, 1, 4, 4)
        teacher[..., :, :2] = 1
        kept = torch.zeros(1, 1, 4, 4)
        kept[..., :, :2] = 1
        config = GateConfig(mode="soft", level="pixel", gamma=1.0, hard_targets=True)
        q = torch.ones(1)
        loss_kept = quality_gated_consistency_loss(
            student, teacher, _estimate(q, kept), config
        ).loss
        loss_all = quality_gated_consistency_loss(
            student, teacher, _estimate(q, torch.ones_like(kept)), config
        ).loss
        self.assertLess(loss_kept.item(), loss_all.item())

    def test_hard_gate_keep_fraction(self):
        student = torch.full((2, 1, 4, 4), 0.8)
        teacher = torch.full((2, 1, 4, 4), 0.8)
        pixel = torch.ones(2, 1, 4, 4)
        config = GateConfig(mode="hard", level="image", tau=0.5, hard_targets=True)
        result = quality_gated_consistency_loss(
            student,
            teacher,
            _estimate(torch.tensor([0.2, 0.9]), pixel),
            config,
        )
        self.assertAlmostEqual(result.keep_fraction.item(), 0.5, places=5)
        self.assertTrue(torch.isfinite(result.loss))

    def test_gated_mode_multiplies_image_score_and_pixel_map(self):
        pixel = torch.zeros(1, 1, 2, 2)
        pixel[..., 0, :] = 1
        weights = gate_weights(
            _estimate(torch.tensor([0.9]), pixel),
            GateConfig(mode="gated", level="both", tau=0.5, gamma=1.0),
        )
        self.assertTrue(torch.allclose(weights[0, 0, 0], torch.full((2,), 0.9)))
        self.assertTrue(torch.allclose(weights[0, 0, 1], torch.zeros(2)))


class EmptyGateTests(unittest.TestCase):
    def test_empty_gate_is_finite_and_has_zero_grad(self):
        student = torch.rand(2, 1, 8, 8, requires_grad=True)
        teacher = torch.rand(2, 1, 8, 8, requires_grad=True)
        quality = _estimate(torch.zeros(2), torch.zeros(2, 1, 8, 8))
        config = GateConfig(mode="hard", level="both", tau=0.5)
        result = quality_gated_consistency_loss(student, teacher, quality, config)
        self.assertEqual(result.keep_fraction.item(), 0.0)
        self.assertTrue(torch.isfinite(result.loss))
        self.assertEqual(result.loss.item(), 0.0)
        result.loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.all(student.grad == 0))
        self.assertIsNone(teacher.grad)
        self.assertFalse(torch.isnan(student.grad).any())

    def test_soft_all_zero_weights_do_not_divide_by_zero(self):
        student = torch.full((1, 1, 4, 4), 0.7, requires_grad=True)
        teacher = torch.ones(1, 1, 4, 4)
        result = quality_gated_consistency_loss(
            student,
            teacher,
            _estimate(torch.zeros(1), torch.zeros(1, 1, 4, 4)),
            GateConfig(mode="soft", level="both", gamma=1.0),
        )
        self.assertEqual(result.loss.item(), 0.0)
        self.assertFalse(torch.isnan(result.loss))
        result.loss.backward()
        self.assertTrue(torch.all(student.grad == 0))

    def test_teacher_is_stop_grad_when_the_gate_is_open(self):
        student = torch.rand(2, 1, 6, 6, requires_grad=True)
        teacher = torch.rand(2, 1, 6, 6, requires_grad=True)
        quality = _estimate(torch.ones(2), torch.ones(2, 1, 6, 6))
        result = quality_gated_consistency_loss(
            student,
            teacher,
            quality,
            GateConfig(mode="soft", level="image", hard_targets=False),
        )
        result.loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_random_soft_gate_stays_finite(self):
        torch.manual_seed(0)
        student = torch.rand(2, 1, 8, 8, requires_grad=True)
        teacher = torch.rand(2, 1, 8, 8)
        quality = estimate_quality(teacher, student.detach(), student.detach())
        result = quality_gated_consistency_loss(
            student,
            teacher,
            quality,
            GateConfig(mode="soft", level="both"),
        )
        self.assertTrue(torch.isfinite(result.loss))
        self.assertGreaterEqual(result.keep_fraction.item(), 0.0)
        self.assertLessEqual(result.keep_fraction.item(), 1.0)
        result.loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())


class CurriculumTests(unittest.TestCase):
    def test_lambda_ramp_endpoints(self):
        start = lambda_u_at_step(0, 100, lambda_max=1.0)
        end = lambda_u_at_step(100, 100, lambda_max=1.0)
        mid_cap = lambda_u_at_step(250, 100, lambda_max=0.5)
        self.assertAlmostEqual(start, math.exp(-5.0), places=6)
        self.assertAlmostEqual(end, 1.0, places=6)
        self.assertAlmostEqual(mid_cap, 0.5, places=6)
        self.assertLess(start, end)

    def test_tau_ramp(self):
        self.assertAlmostEqual(tau_at_step(0, 0.2, 0.8, 10), 0.2)
        self.assertAlmostEqual(tau_at_step(5, 0.2, 0.8, 10), 0.5)
        self.assertAlmostEqual(tau_at_step(10, 0.2, 0.8, 10), 0.8)
        self.assertAlmostEqual(tau_at_step(30, 0.2, 0.8, 10), 0.8)
        self.assertAlmostEqual(tau_at_step(0, 0.2, 0.9, 0), 0.9)


if __name__ == "__main__":
    unittest.main()
