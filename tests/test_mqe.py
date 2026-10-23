"""MQE on synthetic probability maps. No GPU and no medical images."""

import unittest

import torch

from qamseg.mqe import (
    LearnedQualityHead,
    QualityWeights,
    absolute_disagreement,
    apply_boundary_aware_prior,
    boundary_band_mask,
    dice_agreement,
    estimate_quality,
    gradient_magnitude,
    normalized_entropy,
    reliability_target,
)


class EntropyTests(unittest.TestCase):
    def test_half_probability_is_maximum_entropy(self):
        probs = torch.full((1, 1, 8, 8), 0.5)
        entropy = normalized_entropy(probs)
        self.assertTrue(torch.allclose(entropy, torch.ones_like(entropy), atol=1e-5))

    def test_certain_pixels_have_low_entropy(self):
        probs = torch.zeros(1, 1, 8, 8)
        probs[..., 4:, :] = 1
        entropy = normalized_entropy(probs)
        self.assertTrue(torch.all(entropy < 0.02))
        self.assertTrue(torch.all(entropy >= 0))


class DisagreementAndAgreementTests(unittest.TestCase):
    def test_absolute_disagreement_endpoints(self):
        teacher = torch.zeros(1, 1, 4, 4)
        student = torch.ones(1, 1, 4, 4)
        same = absolute_disagreement(teacher, teacher)
        opposite = absolute_disagreement(teacher, student)
        self.assertTrue(torch.allclose(same, torch.zeros_like(same)))
        self.assertTrue(torch.allclose(opposite, torch.ones_like(opposite)))

    def test_hard_dice_identical_and_disjoint(self):
        mask = torch.zeros(2, 1, 16, 16)
        mask[0, :, 4:10, 4:10] = 1
        mask[1, :, 1:6, 1:6] = 1
        identical = dice_agreement(mask, mask, mode="hard")
        self.assertTrue(torch.allclose(identical, torch.ones(2), atol=1e-5))
        flipped = 1 - mask
        disjoint = dice_agreement(mask, flipped, mode="hard")
        self.assertTrue(torch.all(disjoint < 0.05))

    def test_both_empty_dice_is_one(self):
        empty = torch.zeros(1, 1, 8, 8)
        score = dice_agreement(empty, empty, mode="hard")
        self.assertAlmostEqual(score.item(), 1.0, places=5)


class BoundaryTests(unittest.TestCase):
    def test_band_follows_the_contour(self):
        probs = torch.zeros(1, 1, 16, 16)
        probs[:, :, 4:12, 4:12] = 1
        band = boundary_band_mask(probs, radius=1)
        self.assertEqual(band[0, 0, 8, 8].item(), 0.0)
        self.assertEqual(band[0, 0, 4, 6].item(), 1.0)
        self.assertEqual(band[0, 0, 0, 0].item(), 0.0)
        self.assertTrue(torch.all((band == 0) | (band == 1)))


class FusionTests(unittest.TestCase):
    def _pair(self):
        confident = torch.zeros(1, 1, 16, 16)
        confident[:, :, 4:12, 4:12] = 1
        uncertain = torch.full((1, 1, 16, 16), 0.5)
        return confident, uncertain

    def test_linear_fusion_stays_in_unit_interval_and_ranks_quality(self):
        confident, uncertain = self._pair()
        high = estimate_quality(confident, confident, confident, fusion="linear")
        low = estimate_quality(uncertain, torch.zeros_like(uncertain), torch.zeros_like(uncertain))
        self.assertGreater(high.q.item(), 0.8)
        self.assertLess(low.q.item(), 0.5)
        self.assertTrue(torch.all((high.pixel_weight >= 0) & (high.pixel_weight <= 1)))
        self.assertTrue(torch.all((low.pixel_weight >= 0) & (low.pixel_weight <= 1)))
        self.assertEqual(tuple(high.pixel_weight.shape), (1, 1, 16, 16))
        self.assertEqual(tuple(high.q.shape), (1,))

    def test_sigmoid_fusion_preserves_order(self):
        confident, uncertain = self._pair()
        high = estimate_quality(confident, confident, confident, fusion="sigmoid", center=True)
        low = estimate_quality(
            uncertain,
            torch.zeros_like(uncertain),
            torch.zeros_like(uncertain),
            fusion="sigmoid",
            center=True,
        )
        self.assertGreater(high.q.item(), low.q.item())
        self.assertGreaterEqual(low.q.item(), 0.0)
        self.assertLessEqual(high.q.item(), 1.0)

    def test_learned_weight_defaults_to_ignored(self):
        confident, _uncertain = self._pair()
        base = estimate_quality(confident, confident, confident)
        score = torch.zeros_like(confident)
        ignored = estimate_quality(
            confident,
            confident,
            confident,
            learned_score=score,
            weights=QualityWeights(learned=0.0),
        )
        self.assertTrue(torch.allclose(base.q, ignored.q))
        used = estimate_quality(
            confident,
            confident,
            confident,
            learned_score=score,
            weights=QualityWeights(learned=1.0),
        )
        self.assertLess(used.q.item(), base.q.item())

    def test_quality_does_not_require_grad(self):
        teacher = torch.rand(1, 1, 8, 8, requires_grad=True)
        estimate = estimate_quality(teacher)
        self.assertFalse(estimate.q.requires_grad)
        self.assertFalse(estimate.pixel_weight.requires_grad)
        self.assertIsNone(teacher.grad)


class LearnedHeadAndBaqTests(unittest.TestCase):
    def test_head_outputs_unit_scores(self):
        head = LearnedQualityHead(in_channels=4)
        features = torch.randn(2, 4, 8, 8)
        scores = head(features)
        self.assertEqual(tuple(scores.shape), (2, 1, 8, 8))
        self.assertTrue(torch.all((scores > 0) & (scores < 1)))

    def test_reliability_target_rewards_correct_confidence(self):
        correct = torch.full((1, 1, 4, 4), 0.99)
        labels = torch.ones(1, 1, 4, 4)
        good = reliability_target(correct, labels)
        bad = reliability_target(correct, torch.zeros_like(labels))
        self.assertTrue(torch.all(good > 0.9))
        self.assertTrue(torch.all(bad == 0))

    def test_baq_downweights_strong_edges_with_low_quality(self):
        weight = torch.ones(1, 1, 4, 4)
        grad = torch.zeros(1, 1, 4, 4)
        grad[..., :, 2:] = 1
        quality = torch.ones(1, 1, 4, 4)
        quality[..., :, 2:] = 0
        adjusted = apply_boundary_aware_prior(weight, grad, quality)
        self.assertTrue(torch.allclose(adjusted[..., :, :2], torch.ones(1, 1, 4, 2)))
        self.assertTrue(torch.allclose(adjusted[..., :, 2:], torch.zeros(1, 1, 4, 2)))

    def test_gradient_magnitude_is_normalized(self):
        image = torch.zeros(1, 3, 8, 8)
        image[..., :, 4:] = 1
        mag = gradient_magnitude(image)
        self.assertEqual(tuple(mag.shape), (1, 1, 8, 8))
        self.assertGreaterEqual(mag.min().item(), 0.0)
        self.assertLessEqual(mag.max().item(), 1.0)
        self.assertGreater(mag[0, 0, 0, 4].item(), mag[0, 0, 0, 0].item())


if __name__ == "__main__":
    unittest.main()
