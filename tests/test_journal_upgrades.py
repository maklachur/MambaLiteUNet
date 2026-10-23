"""Unit tests verifying all QAM-Seg bug fixes and journal-level upgrades."""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import torch
import torch.nn as nn

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from qamseg.augment import (
    augment_labeled_batch,
    quality_guided_cutmix,
    strong_augment,
    weak_augment,
)
from qamseg.data import ArrayImageDataset, build_dual_loaders, iter_dual_batches
from qamseg.mqe import QualityWeights, dice_agreement, estimate_quality
from qamseg.qar import prototype_quality_regularization
from qamseg.qgc import GateConfig, quality_gated_consistency_loss
from qamseg.backbones import build_mamba_student
from qamseg.train_ssl import run_ssl_loop
from require_mamba_cuda import skip_unless_mamba_cuda


def _install_mock_mamba_and_timm():
    """Provide mock timm and mamba_ssm modules when testing MambaLiteUNet on CPU."""
    timm_mod = types.ModuleType("timm")
    models_mod = types.ModuleType("timm.models")
    layers_mod = types.ModuleType("timm.models.layers")

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()

        def forward(self, x):
            return x

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        # Match timm: initialization must not run as an in-place grad op.
        with torch.no_grad():
            tensor.normal_(mean, std)
        return tensor

    layers_mod.DropPath = DropPath
    layers_mod.trunc_normal_ = trunc_normal_
    models_mod.layers = layers_mod
    timm_mod.models = models_mod

    mamba_mod = types.ModuleType("mamba_ssm")

    class DummyMamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.d_model = d_model
            # Specialized nonzero dt_proj.bias that must NOT be zeroed by _init_weights
            self.dt_proj = nn.Linear(d_model, d_model, bias=True)
            nn.init.constant_(self.dt_proj.bias, 0.777)

        def forward(self, x):
            return self.dt_proj(x)

    mamba_mod.Mamba = DummyMamba
    return {
        "timm": timm_mod,
        "timm.models": models_mod,
        "timm.models.layers": layers_mod,
        "mamba_ssm": mamba_mod,
    }


class MambaArchitectureFixesTests(unittest.TestCase):
    def test_scan_2d_and_unscan_2d_are_exact_inverses(self):
        with mock.patch.dict(sys.modules, _install_mock_mamba_and_timm()):
            from models.MambaLiteUNet import scan_2d, unscan_2d

            x = torch.arange(2 * 6 * 4, dtype=torch.float32).view(2, 6, 4)
            for direction in range(4):
                scanned = scan_2d(x, H=2, W=3, direction=direction)
                restored = unscan_2d(scanned, H=2, W=3, direction=direction)
                self.assertTrue(
                    torch.allclose(x, restored),
                    f"unscan_2d failed to invert scan_2d for direction={direction}",
                )
            for direction in (1, 2, 3):
                self.assertFalse(
                    torch.allclose(x, scan_2d(x, H=2, W=3, direction=direction))
                )

    def test_mamba_internal_weights_protected_from_init_overwrite(self):
        with mock.patch.dict(sys.modules, _install_mock_mamba_and_timm()):
            sys.modules.pop("models.MambaLiteUNet", None)
            from models.MambaLiteUNet import MambaLiteUNet

            model = MambaLiteUNet(
                num_classes=1,
                input_channels=3,
                c_list=[8, 8, 8, 8, 8, 8],
            )
            branch0_bias = model.encoder4.mamba_layers[0].mamba.dt_proj.bias
            self.assertTrue(
                torch.allclose(branch0_bias, torch.full_like(branch0_bias, 0.777)),
                "Mamba internal dt_proj.bias was overwritten by _init_weights!",
            )

    def test_mamba_forward_populates_branch_disagreement_and_supports_ablations(self):
        with mock.patch.dict(sys.modules, _install_mock_mamba_and_timm()):
            sys.modules.pop("models.MambaLiteUNet", None)
            from models.MambaLiteUNet import MambaLiteUNet

            # Width 8 divides GroupNorm groups (4), AMF branches (4), and the
            # default 8-head mixer. Width 4 cannot build MultiheadAttention.
            model_full = MambaLiteUNet(
                num_classes=1,
                input_channels=3,
                c_list=[8, 8, 8, 8, 8, 8],
            )
            x = torch.rand(2, 3, 32, 32)
            out = model_full(x)
            self.assertEqual(tuple(out.shape), (2, 1, 32, 32))
            self.assertIn("branch_disagreement", model_full.last_aux)
            self.assertIn("features", model_full.last_aux)
            d_map = model_full.last_aux["branch_disagreement"]
            self.assertIsNotNone(d_map)
            self.assertEqual(tuple(d_map.shape), (2, 1, 32, 32))

            p_full = sum(p.numel() for p in model_full.parameters())
            model_no_lgfm = MambaLiteUNet(c_list=[8, 8, 8, 8, 8, 8], use_lgfm=False)
            model_no_cga = MambaLiteUNet(c_list=[8, 8, 8, 8, 8, 8], use_cga=False)
            model_no_amf = MambaLiteUNet(c_list=[8, 8, 8, 8, 8, 8], use_amf=False)
            self.assertLess(sum(p.numel() for p in model_no_lgfm.parameters()), p_full)
            self.assertLess(sum(p.numel() for p in model_no_cga.parameters()), p_full)
            self.assertLess(sum(p.numel() for p in model_no_amf.parameters()), p_full)
            self.assertEqual(tuple(model_no_amf(x).shape), (2, 1, 32, 32))
            self.assertEqual(tuple(model_no_cga(x).shape), (2, 1, 32, 32))


class AugmentationAlignmentTests(unittest.TestCase):
    def test_weak_and_strong_views_share_spatial_coordinates(self):
        # Asymmetric patch at top-left (rows 2:6, cols 2:6); under H-flip it moves
        # to top-right (rows 2:6, cols 10:14) in BOTH weak and strong views,
        # avoiding the [:4, :4] top-left cutout.
        images = torch.zeros(2, 3, 16, 16)
        images[:, :, 2:6, 2:6] = 1.0
        weak = weak_augment(images)
        strong = strong_augment(images)
        weak_fg = (weak.mean(dim=1, keepdim=True) > 0.5).float()
        strong_fg = (strong.mean(dim=1, keepdim=True) > 0.5).float()
        agreement = dice_agreement(weak_fg, strong_fg, mode="hard")
        self.assertTrue(torch.allclose(agreement, torch.ones(2), atol=1e-5))

    def test_quality_guided_cutmix_warps_images_targets_and_weights(self):
        view_strong = torch.zeros(2, 3, 16, 16)
        view_strong[0] = 0.2
        view_strong[1] = 0.8
        teacher_probs = torch.zeros(2, 1, 16, 16)
        teacher_probs[1] = 1.0
        quality = estimate_quality(teacher_probs)
        mixed_s, mixed_t, mixed_q = quality_guided_cutmix(
            view_strong, teacher_probs, quality, step=1, seed=0, box_frac=0.5
        )
        self.assertEqual(tuple(mixed_s.shape), (2, 3, 16, 16))
        self.assertEqual(tuple(mixed_t.shape), (2, 1, 16, 16))
        self.assertEqual(tuple(mixed_q.pixel_weight.shape), (2, 1, 16, 16))
        self.assertGreater(float(mixed_t[0].mean().item()), 0.1)
        self.assertLess(float(mixed_t[0].mean().item()), 0.9)

    def test_labeled_batch_augmentation_preserves_mask_alignment(self):
        images = torch.zeros(2, 3, 12, 12)
        masks = torch.zeros(2, 1, 12, 12)
        images[:, :, 1:4, 8:11] = 1.0
        masks[:, :, 1:4, 8:11] = 1.0
        aug_img, aug_msk = augment_labeled_batch(images, masks, step=3, seed=5)
        self.assertTrue(
            torch.allclose(aug_img[:, :1], aug_msk, atol=1e-6),
            "Image and mask must undergo identical geometric transformation",
        )


class UnlabeledPoolCoverageTests(unittest.TestCase):
    def test_iter_dual_batches_covers_full_unlabeled_pool(self):
        images_l = torch.zeros(2, 3, 8, 8)
        masks_l = torch.zeros(2, 1, 8, 8)
        images_u = torch.zeros(10, 3, 8, 8)
        for i in range(10):
            images_u[i] = float(i) / 10.0
        labeled_ds = ArrayImageDataset(images_l, masks_l)
        unlabeled_ds = ArrayImageDataset(images_u)
        l_loader, u_loader = build_dual_loaders(
            labeled_ds, unlabeled_ds, batch_size=2, shuffle=False
        )
        batches = list(iter_dual_batches(l_loader, u_loader))
        self.assertEqual(len(batches), 5)
        seen_unlabeled_vals = [
            round(float(b[2][0, 0, 0, 0].item()), 2) for b in batches
        ]
        self.assertEqual(seen_unlabeled_vals, [0.0, 0.2, 0.4, 0.6, 0.8])


class ImageQualityScalingTests(unittest.TestCase):
    def test_qgc_does_not_cancel_image_quality_for_single_sample(self):
        student = torch.full((1, 1, 4, 4), 0.8)
        teacher = torch.zeros(1, 1, 4, 4)
        pixel = torch.ones(1, 1, 4, 4)
        cfg = GateConfig(mode="gated", level="both", tau=0.3, gamma=1.0)
        high_q = estimate_quality(teacher)
        high_q.q = torch.tensor([0.9])
        high_q.pixel_weight = pixel
        low_q = estimate_quality(teacher)
        low_q.q = torch.tensor([0.45])
        low_q.pixel_weight = pixel

        loss_high = quality_gated_consistency_loss(student, teacher, high_q, cfg).loss
        loss_low = quality_gated_consistency_loss(student, teacher, low_q, cfg).loss
        self.assertAlmostEqual(
            float(loss_high.item()), float(loss_low.item()) * 2.0, places=4
        )


class MambaDisagreementAndPrototypeQarTests(unittest.TestCase):
    def test_mamba_disagreement_cue_lowers_quality_when_branches_disagree(self):
        confident = torch.zeros(1, 1, 8, 8)
        confident[:, :, 2:6, 2:6] = 1.0
        zero_disagree = torch.zeros(1, 1, 8, 8)
        high_disagree = torch.ones(1, 1, 8, 8)
        est_agree = estimate_quality(
            confident,
            confident,
            confident,
            mamba_disagreement=zero_disagree,
            weights=QualityWeights(mamba=1.0),
        )
        est_disagree = estimate_quality(
            confident,
            confident,
            confident,
            mamba_disagreement=high_disagree,
            weights=QualityWeights(mamba=1.0),
        )
        self.assertGreater(float(est_agree.q.item()), float(est_disagree.q.item()))
        self.assertTrue(
            torch.all(est_agree.pixel_weight > est_disagree.pixel_weight)
        )

    def test_prototype_qar_is_finite_and_differentiable(self):
        student_features = torch.randn(2, 4, 8, 8, requires_grad=True)
        teacher_probs = torch.zeros(2, 1, 8, 8)
        teacher_probs[:, :, 2:6, 2:6] = 0.95
        quality = estimate_quality(teacher_probs)
        loss_qar = prototype_quality_regularization(
            student_features, teacher_probs, quality, high_tau=0.5, boundary_radius=1
        )
        self.assertTrue(torch.isfinite(loss_qar))
        self.assertGreater(float(loss_qar.item()), 0.0)
        loss_qar.backward()
        self.assertIsNotNone(student_features.grad)

    @skip_unless_mamba_cuda
    def test_ssl_loop_with_cutmix_and_prototype_qar_runs_cleanly(self):
        from qamseg.data import build_ssl_datasets

        def _to_cuda(batch):
            # Unlabeled batches are a single Tensor. Iterating that walks the
            # batch dim, and Tensor() rejects a generator ("data must be a
            # sequence"). Recurse so tensors move as a whole and list/tuple
            # batches keep their type.
            if torch.is_tensor(batch):
                return batch.cuda(non_blocking=True)
            if isinstance(batch, (list, tuple)):
                return type(batch)(_to_cuda(item) for item in batch)
            return batch

        def _cuda_loader(loader):
            class _Loader:
                def __len__(self):
                    return len(loader)

                def __iter__(self):
                    for batch in loader:
                        yield _to_cuda(batch)

            return _Loader()

        labeled, unlabeled = build_ssl_datasets(length=4, size=32, seed=1)
        l_loader, u_loader = build_dual_loaders(labeled, unlabeled, batch_size=2)
        result = run_ssl_loop(
            build_mamba_student().cuda(),
            _cuda_loader(l_loader),
            _cuda_loader(u_loader),
            _cuda_loader(l_loader),
            steps=2,
            gate=GateConfig(mode="gated", level="both"),
            lambda_max=1.0,
            tau0=0.2,
            tau1=0.6,
            warm_steps=2,
            use_baq=True,
            quality_reweight="sample",
            use_cutmix=True,
            qar_mode="prototype",
            lambda_qar=0.1,
            augment_labeled=True,
        )
        self.assertEqual(len(result["history"]), 2)
        self.assertIn("teacher_metrics", result)
        for row in result["history"]:
            self.assertTrue(np.isfinite(row["loss"]))
            self.assertGreaterEqual(row["loss_qar"], 0.0)


if __name__ == "__main__":
    unittest.main()
