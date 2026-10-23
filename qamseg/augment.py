"""Coordinate-aligned weak and strong views plus Quality-Guided CutMix for SSL.

Fixes the spatial misalignment bug where ``weak_augment`` flipped horizontally
while ``strong_augment`` flipped vertically without inverse-transforming the
teacher pseudo-mask. Both views now share an identical geometric coordinate
frame so pixel-level consistency and cross-view Dice agreement are spatially
aligned, while ``strong_augment`` applies photometric intensity/contrast
perturbation, spatial cutout, and optional Gaussian noise.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from qamseg.mqe import QualityEstimate, as_prob_maps


def weak_augment(images: Tensor) -> Tensor:
    """Horizontal flip. ``images`` is ``(B, C, H, W)``."""
    _check_image_batch(images)
    return torch.flip(images, dims=(-1,))


def strong_augment(images: Tensor, noise_std: float = 0.0) -> Tensor:
    """Coordinate-aligned strong view: shares the weak horizontal flip, then
    applies photometric contrast/intensity shift, a corner cutout, and
    optional Gaussian noise.
    """
    _check_image_batch(images)
    # Start from the exact same geometric orientation as weak_augment
    out = weak_augment(images).clone()
    # Photometric perturbation (contrast + brightness shift within [0, 1])
    out = (out * 0.85 + 0.05).clamp(0.0, 1.0)
    height = max(int(out.shape[-2]) // 4, 1)
    width = max(int(out.shape[-1]) // 4, 1)
    out[..., :height, :width] = 0
    if noise_std > 0:
        out = (out + float(noise_std) * torch.randn_like(out)).clamp(0.0, 1.0)
    return out


def augment_labeled_batch(
    images: Tensor,
    masks: Tensor,
    *,
    step: int = 0,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Deterministic-per-step geometric augmentation (rot90 + flips) for labeled batches."""
    _check_image_batch(images)
    if masks.ndim != 4:
        raise ValueError(f"Expected NCHW masks, got shape {tuple(masks.shape)}")
    gen = torch.Generator()
    gen.manual_seed(int(seed) * 100003 + int(step) + 17)
    k = int(torch.randint(0, 4, (1,), generator=gen).item())
    img_out = torch.rot90(images, k, dims=(-2, -1))
    msk_out = torch.rot90(masks, k, dims=(-2, -1))
    if bool(torch.randint(0, 2, (1,), generator=gen).item()):
        img_out = torch.flip(img_out, dims=(-1,))
        msk_out = torch.flip(msk_out, dims=(-1,))
    if bool(torch.randint(0, 2, (1,), generator=gen).item()):
        img_out = torch.flip(img_out, dims=(-2,))
        msk_out = torch.flip(msk_out, dims=(-2,))
    return img_out.contiguous(), msk_out.contiguous()


def quality_guided_cutmix(
    view_strong: Tensor,
    teacher_probs: Tensor,
    quality: QualityEstimate,
    *,
    step: int = 0,
    seed: int = 0,
    box_frac: float = 0.5,
) -> tuple[Tensor, Tensor, QualityEstimate]:
    """Mix pairs of unlabeled samples in a batch and warp teacher targets + MQE weights.

    When ``batch_size < 2``, returns inputs unchanged.
    """
    _check_image_batch(view_strong)
    B, _, H, W = view_strong.shape
    if B < 2 or H < 4 or W < 4:
        return view_strong, teacher_probs, quality

    gen = torch.Generator()
    gen.manual_seed(int(seed) * 100003 + int(step) + 99)
    box_h = max(1, int(round(H * float(box_frac))))
    box_w = max(1, int(round(W * float(box_frac))))
    y0 = int(torch.randint(0, max(1, H - box_h + 1), (1,), generator=gen).item())
    x0 = int(torch.randint(0, max(1, W - box_w + 1), (1,), generator=gen).item())

    mask = torch.zeros((1, 1, H, W), device=view_strong.device, dtype=view_strong.dtype)
    mask[:, :, y0 : y0 + box_h, x0 : x0 + box_w] = 1.0
    area_ratio = float(box_h * box_w) / float(H * W)

    perm = torch.roll(torch.arange(B, device=view_strong.device), shifts=1)
    mixed_strong = view_strong * (1.0 - mask) + view_strong[perm] * mask
    t_probs = as_prob_maps(teacher_probs)
    mixed_teacher = t_probs * (1.0 - mask) + t_probs[perm] * mask

    pix_w = as_prob_maps(quality.pixel_weight)
    mixed_pix_w = pix_w * (1.0 - mask) + pix_w[perm] * mask
    q_vec = quality.q.view(-1)
    mixed_q = q_vec * (1.0 - area_ratio) + q_vec[perm] * area_ratio

    mixed_quality = QualityEstimate(
        q=mixed_q,
        pixel_weight=mixed_pix_w,
        entropy=quality.entropy,
        disagreement=quality.disagreement,
        agreement=quality.agreement,
        boundary_entropy=quality.boundary_entropy,
        learned_score=quality.learned_score,
        mamba_disagreement=getattr(quality, "mamba_disagreement", None),
    )
    return mixed_strong, mixed_teacher, mixed_quality


def _check_image_batch(images: Tensor) -> None:
    if images.ndim != 4:
        raise ValueError(f"Expected NCHW images, got shape {tuple(images.shape)}")
