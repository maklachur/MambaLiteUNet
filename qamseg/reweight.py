"""MQE-guided reweighting of the unlabeled consistency loss.

QG-SSL's PL-QW multiplies each pseudolabel loss by a frozen external
quality score ``w_j = g_φ(x, ŷ)``. This module is that weighting rule
with the score taken from the existing MQE estimate. It does not build
a quality network.

When ``mode`` is ``off`` and no UA-MT / PL baseline is requested, the
call delegates to ``quality_gated_consistency_loss`` so the QGC objective
is used directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from qamseg.mqe import QualityEstimate, as_prob_maps, normalized_entropy
from qamseg.qgc import (
    GateConfig,
    GateResult,
    _image_and_pixel_gate,
    gate_weights,
    quality_gated_consistency_loss,
)


@dataclass(frozen=True)
class ReweightConfig:
    """Configuration for MQE reweighting and uncertainty/PL baselines."""

    mode: str = "off"
    lambda_qw: float = 1.0
    ramp: str = "gaussian"
    uncertainty: str = "off"
    uncertainty_tau: float = 0.75
    pl_threshold: float = 0.0

    def __post_init__(self):
        if self.mode not in ("off", "sample", "pixel"):
            raise ValueError(f"Unknown quality reweight mode: {self.mode}")
        if self.ramp not in ("gaussian", "linear", "none"):
            raise ValueError(f"Unknown quality-reweight ramp: {self.ramp}")
        if self.uncertainty not in ("off", "entropy"):
            raise ValueError(f"Unknown uncertainty mode: {self.uncertainty}")
        if self.lambda_qw < 0:
            raise ValueError(f"lambda_qw must be non-negative, got {self.lambda_qw}")
        if not 0.0 <= float(self.uncertainty_tau) <= 1.0:
            raise ValueError(
                f"uncertainty_tau must lie in [0, 1], got {self.uncertainty_tau}"
            )
        if not 0.0 <= float(self.pl_threshold) <= 1.0:
            raise ValueError(f"pl_threshold must lie in [0, 1], got {self.pl_threshold}")


def qw_alpha(step: int, total_steps: int, ramp: str) -> float:
    """Ramp in ``[0, 1]``. ``step >= total_steps`` is 1 for gaussian and linear."""
    if ramp == "none":
        return 1.0
    if ramp not in ("gaussian", "linear"):
        raise ValueError(f"Unknown quality-reweight ramp: {ramp}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    ratio = min(1.0, max(0.0, float(step) / float(total_steps)))
    if ramp == "linear":
        return ratio
    return math.exp(-5.0 * (1.0 - ratio) ** 2)


def _legacy_qgc(config: ReweightConfig, gate: GateConfig) -> bool:
    return (
        config.mode == "off"
        and config.uncertainty == "off"
        and config.pl_threshold <= 0.0
        and gate.mode in ("hard", "soft", "gated")
    )


def _gate_factors(
    quality: QualityEstimate,
    gate: GateConfig,
    reweight_mode: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Image keep ``(B,1,1,1)``, spatial weights, ``q`` ``(B,)``, pixel map.

    Continuous ``q`` is not folded into the image keep for ``hard`` / ``gated``
    when sample/pixel reweighting is active.
    """
    pixel = as_prob_maps(quality.pixel_weight).detach().clamp(0.0, 1.0)
    q = quality.q.detach().float().flatten()
    if q.numel() != pixel.shape[0]:
        raise ValueError(
            f"Reweight batch mismatch: q {tuple(q.shape)} vs pixels {tuple(pixel.shape)}"
        )
    q = q.to(device=pixel.device, dtype=pixel.dtype).clamp(0.0, 1.0)
    ones_i = torch.ones(pixel.shape[0], 1, 1, 1, device=pixel.device, dtype=pixel.dtype)
    ones_p = torch.ones_like(pixel)
    if gate.mode == "ungated":
        return ones_i, ones_p, q, pixel

    use_image = gate.level in ("image", "both")
    use_pixel = gate.level in ("pixel", "both")
    if gate.mode == "soft":
        image = q.view(-1, 1, 1, 1).pow(gate.gamma) if use_image else ones_i
        spatial = pixel.pow(gate.gamma) if use_pixel else ones_p
        return image, spatial, q, pixel
    if gate.mode not in ("hard", "gated"):
        raise ValueError(f"Unknown gate mode: {gate.mode}")

    image = (q >= gate.tau).to(dtype=pixel.dtype).view(-1, 1, 1, 1) if use_image else ones_i
    if not use_pixel:
        spatial = ones_p
    elif reweight_mode == "pixel":
        spatial = (pixel >= gate.tau).to(dtype=pixel.dtype) if gate.mode == "hard" else ones_p
    elif gate.mode == "hard":
        spatial = (pixel >= gate.tau).to(dtype=pixel.dtype)
    else:
        spatial = pixel
    return image, spatial, q, pixel


def _apply_uncertainty(weights: Tensor, teacher: Tensor, config: ReweightConfig) -> Tensor:
    if config.uncertainty == "off":
        return weights
    entropy = normalized_entropy(teacher).detach().clamp(0.0, 1.0)
    factor = (1.0 - entropy).clamp(0.0, 1.0)
    if float(config.uncertainty_tau) < 1.0:
        factor = factor * (entropy <= float(config.uncertainty_tau)).to(dtype=factor.dtype)
    return weights * factor.to(dtype=weights.dtype)


def _apply_pl(weights: Tensor, teacher: Tensor, config: ReweightConfig) -> Tensor:
    if config.pl_threshold <= 0.0:
        return weights
    confidence = torch.maximum(teacher, 1.0 - teacher)
    keep = (confidence >= float(config.pl_threshold)).to(dtype=weights.dtype)
    return weights * keep


def _prepare_pair(student_probs: Tensor, teacher_probs: Tensor, gate: GateConfig):
    student = as_prob_maps(student_probs).clamp(gate.eps, 1.0 - gate.eps)
    teacher = as_prob_maps(teacher_probs).detach()
    if student.shape != teacher.shape:
        raise ValueError(
            f"Reweight shape mismatch: student {tuple(student.shape)} vs teacher {tuple(teacher.shape)}"
        )
    if gate.hard_targets:
        target = (teacher >= 0.5).to(dtype=student.dtype)
    else:
        target = teacher.clamp(0.0, 1.0).to(dtype=student.dtype)
    return student, teacher, target


def _zero_result(student: Tensor, weights: Tensor, alpha: float) -> GateResult:
    zero = student.sum() * 0.0
    keep = (weights > 0).to(dtype=student.dtype).mean()
    return GateResult(
        loss=zero,
        keep_fraction=keep,
        mean_weight=weights.mean(),
        weights=weights,
        qw_alpha=float(alpha),
    )


def _finish(
    student: Tensor,
    target: Tensor,
    weights: Tensor,
    gate: GateConfig,
    *,
    alpha: float,
    lambda_qw: float,
    reduction: str,
    sample_weight: Optional[Tensor] = None,
    image_w: Optional[Tensor] = None,
    spatial_w: Optional[Tensor] = None,
) -> GateResult:
    weights = weights.to(device=student.device, dtype=student.dtype)
    if weights.shape != student.shape:
        raise ValueError(
            f"Reweight weight shape {tuple(weights.shape)} != student {tuple(student.shape)}"
        )
    if not bool((weights > 0).any()) or (
        sample_weight is not None and not bool((sample_weight > 0).any())
    ):
        return _zero_result(student, weights, alpha)

    bce_map = F.binary_cross_entropy(student, target, reduction="none")
    batch = student.shape[0]
    flat_w = weights.view(batch, -1)
    flat_bce = bce_map.view(batch, -1)
    flat_p = student.view(batch, -1)
    flat_t = target.view(batch, -1)
    mass = flat_w.sum(dim=1)
    alive = mass > 0
    intersection = (flat_w * flat_p * flat_t).sum(dim=1)
    denom = (flat_w * flat_p).sum(dim=1) + (flat_w * flat_t).sum(dim=1)
    dice = (2.0 * intersection + gate.eps) / (denom + gate.eps)
    keep = (weights > 0).to(dtype=student.dtype).mean()
    mean_weight = weights.mean()

    if reduction == "normalized":
        if image_w is not None and spatial_w is not None:
            flat_sp = spatial_w.to(device=student.device, dtype=student.dtype).view(batch, -1)
            sp_mass = flat_sp.sum(dim=1)
            alive_sp = alive & (sp_mass > 0)
            img_scale = image_w.to(device=student.device, dtype=student.dtype).view(batch)
            bce_per_img = (flat_sp[alive_sp] * flat_bce[alive_sp]).sum(dim=1) / sp_mass[alive_sp].clamp_min(gate.eps)
            inter_sp = (flat_sp[alive_sp] * flat_p[alive_sp] * flat_t[alive_sp]).sum(dim=1)
            denom_sp = (flat_sp[alive_sp] * flat_p[alive_sp]).sum(dim=1) + (flat_sp[alive_sp] * flat_t[alive_sp]).sum(dim=1)
            dice_sp = (2.0 * inter_sp + gate.eps) / (denom_sp + gate.eps)
            num_alive = max(float(int(alive_sp.sum().item())), 1.0)
            bce_loss = (img_scale[alive_sp] * bce_per_img).sum() / num_alive
            dice_loss = (img_scale[alive_sp] * (1.0 - dice_sp)).sum() / num_alive
        else:
            bce_loss = (weights * bce_map).sum() / (weights.sum() + gate.eps)
            dice_loss = (mass[alive] * (1.0 - dice[alive])).sum() / mass[alive].sum().clamp_min(gate.eps)
        loss = gate.bce_weight * bce_loss + gate.dice_weight * dice_loss
    elif reduction == "sample":
        if sample_weight is None:
            raise ValueError("sample reduction requires sample_weight")
        sample_weight = sample_weight.to(device=student.device, dtype=student.dtype).view(-1)
        if sample_weight.numel() != batch:
            raise ValueError(
                f"sample_weight length {sample_weight.numel()} != batch {batch}"
            )
        raw_bce_i = (flat_w * flat_bce).sum(dim=1) / mass.clamp_min(gate.eps)
        raw_dice_i = 1.0 - dice
        zeros_b = torch.zeros(batch, device=student.device, dtype=student.dtype)
        bce_i = torch.where(alive, raw_bce_i, zeros_b)
        dice_i = torch.where(alive, raw_dice_i, zeros_b)
        per = gate.bce_weight * bce_i + gate.dice_weight * dice_i
        loss = (sample_weight * per).mean() * float(lambda_qw)
        mean_weight = (weights * sample_weight.view(-1, 1, 1, 1)).mean()
    elif reduction == "pixel":
        bce_loss = (weights * bce_map).mean()
        mean_w = flat_w.mean(dim=1)
        zeros_b = torch.zeros(batch, device=student.device, dtype=student.dtype)
        dice_i = torch.where(alive, 1.0 - dice, zeros_b)
        dice_loss = (mean_w * dice_i).mean()
        loss = (gate.bce_weight * bce_loss + gate.dice_weight * dice_loss) * float(lambda_qw)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    return GateResult(
        loss=loss,
        keep_fraction=keep,
        mean_weight=mean_weight,
        weights=weights,
        qw_alpha=float(alpha),
        sample_weight=None if sample_weight is None else sample_weight.detach(),
    )


def unlabeled_consistency_loss(
    student_probs: Tensor,
    teacher_probs: Tensor,
    quality: QualityEstimate,
    gate: Optional[GateConfig] = None,
    config: Optional[ReweightConfig] = None,
    *,
    step: int = 0,
    total_steps: int = 1,
) -> GateResult:
    """Unlabeled BCE + Dice with optional MQE reweight, entropy, or PL mask.

    Teacher targets and every quality weight are stop-grad. An all-zero
    weight returns ``student.sum() * 0``.
    """
    if gate is None:
        gate = GateConfig()
    if config is None:
        config = ReweightConfig()
    if _legacy_qgc(config, gate):
        result = quality_gated_consistency_loss(student_probs, teacher_probs, quality, gate)
        result.qw_alpha = 0.0
        return result

    student, teacher, target = _prepare_pair(student_probs, teacher_probs, gate)
    if config.mode == "off" and gate.mode in ("hard", "soft", "gated"):
        image_w, spatial_w = _image_and_pixel_gate(quality, gate)
        image_w = image_w.to(device=student.device, dtype=student.dtype)
        spatial_w = spatial_w.to(device=student.device, dtype=student.dtype)
        spatial_w = _apply_uncertainty(spatial_w, teacher, config)
        spatial_w = _apply_pl(spatial_w, teacher, config)
        weights = (image_w * spatial_w).clamp(0.0, 1.0)
        q = quality.q.detach().float().flatten().to(device=student.device)
        pixel = as_prob_maps(quality.pixel_weight).detach()
    else:
        image_w, spatial_w, q, pixel = _gate_factors(quality, gate, config.mode)
        image_w = image_w.to(device=student.device, dtype=student.dtype)
        spatial_w = spatial_w.to(device=student.device, dtype=student.dtype)
        spatial_w = _apply_uncertainty(spatial_w, teacher, config)
        spatial_w = _apply_pl(spatial_w, teacher, config)
        weights = (image_w * spatial_w).clamp(0.0, 1.0)

    if config.mode == "off":
        return _finish(
            student,
            target,
            weights,
            gate,
            alpha=0.0,
            lambda_qw=1.0,
            reduction="normalized",
            image_w=image_w,
            spatial_w=spatial_w,
        )

    alpha = qw_alpha(step, total_steps, config.ramp)
    if config.mode == "sample":
        sample_w = (1.0 - alpha) + alpha * q.to(dtype=student.dtype)
        return _finish(
            student,
            target,
            weights,
            gate,
            alpha=alpha,
            lambda_qw=float(config.lambda_qw),
            reduction="sample",
            sample_weight=sample_w,
        )

    mix = (1.0 - alpha) + alpha * pixel.to(device=student.device, dtype=student.dtype)
    return _finish(
        student,
        target,
        weights * mix,
        gate,
        alpha=alpha,
        lambda_qw=float(config.lambda_qw),
        reduction="pixel",
    )
