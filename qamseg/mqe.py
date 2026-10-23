"""Multi-cue Quality Estimator (MQE).

Pure functions on probability maps and optional Mamba directional branch
disagreement maps.

Cues, all in ``[0, 1]`` after scaling:

- predictive entropy of a Bernoulli (sigmoid) mask, divided by ``ln 2``
- teacher–student absolute disagreement on a shared view
- weak/strong pseudo-mask Dice agreement (on spatially aligned views)
- mean entropy inside a band around the 0.5 contour
- optional Mamba 4-way directional scan disagreement (from AMF branches)

Default fusion is a weighted average of the active cues (the linear part
of the design combination, already in ``[0, 1]``). ``fusion="sigmoid"``
applies the written ``σ(a·cues)`` form. With non-negative weights that
sum is non-negative, so a plain sigmoid cannot fall below 0.5;
``center=True`` subtracts half the active weight sum first so a threshold
can sit on both sides of the score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


_LN2 = math.log(2.0)


@dataclass(frozen=True)
class QualityWeights:
    """Non-negative cue weights. ``learned`` defaults to 0 (head off)."""

    entropy: float = 1.0
    disagreement: float = 1.0
    agreement: float = 1.0
    boundary: float = 1.0
    mamba: float = 1.0
    learned: float = 0.0

    def __post_init__(self):
        for name in ("entropy", "disagreement", "agreement", "boundary", "mamba", "learned"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"MQE weight {name} must be non-negative, got {value}")


@dataclass
class QualityEstimate:
    """Image score ``q`` and a pixel weight map, both in ``[0, 1]``."""

    q: Tensor
    pixel_weight: Tensor
    entropy: Optional[Tensor] = None
    disagreement: Optional[Tensor] = None
    agreement: Optional[Tensor] = None
    boundary_entropy: Optional[Tensor] = None
    learned_score: Optional[Tensor] = None
    mamba_disagreement: Optional[Tensor] = None


class LearnedQualityHead(nn.Module):
    """1×1 map from teacher features to a reliability score in ``(0, 1)``."""

    def __init__(self, in_channels: int):
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        self.proj = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        return torch.sigmoid(self.proj(features))


def as_prob_maps(probs: Tensor) -> Tensor:
    """Reshape a mask to ``(B, 1, H, W)`` float."""
    if not torch.is_tensor(probs):
        raise TypeError("MQE expected a torch.Tensor probability map")
    if probs.ndim == 2:
        probs = probs.unsqueeze(0).unsqueeze(0)
    elif probs.ndim == 3:
        probs = probs.unsqueeze(1)
    elif probs.ndim == 4:
        if probs.shape[1] != 1:
            raise ValueError(
                f"Expected a single-channel mask, got shape {tuple(probs.shape)}"
            )
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D probabilities, got {probs.ndim}D")
    return probs.float()


def bernoulli_entropy(probs: Tensor, eps: float = 1e-6) -> Tensor:
    """Pixel Bernoulli entropy in nats. Shape matches ``as_prob_maps``."""
    p = as_prob_maps(probs).clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())


def normalized_entropy(probs: Tensor, eps: float = 1e-6) -> Tensor:
    """Bernoulli entropy divided by ``ln 2``, clipped to ``[0, 1]``."""
    return (bernoulli_entropy(probs, eps=eps) / _LN2).clamp(0.0, 1.0)


def absolute_disagreement(teacher_probs: Tensor, student_probs: Tensor) -> Tensor:
    """``|p_teacher - p_student|`` on a shared view. Values lie in ``[0, 1]``."""
    teacher = as_prob_maps(teacher_probs)
    student = as_prob_maps(student_probs)
    if teacher.shape != student.shape:
        raise ValueError(
            f"Disagreement shape mismatch: {tuple(teacher.shape)} vs {tuple(student.shape)}"
        )
    return (teacher - student).abs()


def dice_agreement(
    weak_probs: Tensor,
    strong_probs: Tensor,
    mode: str = "hard",
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Tensor:
    """Per-image Dice between a weak view and a strong view.

    ``mode="hard"`` thresholds both maps at 0.5 (pseudo-masks).
    ``mode="soft"`` uses the probabilities themselves.
    Both-empty maps score 1. Returns shape ``(B,)`` in ``[0, 1]``.
    """
    weak = as_prob_maps(weak_probs)
    strong = as_prob_maps(strong_probs)
    if weak.shape != strong.shape:
        raise ValueError(
            f"Agreement shape mismatch: {tuple(weak.shape)} vs {tuple(strong.shape)}"
        )
    if mode == "hard":
        weak = (weak >= threshold).float()
        strong = (strong >= threshold).float()
    elif mode != "soft":
        raise ValueError(f"Unknown agreement mode: {mode}")
    return _soft_dice_per_image(weak, strong, eps=eps)


def _soft_dice_per_image(left: Tensor, right: Tensor, eps: float) -> Tensor:
    left_flat = left.flatten(1)
    right_flat = right.flatten(1)
    intersection = (left_flat * right_flat).sum(dim=1)
    denom = left_flat.sum(dim=1) + right_flat.sum(dim=1)
    both_empty = denom <= eps
    dice = (2.0 * intersection + eps) / (denom + eps)
    dice = torch.where(both_empty, torch.ones_like(dice), dice)
    return dice.clamp(0.0, 1.0)


def boundary_band_mask(probs: Tensor, threshold: float = 0.5, radius: int = 1) -> Tensor:
    """Binary band around the 0.5 contour.

    Outside the image counts as background, so a mask that touches the
    border still has a band. ``radius`` is in pixels and must be >= 1.
    """
    if radius < 1:
        raise ValueError(f"boundary radius must be >= 1, got {radius}")
    binary = (as_prob_maps(probs) >= threshold).float()
    dilated = _dilate(binary, radius)
    eroded = _erode(binary, radius)
    return (dilated - eroded).clamp(0.0, 1.0)


def _dilate(binary: Tensor, radius: int) -> Tensor:
    padded = F.pad(binary, (radius, radius, radius, radius), value=0.0)
    kernel = 2 * radius + 1
    return F.max_pool2d(padded, kernel_size=kernel, stride=1, padding=0)


def _erode(binary: Tensor, radius: int) -> Tensor:
    # Outside is background, so the inverse is padded with ones.
    inverted = 1.0 - binary
    padded = F.pad(inverted, (radius, radius, radius, radius), value=1.0)
    kernel = 2 * radius + 1
    dilated_inv = F.max_pool2d(padded, kernel_size=kernel, stride=1, padding=0)
    return 1.0 - dilated_inv


def boundary_band_entropy(
    probs: Tensor,
    threshold: float = 0.5,
    radius: int = 1,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Mean normalized entropy on the contour band.

    Returns ``(band_entropy (B,), band (B, 1, H, W))``. An empty band
    falls back to the full-image mean so the reduction never divides by
    zero.
    """
    entropy = normalized_entropy(probs, eps=eps)
    band = boundary_band_mask(probs, threshold=threshold, radius=radius)
    band_mass = band.sum(dim=(1, 2, 3))
    weighted = (entropy * band).sum(dim=(1, 2, 3))
    image_mean = entropy.mean(dim=(1, 2, 3))
    empty = band_mass <= 0
    band_entropy = torch.where(
        empty,
        image_mean,
        weighted / band_mass.clamp_min(eps),
    )
    return band_entropy.clamp(0.0, 1.0), band


def reliability_target(
    probs: Tensor,
    labels: Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Tensor:
    """Labeled reliability target ``r = 1[correct] * (1 - H)``."""
    pred = as_prob_maps(probs)
    target = as_prob_maps(labels)
    if pred.shape != target.shape:
        raise ValueError(
            f"Reliability shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    correct = ((pred >= threshold) == (target >= threshold)).float()
    confidence = 1.0 - normalized_entropy(pred, eps=eps)
    return correct * confidence


def gradient_magnitude(
    image: Tensor,
    eps: float = 1e-6,
    smooth: Optional[bool] = None,
) -> Tensor:
    """Per-image min–max normalized gradient magnitude, shape ``(B, 1, H, W)``.

    For full-resolution dermoscopy images (spatial size >= 32), applies a 3x3
    smoothing filter prior to finite differences so thin hair artifacts do not
    dominate the boundary prior.
    """
    if image.ndim == 3:
        image = image.unsqueeze(1)
    if image.ndim != 4:
        raise ValueError(f"Expected an image tensor, got shape {tuple(image.shape)}")
    work = image.float()
    if smooth is None:
        smooth = min(int(work.shape[-2]), int(work.shape[-1])) >= 32
    if smooth:
        work = F.avg_pool2d(work, kernel_size=3, stride=1, padding=1)
    dx = torch.zeros_like(work)
    dy = torch.zeros_like(work)
    dx[..., :, 1:] = work[..., :, 1:] - work[..., :, :-1]
    dy[..., 1:, :] = work[..., 1:, :] - work[..., :-1, :]
    magnitude = torch.sqrt(dx.square() + dy.square() + eps).mean(dim=1, keepdim=True)
    low = magnitude.amin(dim=(2, 3), keepdim=True)
    high = magnitude.amax(dim=(2, 3), keepdim=True)
    return ((magnitude - low) / (high - low).clamp_min(eps)).clamp(0.0, 1.0)


def apply_boundary_aware_prior(
    pixel_weight: Tensor,
    grad_mag: Tensor,
    quality_pixels: Tensor,
) -> Tensor:
    """Down-weight pixels where the image gradient is high and quality is low.

    Optional BAQ factor ``1 - grad * (1 - q_pix)``. The result stays in ``[0, 1]``.
    """
    weight = as_prob_maps(pixel_weight).clamp(0.0, 1.0)
    grad = as_prob_maps(grad_mag).clamp(0.0, 1.0)
    quality = as_prob_maps(quality_pixels).clamp(0.0, 1.0)
    if weight.shape != grad.shape or weight.shape != quality.shape:
        raise ValueError(
            "BAQ shape mismatch: "
            f"weight {tuple(weight.shape)}, grad {tuple(grad.shape)}, "
            f"quality {tuple(quality.shape)}"
        )
    factor = 1.0 - grad * (1.0 - quality)
    return (weight * factor).clamp(0.0, 1.0)


def _fuse(parts: list[tuple[Tensor, float]], fusion: str, center: bool) -> Tensor:
    active = [(value, weight) for value, weight in parts if value is not None and weight > 0]
    if not active:
        raise ValueError("MQE fusion has no active cues")
    weight_sum = sum(weight for _, weight in active)
    acc = None
    for value, weight in active:
        term = weight * value
        acc = term if acc is None else acc + term
    if fusion == "linear":
        return (acc / weight_sum).clamp(0.0, 1.0)
    if fusion == "sigmoid":
        raw = acc - (0.5 * weight_sum if center else 0.0)
        return torch.sigmoid(raw)
    raise ValueError(f"Unknown fusion mode: {fusion}")


def estimate_quality(
    teacher_probs: Tensor,
    student_probs: Optional[Tensor] = None,
    view_probs: Optional[Tensor] = None,
    learned_score: Optional[Tensor] = None,
    mamba_disagreement: Optional[Tensor] = None,
    weights: Optional[QualityWeights] = None,
    boundary_radius: int = 1,
    agreement_mode: str = "hard",
    fusion: str = "linear",
    center: bool = True,
    eps: float = 1e-6,
) -> QualityEstimate:
    """Fuse MQE cues into an image score and a pixel weight map.

    Inputs are detached. ``student_probs`` is the student on the same
    (weak) view as the teacher. ``view_probs`` is the other view used for
    Dice agreement, aligned to the same coordinate frame.
    ``mamba_disagreement`` is the optional 4-way directional state-space
    disagreement map from ``MambaLiteUNet``'s AMF blocks.
    """
    if weights is None:
        weights = QualityWeights()
    if fusion not in ("linear", "sigmoid"):
        raise ValueError(f"Unknown fusion mode: {fusion}")

    teacher = as_prob_maps(teacher_probs).detach()
    entropy = normalized_entropy(teacher, eps=eps)
    disagreement = None
    if student_probs is not None:
        disagreement = absolute_disagreement(teacher, as_prob_maps(student_probs).detach())
    agreement = None
    if view_probs is not None:
        agreement = dice_agreement(
            teacher,
            as_prob_maps(view_probs).detach(),
            mode=agreement_mode,
            eps=eps,
        )
    boundary_entropy, _band = boundary_band_entropy(
        teacher,
        radius=boundary_radius,
        eps=eps,
    )
    mamba_d = None
    if mamba_disagreement is not None and weights.mamba > 0:
        mamba_d = as_prob_maps(mamba_disagreement).detach().clamp(0.0, 1.0)
        if mamba_d.shape[-2:] != teacher.shape[-2:]:
            mamba_d = F.interpolate(
                mamba_d, size=teacher.shape[-2:], mode="bilinear", align_corners=False
            ).clamp(0.0, 1.0)
        if mamba_d.shape[0] != teacher.shape[0]:
            raise ValueError(
                f"Mamba disagreement batch {mamba_d.shape[0]} != teacher {teacher.shape[0]}"
            )
    learned = None
    if learned_score is not None and weights.learned > 0:
        learned = as_prob_maps(learned_score).detach().clamp(0.0, 1.0)
        if learned.shape != teacher.shape:
            raise ValueError(
                f"Learned score shape {tuple(learned.shape)} != {tuple(teacher.shape)}"
            )

    image_parts: list[tuple[Tensor, float]] = [
        (1.0 - entropy.mean(dim=(1, 2, 3)), weights.entropy),
        (1.0 - boundary_entropy, weights.boundary),
    ]
    if disagreement is not None:
        image_parts.append((1.0 - disagreement.mean(dim=(1, 2, 3)), weights.disagreement))
    if agreement is not None:
        image_parts.append((agreement, weights.agreement))
    if mamba_d is not None:
        image_parts.append((1.0 - mamba_d.mean(dim=(1, 2, 3)), weights.mamba))
    if learned is not None:
        image_parts.append((learned.mean(dim=(1, 2, 3)), weights.learned))

    pixel_parts: list[tuple[Tensor, float]] = [(1.0 - entropy, weights.entropy)]
    if disagreement is not None:
        pixel_parts.append((1.0 - disagreement, weights.disagreement))
    if mamba_d is not None:
        pixel_parts.append((1.0 - mamba_d, weights.mamba))
    if learned is not None:
        pixel_parts.append((learned, weights.learned))

    q = _fuse(image_parts, fusion=fusion, center=center)
    pixel_weight = _fuse(pixel_parts, fusion=fusion, center=center)
    return QualityEstimate(
        q=q,
        pixel_weight=pixel_weight,
        entropy=entropy,
        disagreement=disagreement,
        agreement=agreement,
        boundary_entropy=boundary_entropy,
        learned_score=learned,
        mamba_disagreement=mamba_d,
    )
