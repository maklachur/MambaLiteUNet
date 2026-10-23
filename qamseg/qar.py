"""Quality-Aware Regularization (QAR) and Feature Prototype Alignment (QG-PA).

Why QAM-Seg skips external-predictor QAR by default:
QG-SSL's QAR is ``mean(1 - g_φ(x, f_θ(x)))`` with gradients through a
frozen external quality network into the predicted mask. QAM-Seg has no
``g_φ``. Differentiating directly through output-probability MQE cues
would collapse into entropy minimization + consistency and fight BAQ.
Accordingly, ``quality_aware_regularization()`` with no feature map
continues to raise ``RuntimeError(QAR_REASON)`` and ``QAR_STATUS`` stays
``"skipped"`` by default.

To provide a mathematically sound, non-collapsing regularizer in
representation space (enabled via ``--ssl-qar prototype``), this module
implements **Quality-Guided Feature Prototype Alignment (QG-PA)** via
``prototype_quality_regularization``:
- High-reliability pixels (``pixel_weight >= high_tau``) from the weak
  teacher view define detached foreground and background feature prototypes
  in the decoder's penultimate representation space.
- Ambiguous contour-band pixels on the student view are pulled toward the
  corresponding class prototype via cosine similarity alignment without
  forcing premature sigmoid probability saturation on noisy edges.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from qamseg.mqe import QualityEstimate, as_prob_maps, boundary_band_mask

QAR_STATUS = "skipped"
QAR_REASON = (
    "QAM-Seg does not add a QAR loss. Without a separate g_phi, "
    "differentiating through MQE duplicates QGC and the entropy-weighted "
    "UA-MT baseline, and it conflicts with BAQ. "
    "See docs/MEDIA_NOTES.md."
)


def quality_aware_regularization(*_args, **_kwargs):
    """Refuse the external-network QG-SSL QAR objective. No g_phi network is built."""
    raise RuntimeError(QAR_REASON)


def prototype_quality_regularization(
    student_features: Tensor,
    teacher_probs: Tensor,
    quality: QualityEstimate,
    *,
    high_tau: float = 0.65,
    boundary_radius: int = 2,
    temperature: float = 0.1,
    eps: float = 1e-6,
) -> Tensor:
    """Quality-Guided Feature Prototype Alignment (QG-PA) on boundary representations.

    Args:
        student_features: ``(B, C, H_f, W_f)`` decoder features from the student.
        teacher_probs: ``(B, 1, H, W)`` detached teacher probabilities.
        quality: ``QualityEstimate`` containing detached ``pixel_weight``.
        high_tau: reliability threshold for selecting anchor pixels to form
            foreground and background prototypes.
        boundary_radius: pixel radius around the 0.5 contour where ambiguous
            student features are regularized toward class prototypes.
        temperature: cosine similarity temperature.
    """
    if student_features.ndim != 4:
        raise ValueError(
            f"Expected 4D student_features (B, C, H, W), got {tuple(student_features.shape)}"
        )
    B, C, H_f, W_f = student_features.shape
    teacher = as_prob_maps(teacher_probs).detach().to(
        device=student_features.device, dtype=student_features.dtype
    )
    pix_w = as_prob_maps(quality.pixel_weight).detach().to(
        device=student_features.device, dtype=student_features.dtype
    )

    if teacher.shape[-2:] != (H_f, W_f):
        teacher = F.interpolate(teacher, size=(H_f, W_f), mode="bilinear", align_corners=False)
        pix_w = F.interpolate(pix_w, size=(H_f, W_f), mode="bilinear", align_corners=False)

    # Normalize student feature vectors along channel axis
    feat_norm = F.normalize(student_features, p=2, dim=1, eps=eps)  # (B, C, H_f, W_f)
    feat_detached = feat_norm.detach()

    fg_anchor = ((teacher >= 0.5) & (pix_w >= float(high_tau))).to(dtype=student_features.dtype)
    bg_anchor = ((teacher < 0.5) & (pix_w >= float(high_tau))).to(dtype=student_features.dtype)

    fg_mass = fg_anchor.sum(dim=(2, 3), keepdim=True)  # (B, 1, 1, 1)
    bg_mass = bg_anchor.sum(dim=(2, 3), keepdim=True)  # (B, 1, 1, 1)
    valid_batch = (fg_mass.view(B) > 0) & (bg_mass.view(B) > 0)
    if not bool(valid_batch.any()):
        return student_features.sum() * 0.0

    # Detached class prototypes per image: (B, C, 1, 1)
    proto_fg = (feat_detached * fg_anchor).sum(dim=(2, 3), keepdim=True) / fg_mass.clamp_min(eps)
    proto_bg = (feat_detached * bg_anchor).sum(dim=(2, 3), keepdim=True) / bg_mass.clamp_min(eps)
    proto_fg = F.normalize(proto_fg, p=2, dim=1, eps=eps)
    proto_bg = F.normalize(proto_bg, p=2, dim=1, eps=eps)

    # Cosine similarity logits at every spatial pixel: (B, 1, H_f, W_f)
    sim_fg = (feat_norm * proto_fg).sum(dim=1, keepdim=True) / max(float(temperature), eps)
    sim_bg = (feat_norm * proto_bg).sum(dim=1, keepdim=True) / max(float(temperature), eps)
    proto_prob_fg = torch.sigmoid(sim_fg - sim_bg).clamp(eps, 1.0 - eps)

    # Focus prototype alignment on the contour band where boundary ambiguity lives
    band = boundary_band_mask(teacher, threshold=0.5, radius=max(1, int(boundary_radius)))
    valid_mask = band * valid_batch.view(B, 1, 1, 1).to(dtype=student_features.dtype)
    if not bool((valid_mask > 0).any()):
        return student_features.sum() * 0.0

    target_fg = (teacher >= 0.5).to(dtype=student_features.dtype)
    ce_map = F.binary_cross_entropy(proto_prob_fg, target_fg, reduction="none")
    return (ce_map * valid_mask).sum() / valid_mask.sum().clamp_min(eps)
