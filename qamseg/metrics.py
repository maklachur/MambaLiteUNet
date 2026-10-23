"""CPU copies of the supervised log metrics for the SSL smoke path.

Ratios match ``engine.py`` (pixel-pooled accuracy, sensitivity,
specificity, Dice / ``f1_or_dsc``, mIoU). Empty-class batches return 0
instead of indexing a 1×1 confusion matrix. HD95 is the same helper the
supervised val/test logs already call (``utils.summarize_hd95``). Values
on synthetic blobs are wiring checks, not dataset scores.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


def binary_pixel_metrics(pred, gt, threshold: float = 0.5, gt_threshold: float = 0.5) -> dict:
    y_pre = np.asarray(pred).reshape(-1) >= threshold
    y_true = np.asarray(gt).reshape(-1) >= gt_threshold
    tp = int(np.count_nonzero(y_pre & y_true))
    tn = int(np.count_nonzero(~y_pre & ~y_true))
    fp = int(np.count_nonzero(y_pre & ~y_true))
    fn = int(np.count_nonzero(~y_pre & y_true))
    total = tp + tn + fp + fn
    accuracy = (tn + tp) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    denom_f1 = 2 * tp + fp + fn
    f1_or_dsc = (2 * tp) / denom_f1 if denom_f1 else 0.0
    denom_iou = tp + fp + fn
    miou = tp / denom_iou if denom_iou else 0.0
    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_or_dsc": float(f1_or_dsc),
        "miou": float(miou),
    }


def hd95_for_log(preds: Sequence, gts: Sequence, threshold: float = 0.5) -> str:
    """Same text ``engine._hd95_for_log`` would append, without CUDA."""
    try:
        from utils import format_hd95, summarize_hd95
    except Exception as exc:
        return f"n/a ({type(exc).__name__}: {exc})"
    try:
        return format_hd95(summarize_hd95(preds, gts, threshold=threshold))
    except Exception as exc:
        return f"n/a ({type(exc).__name__}: {exc})"


def _distance_pair(pred, gt, threshold: float) -> dict:
    """Numeric HD95 / ASSD plus the log strings. Failures stay ``n/a``."""
    try:
        from utils import format_hd95, summarize_assd, summarize_hd95
    except Exception as exc:
        text = f"n/a ({type(exc).__name__}: {exc})"
        return {"hd95": text, "hd95_mean": None, "assd": text, "assd_mean": None}
    try:
        hd = summarize_hd95([pred], [gt], threshold=threshold)
        ad = summarize_assd([pred], [gt], threshold=threshold)
    except Exception as exc:
        text = f"n/a ({type(exc).__name__}: {exc})"
        return {"hd95": text, "hd95_mean": None, "assd": text, "assd_mean": None}
    return {
        "hd95": format_hd95(hd),
        "hd95_mean": _finite_mean(hd),
        "assd": format_hd95(ad),
        "assd_mean": _finite_mean(ad),
    }


def _finite_mean(summary: dict) -> Optional[float]:
    if not summary or summary.get("status") != "ok":
        return None
    mean = summary.get("mean")
    if mean is None:
        return None
    mean = float(mean)
    if math.isnan(mean) or math.isinf(mean):
        return None
    return mean


def format_ssl_log(row: dict) -> str:
    text = (
        f"ssl: step {row['step']}, loss: {row['loss']:.4f}, "
        f"sup: {row['loss_sup']:.4f}, unsup: {row['loss_unsup']:.4f}, "
        f"lambda_u: {row['lambda_u']:.4f}, "
        f"gate_keep_fraction: {row['gate_keep_fraction']:.4f}, "
        f"q_mean: {row['q_mean']:.4f}, tau: {row['tau']:.4f}"
    )
    if "quality_reweight" in row:
        text += f", reweight: {row['quality_reweight']}"
    if "qw_alpha" in row:
        text += f", qw_alpha: {float(row['qw_alpha']):.4f}"
    return text


def format_eval_log(metrics: dict) -> str:
    assd = metrics.get("assd", "n/a")
    return (
        f"ssl-eval: loss: {metrics['loss']:.4f}, miou: {metrics['miou']}, "
        f"f1_or_dsc: {metrics['f1_or_dsc']}, accuracy: {metrics['accuracy']}, "
        f"specificity: {metrics['specificity']}, sensitivity: {metrics['sensitivity']}, "
        f"hd95: {metrics['hd95']}, assd: {assd}"
    )
