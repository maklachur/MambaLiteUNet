"""Quality-aware SSL loop (MQE + QGC + optional QG-CutMix & Prototype QAR).

``python train.py`` with no flags never imports this module. ``--ssl`` and
``python -m qamseg.train_ssl`` do. The default run is a few CUDA steps on
synthetic blobs with MambaLiteUNet. ``mamba_ssm`` selective scan is
CUDA-only: if CUDA is unavailable the process exits and does not substitute
another student. That run checks the wiring. It is not an experiment and it
does not report a Dice or HD95 result. It does not need ISIC/HAM/PH2.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import copy
import json
from dataclasses import replace
from typing import Optional

import numpy as np
import torch
from torch import nn

from qamseg.aggregate import write_metrics_file
from qamseg.augment import (
    augment_labeled_batch,
    quality_guided_cutmix,
    strong_augment,
    weak_augment,
)
from qamseg.backbones import BackboneUnavailable, build_mamba_student
from qamseg.cli import parse_train_args, refuse_reserved_flags, resolve_ssl_method
from qamseg.data import (
    ArrayImageDataset,
    build_dual_loaders,
    build_ssl_datasets,
    iter_dual_batches,
    load_eval_arrays,
)
from qamseg.ema import ema_update
from qamseg.metrics import (
    _distance_pair,
    binary_pixel_metrics,
    format_eval_log,
    format_ssl_log,
)
from qamseg.mqe import apply_boundary_aware_prior, estimate_quality, gradient_magnitude
from qamseg.qar import (
    QAR_REASON,
    QAR_STATUS,
    prototype_quality_regularization,
)
from qamseg.qgc import GateConfig, lambda_u_at_step, tau_at_step
from qamseg.reweight import ReweightConfig, unlabeled_consistency_loss
from utils import BceDiceLoss


def ssl_training_step(
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    labeled_images: torch.Tensor,
    labeled_masks: torch.Tensor,
    unlabeled_images: torch.Tensor,
    *,
    step: int,
    total_steps: int,
    criterion_sup: nn.Module,
    gate: GateConfig,
    lambda_max: float = 1.0,
    tau0: float = 0.35,
    tau1: float = 0.55,
    warm_steps: int = 1,
    ema_alpha: float = 0.99,
    ema_warmup_steps: int = 0,
    noise_std: float = 0.0,
    use_baq: bool = False,
    quality_reweight: str = "off",
    lambda_qw: float = 1.0,
    qw_ramp: str = "gaussian",
    uncertainty: str = "off",
    uncertainty_tau: float = 0.75,
    pl_threshold: float = 0.0,
    use_cutmix: bool = False,
    qar_mode: str = "off",
    lambda_qar: float = 0.1,
    augment_labeled: bool = False,
    seed: int = 0,
    scheduler=None,
) -> dict:
    """One labeled step plus one quality-gated unlabeled step."""
    student.train()
    if augment_labeled:
        labeled_images, labeled_masks = augment_labeled_batch(
            labeled_images, labeled_masks, step=step, seed=seed
        )
    pred_labeled = student(labeled_images)
    loss_sup = criterion_sup(pred_labeled, labeled_masks)

    view_weak = weak_augment(unlabeled_images)
    view_strong = strong_augment(unlabeled_images, noise_std=noise_std)
    with torch.no_grad():
        teacher_weak = teacher(view_weak)
        teacher_aux = getattr(teacher, "last_aux", None) or {}
        mamba_disagreement = teacher_aux.get("branch_disagreement")
        student_weak = student(view_weak)
    pred_strong = student(view_strong)
    student_aux = getattr(student, "last_aux", None) or {}
    student_strong_features = student_aux.get("features")

    quality = estimate_quality(
        teacher_weak,
        student_probs=student_weak,
        view_probs=pred_strong.detach(),
        mamba_disagreement=mamba_disagreement,
    )
    if use_baq:
        grad = gradient_magnitude(view_weak)
        # Use (1 - entropy) as the boundary quality factor rather than passing
        # pixel_weight twice (which previously caused quadratic self-damping).
        baq_quality = (
            (1.0 - quality.entropy)
            if quality.entropy is not None
            else quality.pixel_weight
        )
        quality.pixel_weight = apply_boundary_aware_prior(
            quality.pixel_weight,
            grad,
            baq_quality,
        )

    if use_cutmix and unlabeled_images.shape[0] >= 2:
        view_strong, teacher_weak, quality = quality_guided_cutmix(
            view_strong,
            teacher_weak,
            quality,
            step=step,
            seed=seed,
        )
        pred_strong = student(view_strong)
        student_aux = getattr(student, "last_aux", None) or {}
        student_strong_features = student_aux.get("features")

    tau = tau_at_step(step, tau0, tau1, warm_steps)
    horizon = max(int(total_steps) - 1, 1)
    lam = lambda_u_at_step(step, horizon, lambda_max)
    reweight = ReweightConfig(
        mode=quality_reweight,
        lambda_qw=float(lambda_qw),
        ramp=qw_ramp,
        uncertainty=uncertainty,
        uncertainty_tau=float(uncertainty_tau),
        pl_threshold=float(pl_threshold),
    )
    qar_value = 0.0
    if lam == 0.0:
        loss = loss_sup
        unsup_value = 0.0
        keep_fraction = 0.0
        mean_weight = 0.0
        qw_alpha = 0.0
    else:
        gated = unlabeled_consistency_loss(
            pred_strong,
            teacher_weak,
            quality,
            replace(gate, tau=tau),
            reweight,
            step=step,
            total_steps=horizon,
        )
        loss = loss_sup + lam * gated.loss
        if qar_mode == "prototype" and student_strong_features is not None and float(lambda_qar) > 0:
            loss_qar = prototype_quality_regularization(
                student_strong_features,
                teacher_weak,
                quality,
            )
            loss = loss + lam * float(lambda_qar) * loss_qar
            qar_value = float(loss_qar.detach())
        unsup_value = float(gated.loss.detach())
        keep_fraction = float(gated.keep_fraction.detach())
        mean_weight = float(gated.mean_weight.detach())
        qw_alpha = float(gated.qw_alpha)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if step >= ema_warmup_steps:
        ema_update(teacher, student, alpha=ema_alpha)

    return {
        "step": int(step),
        "loss": float(loss.detach()),
        "loss_sup": float(loss_sup.detach()),
        "loss_unsup": unsup_value,
        "loss_qar": qar_value,
        "lambda_u": float(lam),
        "gate_keep_fraction": keep_fraction,
        "mean_gate_weight": mean_weight,
        "q_mean": float(quality.q.detach().mean()),
        "tau": float(tau),
        "quality_reweight": str(quality_reweight),
        "qw_alpha": qw_alpha,
        "lambda_qw": float(lambda_qw) if quality_reweight != "off" else 0.0,
        "uncertainty": str(uncertainty),
        "pl_threshold": float(pl_threshold),
    }


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader, criterion: nn.Module, threshold: float = 0.5) -> dict:
    """Labeled eval with the same metric names as the supervised logs."""
    model.eval()
    losses = []
    pred_maps = []
    gt_maps = []
    for images, masks in loader:
        out = model(images)
        if isinstance(out, tuple):
            out = out[0]
        losses.append(float(criterion(out, masks)))
        pred_maps.append(out[:, 0].detach().cpu().numpy())
        gt_maps.append(masks[:, 0].detach().cpu().numpy())
    pred = np.concatenate(pred_maps, axis=0)
    gt = np.concatenate(gt_maps, axis=0)
    metrics = binary_pixel_metrics(pred, gt, threshold=threshold)
    metrics.update(_distance_pair(pred, gt, threshold=threshold))
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def run_ssl_loop(
    student: nn.Module,
    labeled_loader,
    unlabeled_loader,
    val_loader,
    *,
    steps: int,
    gate: GateConfig,
    lambda_max: float,
    tau0: float,
    tau1: float,
    warm_steps: int,
    lr: float = 1e-3,
    seed: int = 0,
    use_baq: bool = False,
    quality_reweight: str = "off",
    lambda_qw: float = 1.0,
    qw_ramp: str = "gaussian",
    uncertainty: str = "off",
    uncertainty_tau: float = 0.75,
    pl_threshold: float = 0.0,
    use_cutmix: bool = False,
    qar_mode: str = "off",
    lambda_qar: float = 0.1,
    augment_labeled: bool = False,
) -> dict:
    torch.manual_seed(seed)
    teacher = copy.deepcopy(student).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(steps), 1), eta_min=lr * 0.01
    )
    # Prevent gate lock-out when --ssl-warm-steps was left at its smoke default (1)
    # on a multi-step training run (steps >= 10). ~30% of the horizon gives MQE q time to rise before tau1.
    effective_warm_steps = (
        max(1, int(round(0.3 * steps)))
        if (int(warm_steps) == 1 and int(steps) >= 10)
        else int(warm_steps)
    )
    criterion = BceDiceLoss()
    history = []
    step = 0
    while step < steps:
        for images_l, masks_l, images_u in iter_dual_batches(labeled_loader, unlabeled_loader):
            row = ssl_training_step(
                student,
                teacher,
                optimizer,
                images_l,
                masks_l,
                images_u,
                step=step,
                total_steps=steps,
                criterion_sup=criterion,
                gate=gate,
                lambda_max=lambda_max,
                tau0=tau0,
                tau1=tau1,
                warm_steps=effective_warm_steps,
                use_baq=use_baq,
                quality_reweight=quality_reweight,
                lambda_qw=lambda_qw,
                qw_ramp=qw_ramp,
                uncertainty=uncertainty,
                uncertainty_tau=uncertainty_tau,
                pl_threshold=pl_threshold,
                use_cutmix=use_cutmix,
                qar_mode=qar_mode,
                lambda_qar=lambda_qar,
                augment_labeled=augment_labeled,
                seed=seed,
                scheduler=scheduler,
            )
            history.append(row)
            print(format_ssl_log(row))
            step += 1
            if step >= steps:
                break
    metrics = evaluate_loader(student, val_loader, criterion)
    teacher_metrics = evaluate_loader(teacher, val_loader, criterion)
    print(format_eval_log(metrics))
    return {
        "history": history,
        "metrics": metrics,
        "teacher_metrics": teacher_metrics,
    }


def _validate_ssl_args(args) -> Optional[str]:
    if int(args.ssl_steps) < 1:
        return "--ssl-steps must be >= 1."
    if int(args.ssl_batch_size) < 1:
        return "--ssl-batch-size must be >= 1."
    if float(args.ssl_lambda_qw) < 0:
        return "--ssl-lambda-qw must be non-negative."
    if not 0.0 <= float(args.ssl_uncertainty_tau) <= 1.0:
        return "--ssl-uncertainty-tau must lie in [0, 1]."
    if not 0.0 <= float(args.ssl_pl_threshold) <= 1.0:
        return "--ssl-pl-threshold must lie in [0, 1]."
    if bool(args.ssl_eval_npy) != bool(args.ssl_eval_mask_npy):
        return "Pass both --ssl-eval-npy and --ssl-eval-mask-npy, or neither."
    if str(getattr(args, "ssl_backbone", "mamba")) != "mamba":
        return "--ssl-backbone must be mamba (MambaLiteUNet). No other student is substituted."
    synthetic = not (
        args.ssl_unlabeled_npy or args.ssl_labeled_npy or args.ssl_mask_npy or args.ssl_split
    )
    if synthetic and int(args.ssl_image_size) < 32:
        return (
            "--ssl-image-size must be at least 32 for MambaLiteUNet "
            "(five stride-2 pools). No other student is substituted."
        )
    return None


def _refuse_unsupported_device(device: str) -> bool:
    """Stop before building a student when MambaLiteUNet cannot run.

    Selective scan is CUDA-only. A CPU request or a missing GPU exits.
    Nothing is substituted.
    """
    if device == "cpu":
        print("QAM-Seg: --ssl-device cpu was requested.")
        print("The SSL student is MambaLiteUNet. mamba_ssm selective scan is CUDA-only.")
        print("No other student is substituted.")
        return True
    if not device.startswith("cuda") or not torch.cuda.is_available():
        print("QAM-Seg: --ssl-device cuda was requested but CUDA is not available.")
        print("The SSL student is MambaLiteUNet. mamba_ssm selective scan is CUDA-only.")
        print("No other student is substituted.")
        return True
    return False



def _gate_stats_from_history(history: list) -> dict:
    """Summarize per-step gate / quality rows for metrics.json."""
    if not history:
        return {
            "mean_gate_keep": None,
            "frac_gate_open": None,
            "frac_steps_gate_open": None,
            "mean_q": None,
            "mean_unsup": None,
            "first_zero_keep_step": None,
            "tau_end": None,
        }
    import math
    keeps = [float(r.get("gate_keep_fraction", 0.0)) for r in history]
    qs = [float(r.get("q_mean", float("nan"))) for r in history]
    unsups = [float(r.get("loss_unsup", r.get("unsup", 0.0))) for r in history]
    taus = [float(r.get("tau", float("nan"))) for r in history]
    first_zero = next(
        (int(r.get("step", i)) for i, r in enumerate(history) if float(r.get("gate_keep_fraction", 1.0)) <= 0.0),
        None,
    )
    qs_ok = [q for q in qs if not math.isnan(q)]
    frac_open = sum(1 for k in keeps if k > 0.0) / len(keeps)
    return {
        "mean_gate_keep": sum(keeps) / len(keeps),
        "frac_gate_open": frac_open,
        "frac_steps_gate_open": frac_open,  # alias
        "mean_q": (sum(qs_ok) / len(qs_ok)) if qs_ok else None,
        "mean_unsup": sum(unsups) / len(unsups),
        "first_zero_keep_step": first_zero,
        "tau_end": taus[-1] if taus else None,
        "n_steps_logged": len(history),
    }


def _metrics_payload(
    args,
    metrics: dict,
    *,
    synthetic: bool,
    eval_split: str,
    backbone_name: str,
    teacher_metrics: Optional[dict] = None,
    gate_stats: Optional[dict] = None,
) -> dict:
    dice = metrics.get("f1_or_dsc")
    qar_mode = getattr(args, "ssl_qar", "off")
    qar_field = QAR_STATUS if qar_mode == "off" else str(qar_mode)
    payload = {
        "recipe": str(args.ssl_recipe or args.ssl_method),
        "method": str(args.ssl_method),
        "group": str(args.ssl_group),
        "seed": int(args.ssl_seed),
        "fraction_token": str(args.ssl_fraction),
        "fraction": str(args.ssl_fraction),
        "backbone": backbone_name,
        "gate": str(args.ssl_gate),
        "baq": bool(args.ssl_baq),
        "cutmix": bool(getattr(args, "ssl_cutmix", False)),
        "quality_reweight": str(args.quality_reweight),
        "lambda_qw": float(args.ssl_lambda_qw) if args.quality_reweight != "off" else 0.0,
        "qw_ramp": str(args.ssl_qw_ramp),
        "uncertainty": str(args.ssl_uncertainty),
        "uncertainty_tau": float(args.ssl_uncertainty_tau),
        "pl_threshold": float(args.ssl_pl_threshold),
        "lambda_max": float(args.ssl_lambda_max),
        "data": "synthetic" if synthetic else "local_npy",
        "eval_split": eval_split,
        "experimentally_validated": False,
        "qar": qar_field,
        "steps": int(args.ssl_steps),
        "metrics": {
            "dice": dice,
            "f1_or_dsc": dice,
            "miou": metrics.get("miou"),
            "hd95": metrics.get("hd95"),
            "hd95_mean": metrics.get("hd95_mean"),
            "assd": metrics.get("assd"),
            "assd_mean": metrics.get("assd_mean"),
            "loss": metrics.get("loss"),
            "accuracy": metrics.get("accuracy"),
            "sensitivity": metrics.get("sensitivity"),
            "specificity": metrics.get("specificity"),
        },
    }
    if teacher_metrics is not None:
        t_dice = teacher_metrics.get("f1_or_dsc")
        payload["teacher_metrics"] = {
            "dice": t_dice,
            "f1_or_dsc": t_dice,
            "miou": teacher_metrics.get("miou"),
            "hd95": teacher_metrics.get("hd95"),
            "hd95_mean": teacher_metrics.get("hd95_mean"),
            "assd": teacher_metrics.get("assd"),
            "assd_mean": teacher_metrics.get("assd_mean"),
            "loss": teacher_metrics.get("loss"),
        }
    payload["tau0"] = float(args.ssl_tau0)
    payload["tau1"] = float(args.ssl_tau1)
    payload["warm_steps_requested"] = int(args.ssl_warm_steps)
    if gate_stats:
        payload["gate_stats"] = gate_stats
        # Flat aliases for tables / quick greps
        payload["mean_gate_keep"] = gate_stats.get("mean_gate_keep")
        payload["frac_gate_open"] = gate_stats.get("frac_gate_open", gate_stats.get("frac_steps_gate_open"))
        payload["mean_q"] = gate_stats.get("mean_q")
    return payload


def run_from_args(args) -> int:
    problem = _validate_ssl_args(args)
    if problem:
        print(f"QAM-Seg: {problem}")
        return 2
    args = resolve_ssl_method(args)
    device = str(args.ssl_device)
    if _refuse_unsupported_device(device):
        return 2

    synthetic = not (
        args.ssl_unlabeled_npy or args.ssl_labeled_npy or args.ssl_mask_npy or args.ssl_split
    )
    qar_mode = str(getattr(args, "ssl_qar", "off"))
    print("QAM-Seg SSL path: MQE + QGC.")
    print("Implemented in code and exercised on this process only.")
    print("Not experimentally validated. No ISIC, HAM10000, or PH2 download.")
    print("No CVPR MambaLiteUNet metric is computed or claimed.")
    if qar_mode == "off":
        print(f"QAR: {QAR_STATUS}. {QAR_REASON}")
    else:
        print(f"QAR: {qar_mode} (Quality-Guided Feature Prototype Alignment, lambda_qar={float(getattr(args, 'ssl_lambda_qar', 0.1))}).")
    print(
        "SSL method: "
        f"{args.ssl_method}, gate: {args.ssl_gate}, "
        f"reweight: {args.quality_reweight}, baq: {bool(args.ssl_baq)}, "
        f"uncertainty: {args.ssl_uncertainty}, pl_threshold: {float(args.ssl_pl_threshold)}."
    )
    if args.ssl_method != "qam":
        print("Baseline preset replaced gate, BAQ, and MQE reweight. Use --ssl-method qam to set those yourself.")
    if float(args.ssl_lambda_max) == 0.0:
        print("Supervised objective: unlabeled consistency is not applied.")
    if args.use_learned_quality_head:
        print("Learned MQE head flag is set, but the head stays a stub (weight a3 = 0, beta = 0).")

    use_amf = not (bool(getattr(args, "no_amf", False)) or bool(getattr(args, "ablate_amf", False)))
    use_lgfm = not (bool(getattr(args, "no_lgfm", False)) or bool(getattr(args, "ablate_lgfm", False)))
    use_cga = not (bool(getattr(args, "no_cga", False)) or bool(getattr(args, "ablate_cga", False)))
    scan_mode = str(getattr(args, "ssl_scan_mode", "four_way"))
    global_mode = str(getattr(args, "ssl_global_mode", "attn"))

    try:
        student = build_mamba_student(
            use_amf=use_amf,
            use_lgfm=use_lgfm,
            use_cga=use_cga,
            scan_mode=scan_mode,
            global_mode=global_mode,
        )
    except BackboneUnavailable as exc:
        print(exc)
        return 2
    print("Student: MambaLiteUNet. Import succeeded.")
    print("This is not a CVPR reproduction. Logs from this process are not medical Dice or HD95 results.")

    if synthetic:
        print("Data: synthetic blobs.")
    elif args.ssl_split:
        print("Data: local train arrays sliced by --ssl-split. Unlabeled rows do not keep masks.")
        print("Validation and test images are not added. Eval on the labeled subset is a wiring check, not a test score.")
    else:
        print("Data: local npy arrays passed on the CLI.")

    length = max(int(args.ssl_batch_size) * 2, int(args.ssl_batch_size))
    try:
        labeled, unlabeled = build_ssl_datasets(
            length=length,
            size=int(args.ssl_image_size),
            seed=int(args.ssl_seed),
            unlabeled_npy=args.ssl_unlabeled_npy,
            labeled_npy=args.ssl_labeled_npy,
            mask_npy=args.ssl_mask_npy,
            split_json=args.ssl_split,
            augment_labeled=not synthetic,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"QAM-Seg: {exc}")
        return 2
    labeled_loader, unlabeled_loader = build_dual_loaders(
        labeled,
        unlabeled,
        batch_size=int(args.ssl_batch_size),
        shuffle=not synthetic,
        seed=int(args.ssl_seed),
    )
    if len(labeled_loader) == 0 or len(unlabeled_loader) == 0:
        print("QAM-Seg: a loader is empty after drop_last.")
        print("Lower --ssl-batch-size, or pass a local array with at least one full batch.")
        print("This entry point does not download ISIC, HAM10000, or PH2.")
        return 2
    eval_split = "synthetic_labeled" if synthetic else "labeled_train_subset"
    try:
        if args.ssl_eval_npy:
            eval_images, eval_masks = load_eval_arrays(args.ssl_eval_npy, args.ssl_eval_mask_npy)
            val_labeled = ArrayImageDataset(eval_images, eval_masks)
            eval_split = "held_out"
            print("Eval: held-out arrays from --ssl-eval-npy. They are not added to the unlabeled pool.")
            print("This is still not a published Dice, HD95, or ASSD result.")
        else:
            val_labeled, _val_unlabeled = build_ssl_datasets(
                length=length,
                size=int(args.ssl_image_size),
                seed=int(args.ssl_seed) + 17,
                unlabeled_npy="",
                labeled_npy=args.ssl_labeled_npy,
                mask_npy=args.ssl_mask_npy,
                split_json=args.ssl_split,
                augment_labeled=False,
            )
            if not synthetic:
                print("Eval: labeled train rows used for this run. Not a held-out test score.")
                print("Pass --ssl-eval-npy and --ssl-eval-mask-npy when a local val or test array exists.")
    except (FileNotFoundError, ValueError) as exc:
        print(f"QAM-Seg: {exc}")
        return 2
    val_loader, _ = build_dual_loaders(
        val_labeled,
        val_labeled,
        batch_size=int(args.ssl_batch_size),
        shuffle=False,
    )
    if len(val_loader) == 0:
        print("QAM-Seg: the eval loader is empty after drop_last.")
        print("Lower --ssl-batch-size. This entry point does not download a dataset.")
        return 2
    if device == "cuda":
        student = student.cuda()
        labeled_loader = _DeviceLoader(labeled_loader, device)
        unlabeled_loader = _DeviceLoader(unlabeled_loader, device)
        val_loader = _DeviceLoader(val_loader, device)

    gate = GateConfig(mode=args.ssl_gate, level=args.ssl_level, gamma=float(args.ssl_gamma))
    result = run_ssl_loop(
        student,
        labeled_loader,
        unlabeled_loader,
        val_loader,
        steps=int(args.ssl_steps),
        gate=gate,
        lambda_max=float(args.ssl_lambda_max),
        tau0=float(args.ssl_tau0),
        tau1=float(args.ssl_tau1),
        warm_steps=int(args.ssl_warm_steps),
        seed=int(args.ssl_seed),
        use_baq=bool(args.ssl_baq),
        quality_reweight=str(args.quality_reweight),
        lambda_qw=float(args.ssl_lambda_qw),
        qw_ramp=str(args.ssl_qw_ramp),
        uncertainty=str(args.ssl_uncertainty),
        uncertainty_tau=float(args.ssl_uncertainty_tau),
        pl_threshold=float(args.ssl_pl_threshold),
        use_cutmix=bool(getattr(args, "ssl_cutmix", False)),
        qar_mode=qar_mode,
        lambda_qar=float(getattr(args, "ssl_lambda_qar", 0.1)),
        augment_labeled=not synthetic,
    )
    if not result["history"]:
        print("QAM-Seg: SSL loop produced no steps.")
        return 1
    if args.ssl_out_dir:
        gate_stats = _gate_stats_from_history(result.get("history") or [])
        payload = _metrics_payload(
            args,
            result["metrics"],
            synthetic=synthetic,
            eval_split=eval_split,
            backbone_name="mamba",
            teacher_metrics=result.get("teacher_metrics"),
            gate_stats=gate_stats,
        )
        destination = os.path.join(str(args.ssl_out_dir), "metrics.json")
        write_metrics_file(destination, payload)
        print(f"Wrote {destination}")
        history_path = os.path.join(str(args.ssl_out_dir), "train_log.jsonl")
        with open(history_path, "w", encoding="utf-8") as handle:
            for row in result.get("history") or []:
                serializable = {}
                for key, value in row.items():
                    if hasattr(value, "item"):
                        try:
                            serializable[key] = float(value.item())
                            continue
                        except Exception:
                            pass
                    serializable[key] = value if isinstance(value, (int, float, str, bool, type(None))) else str(value)
                handle.write(json.dumps(serializable) + "\n")
        print(
            "gate_stats: "
            f"mean_keep={gate_stats.get('mean_gate_keep')} "
            f"frac_open={gate_stats.get('frac_gate_open', gate_stats.get('frac_steps_gate_open'))} "
            f"mean_q={gate_stats.get('mean_q')} "
            f"first_zero_keep_step={gate_stats.get('first_zero_keep_step')}"
        )
        print(f"Wrote {history_path}")
        print("metrics.json is a wiring record unless a held-out eval array was passed and the run is a real training job.")
    return 0


class _DeviceLoader:
    """Move loader batches to ``device``. Used only when CUDA was requested."""

    def __init__(self, loader, device: str):
        self.loader = loader
        self.device = device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            yield _move(batch, self.device)


def _move(batch, device: str):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, (tuple, list)):
        return type(batch)(_move(item, device) for item in batch)
    return batch


def main(argv=None) -> int:
    args, unknown, reserved = parse_train_args(argv)
    if unknown:
        print("QAM-Seg: ignoring unrecognized arguments: " + " ".join(unknown))
    refuse_reserved_flags(reserved)
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
