#!/usr/bin/env python3
"""Multi-seed MedIA protocol for QAM-Seg.

Default matrix: seeds 0, 1, 2; labeled fractions 5/10/20; supervised full;
plain mean teacher; entropy UA-MT; confidence-threshold PL; and the QAM
ablation ungated / gated / gated+BAQ / gated+BAQ+sample-reweight, plus
the ungated sample-reweight PL-QW analogue (``qam-qw``).

``--dry-run`` writes the command list and does not train.
``--smoke`` runs one synthetic CUDA step per selected job on MambaLiteUNet.
It does not read ISIC, HAM10000, or PH2. There is no CPU student.

A real run needs local train arrays and CUDA. This script does not download them.

    python scripts/run_media_protocol.py --dry-run --out-dir results/media_protocol
    python scripts/run_media_protocol.py --smoke --recipes qam-qw,mt --seeds 0 --fractions 10
    python scripts/run_media_protocol.py \\
        --train-npy ./data/ISIC2018/data_train.npy \\
        --mask-npy ./data/ISIC2018/mask_train.npy \\
        --eval-npy ./data/ISIC2018/data_test.npy \\
        --eval-mask-npy ./data/ISIC2018/mask_test.npy \\
        --steps 20000 --out-dir results/media_isic2018
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qamseg.protocol import (  # noqa: E402
    ABLATION_NAMES,
    DEFAULT_FRACTIONS,
    DEFAULT_RECIPE_NAMES,
    DEFAULT_SEEDS,
    iter_jobs,
    job_argv,
    parse_fractions,
    parse_recipe_names,
    parse_seeds,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QAM-Seg MedIA multi-seed protocol. Does not download data.")
    parser.add_argument("--out-dir", default="results/media_protocol")
    parser.add_argument("--seeds", default="default", help="Comma-separated seeds. Default: 0,1,2.")
    parser.add_argument("--fractions", default="default", help="Labeled percents. Default: 5,10,20. Supervised also runs full.")
    parser.add_argument(
        "--recipes",
        default="default",
        help="Comma-separated recipe names, or 'default'. Pixel reweight is qam-qw-pixel.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Optimizer steps. Required for a real run. Not a tuned budget.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cpu", "cuda"),
        help=(
            "Passed through as --ssl-device. Default cuda. "
            "cpu is refused: MambaLiteUNet selective scan is CUDA-only."
        ),
    )
    parser.add_argument(
        "--backbone",
        default="mamba",
        choices=("mamba",),
        help="SSL student. mamba is MambaLiteUNet. A failed import exits non-zero. No other student is substituted.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Synthetic smoke spatial size. MambaLiteUNet needs at least 32. Ignored for .npy arrays.",
    )
    parser.add_argument("--train-npy", default="", help="Local data_train.npy. Refuses val/test names. No download.")
    parser.add_argument("--mask-npy", default="", help="Local mask_train.npy paired with --train-npy.")
    parser.add_argument("--eval-npy", default="", help="Optional held-out image array (data_val.npy or data_test.npy).")
    parser.add_argument("--eval-mask-npy", default="")
    parser.add_argument("--lambda-qw", type=float, default=1.0, help="Passed through. Default 1. QG-SSL used 0.25.")
    parser.add_argument("--qw-ramp", default="gaussian", choices=("gaussian", "linear", "none"))
    parser.add_argument("--uncertainty-tau", type=float, default=0.75)
    parser.add_argument("--pl-threshold", type=float, default=0.95)
    parser.add_argument("--dry-run", action="store_true", help="Write protocol_manifest.json and do not train.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Synthetic CUDA jobs on MambaLiteUNet. Does not read a medical array.",
    )
    return parser


def _cuda_student_ready(device: str) -> bool:
    """MambaLiteUNet selective scan is CUDA-only. Nothing is substituted."""
    if device != "cuda":
        print("QAM-Seg: --device cpu was requested.")
        print("The protocol student is MambaLiteUNet. mamba_ssm selective scan is CUDA-only.")
        print("No other student is substituted.")
        return False
    try:
        import torch
    except ImportError:
        print("QAM-Seg: PyTorch is not importable.")
        print("The protocol student is MambaLiteUNet. mamba_ssm selective scan is CUDA-only.")
        print("No other student is substituted.")
        return False
    if not torch.cuda.is_available():
        print("QAM-Seg: CUDA is not available.")
        print("The protocol student is MambaLiteUNet. mamba_ssm selective scan is CUDA-only.")
        print("No other student is substituted.")
        return False
    return True


def _job_dir(out_dir: Path, seed: int, recipe: str, fraction_token: str) -> Path:
    return out_dir / f"seed{seed}" / recipe / f"fraction_{fraction_token}"


def _split_map(train_npy: str, out_dir: Path, seeds, fractions) -> dict:
    from qamseg.protocol import fraction_to_float
    from qamseg.splits import build_and_write, train_ids_from_npy

    ids = train_ids_from_npy(train_npy)
    mapping = {}
    fraction_values = tuple(fraction_to_float(int(percent)) for percent in fractions)
    for seed in seeds:
        destination = build_and_write(
            ids,
            out_dir / "splits" / f"seed{seed}",
            seed=int(seed),
            fractions=fraction_values,
            source={"kind": "train-npy", "path": str(Path(train_npy).resolve())},
        )
        for percent in fractions:
            token = str(int(percent))
            mapping[(int(seed), token)] = str(destination / f"fraction_{token}.json")
    return mapping


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        seeds = parse_seeds(args.seeds)
        fractions = parse_fractions(args.fractions)
        recipes = parse_recipe_names(args.recipes)
    except ValueError as exc:
        print(f"QAM-Seg: {exc}")
        return 2

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    jobs = iter_jobs(seeds, fractions, recipes)
    smoke = bool(args.smoke)
    execute = not bool(args.dry_run)

    if execute and not smoke:
        if args.steps is None or int(args.steps) < 1:
            print("QAM-Seg: a real protocol run needs --steps (a placeholder budget, not a tuned result).")
            print("Use --dry-run to list commands, or --smoke for a synthetic wiring check.")
            return 2
        if not args.train_npy or not args.mask_npy:
            print("QAM-Seg: pass --train-npy and --mask-npy for a real run.")
            print("This script does not download ISIC, HAM10000, or PH2.")
            print("Use --smoke to exercise the matrix on synthetic blobs.")
            return 2
        if bool(args.eval_npy) != bool(args.eval_mask_npy):
            print("QAM-Seg: pass both --eval-npy and --eval-mask-npy, or neither.")
            return 2
    if smoke and args.steps is None:
        args.steps = 1
    if smoke and args.image_size is None:
        args.image_size = 32
    if execute and not _cuda_student_ready(args.device):
        return 2

    split_for = {}
    if execute and not smoke:
        try:
            split_for = _split_map(args.train_npy, out_dir, seeds, fractions)
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"QAM-Seg: {exc}")
            print("This script does not download ISIC, HAM10000, or PH2.")
            return 2

    manifest_jobs = []
    for job in jobs:
        destination = _job_dir(out_dir, job.seed, job.recipe, job.fraction_token)
        split_json = ""
        labeled = ""
        mask = ""
        if execute and not smoke and job.fraction_token != "full":
            split_json = split_for[(job.seed, job.fraction_token)]
            labeled = args.train_npy
            mask = args.mask_npy
        elif execute and not smoke and job.fraction_token == "full":
            labeled = args.train_npy
            mask = args.mask_npy
        argv = job_argv(
            job,
            steps=args.steps,
            batch_size=int(args.batch_size),
            device=args.device,
            backbone=args.backbone,
            image_size=args.image_size if smoke or args.image_size else None,
            labeled_npy=labeled,
            mask_npy=mask,
            split_json=split_json,
            eval_image="" if smoke else args.eval_npy,
            eval_mask="" if smoke else args.eval_mask_npy,
            out_dir=str(destination),
            lambda_qw=float(args.lambda_qw),
            qw_ramp=args.qw_ramp,
            uncertainty_tau=float(args.uncertainty_tau),
            pl_threshold=float(args.pl_threshold),
        )
        manifest_jobs.append(
            {
                "recipe": job.recipe,
                "group": job.group,
                "method": job.method,
                "seed": job.seed,
                "fraction": job.fraction_token,
                "gate": job.gate,
                "baq": job.baq,
                "quality_reweight": job.quality_reweight,
                "out_dir": str(destination),
                "argv": argv,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seeds": list(seeds),
        "fractions": list(fractions),
        "recipes": list(recipes),
        "ablation": list(ABLATION_NAMES),
        "default_recipes": list(DEFAULT_RECIPE_NAMES),
        "default_seeds": list(DEFAULT_SEEDS),
        "default_fractions": list(DEFAULT_FRACTIONS),
        "smoke": smoke,
        "dry_run": bool(args.dry_run),
        "qar": "skipped",
        "quality_network": "none",
        "note": (
            "Same student and split for every recipe. "
            "No g_phi, SAM, CLIP, YOLO, or text module. "
            "This file is not a Dice or HD95 result. "
            "The script does not download a dataset."
        ),
        "jobs": manifest_jobs,
    }
    manifest_path = out_dir / "protocol_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Jobs: {len(manifest_jobs)}. Ablation cells: {', '.join(ABLATION_NAMES)}.")
    print("QAR is skipped. Quality reweight uses MQE, not a third network.")
    for item in manifest_jobs:
        print(" ".join(item["argv"]))

    if not execute:
        print("Dry run only. No training was started.")
        return 0

    failures = 0
    for item in manifest_jobs:
        command = [sys.executable, *item["argv"]]
        print("Running: " + " ".join(command))
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
        if completed.returncode != 0:
            failures += 1
            print(f"QAM-Seg: job failed ({completed.returncode}): {item['recipe']} seed {item['seed']} fraction {item['fraction']}")
    if failures:
        print(f"QAM-Seg: {failures} job(s) failed. See the logs above. No dataset was downloaded.")
        return 1
    print("Protocol jobs finished. Aggregate with scripts/aggregate_media_results.py.")
    print("Synthetic smoke metrics are not medical scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
