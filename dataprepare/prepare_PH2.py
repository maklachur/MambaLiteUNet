#!/usr/bin/env python3
"""Prepare the PH2 dermoscopic dataset (200 images) into .npy arrays (no scipy.misc required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataprepare.common import (  # noqa: E402
    IMAGE_EXTENSIONS,
    build_and_save_splits,
    collect_paired_files,
    default_out_dir,
    default_raw_dir,
)


def discover_ph2_pairs(data_dir: str | Path) -> list[tuple[Path, Path]]:
    """Discover PH2 (image, mask) pairs from nested IMDxxx folders or flat directories."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"PH2 directory not found: {root}. "
            "Place local PH2 dataset files under --data-dir. "
            "This script does not download datasets."
        )

    nested_candidates = [root, root / "PH2 Dataset images", root / "PH2_Dataset_images"]
    nested_pairs: list[tuple[Path, Path]] = []
    for base in nested_candidates:
        if not base.is_dir():
            continue
        for case_dir in sorted(base.iterdir()):
            if not case_dir.is_dir():
                continue
            stem = case_dir.name
            img_dir = case_dir / f"{stem}_Dermoscopic_Image"
            msk_dir = case_dir / f"{stem}_lesion"
            if img_dir.is_dir() and msk_dir.is_dir():
                imgs = [
                    p for p in sorted(img_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ]
                msks = [
                    p for p in sorted(msk_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ]
                if imgs and msks:
                    nested_pairs.append((imgs[0], msks[0]))
        if nested_pairs:
            return nested_pairs

    img_dirs = [
        root / sub
        for sub in ("images", "IMD_images", "Dermoscopic_Image", "input")
        if (root / sub).is_dir()
    ] or [root]
    msk_dirs = [
        root / sub
        for sub in ("masks", "lesion", "groundtruth", "GroundTruth")
        if (root / sub).is_dir()
    ] or [root]
    return collect_paired_files(img_dirs, msk_dirs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare PH2 dermoscopic images and masks into data_*.npy / mask_*.npy."
    )
    parser.add_argument("--data-dir", default=default_raw_dir("PH2"), help="Root PH2 dataset folder (default: Image_Datasets/PH2).")
    parser.add_argument("--out-dir", default=default_out_dir("PH2"), help="Output .npy folder (default: project data/PH2).")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Write all PH2 samples to data_test.npy and mask_test.npy for zero-shot cross-dataset evaluation.",
    )
    parser.add_argument(
        "--dtype",
        default="uint8",
        choices=("uint8", "float32", "float64"),
        help="Output array dtype (default: uint8 in [0, 255]).",
    )
    parser.add_argument(
        "--also-save-cwd",
        action="store_true",
        help="Also write data_*.npy / mask_*.npy into the current working directory.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pairs = discover_ph2_pairs(args.data_dir)
        t_cnt, v_cnt, te_cnt = args.train_count, args.val_count, args.test_count
        if (t_cnt is None and v_cnt is None and te_cnt is None) and len(pairs) == 200:
            t_cnt, v_cnt, te_cnt = 140, 20, 40
        summary = build_and_save_splits(
            pairs,
            args.out_dir,
            height=int(args.height),
            width=int(args.width),
            train_count=t_cnt,
            val_count=v_cnt,
            test_count=te_cnt,
            ratios=(0.7, 0.1, 0.2),
            eval_only=bool(args.eval_only),
            save_full_test_copy=not bool(args.eval_only),
            dtype=str(args.dtype),
            also_save_cwd=bool(args.also_save_cwd),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"prepare_PH2: {exc}")
        return 2

    print(f"Prepared PH2 ({len(pairs)} paired samples) -> {args.out_dir}")
    for name, shape in summary.items():
        print(f"  {name}.npy: {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
