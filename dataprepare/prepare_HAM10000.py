#!/usr/bin/env python3
"""Prepare the HAM10000 skin lesion dataset into .npy arrays (no scipy.misc required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataprepare.common import (  # noqa: E402
    build_and_save_splits,
    collect_paired_files,
    default_out_dir,
    default_raw_dir,
)


def discover_ham10000_pairs(
    data_dir: str | Path,
    image_subdirs: tuple[str, ...] = (
        "HAM10000_images_part_1",
        "HAM10000_images_part_2",
        "HAM10000_images",
        "images",
        "ISIC2018_Task3_Training_Input",
    ),
    mask_subdirs: tuple[str, ...] = (
        "HAM10000_segmentations_lesion_tschandl/HAM10000_segmentations_lesion_tschandl",
        "HAM10000_segmentations_lesion_tschandl",
        "HAM10000_segmentations",
        "masks",
        "groundtruth",
    ),
) -> list[tuple[Path, Path]]:
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"HAM10000 directory not found: {root}. "
            "Place local HAM10000 images and segmentation masks under --data-dir. "
            "This script does not download datasets."
        )
    img_dirs = [root / sub for sub in image_subdirs if (root / sub).is_dir()]
    if not img_dirs:
        img_dirs = [root]
    msk_dirs = [root / sub for sub in mask_subdirs if (root / sub).is_dir()]
    if not msk_dirs:
        msk_dirs = [root]
    return collect_paired_files(img_dirs, msk_dirs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare HAM10000 images and masks into data_*.npy / mask_*.npy."
    )
    parser.add_argument("--data-dir", default=default_raw_dir("HAM10000"), help="Root HAM10000 folder (default: Image_Datasets/HAM10000).")
    parser.add_argument("--out-dir", default=default_out_dir("HAM10000"), help="Output .npy folder (default: project data/HAM10000).")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Write all paired samples to data_test.npy and mask_test.npy for cross-dataset evaluation.",
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
        pairs = discover_ham10000_pairs(args.data_dir)
        t_cnt, v_cnt, te_cnt = args.train_count, args.val_count, args.test_count
        if (t_cnt is None and v_cnt is None and te_cnt is None) and len(pairs) == 10015:
            t_cnt, v_cnt, te_cnt = 7010, 1001, 2004
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
            dtype=str(args.dtype),
            also_save_cwd=bool(args.also_save_cwd),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"prepare_HAM10000: {exc}")
        return 2

    print(f"Prepared HAM10000 ({len(pairs)} paired samples) -> {args.out_dir}")
    for name, shape in summary.items():
        print(f"  {name}.npy: {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
