#!/usr/bin/env python3
"""Prepare ISIC 2017 dataset into .npy arrays using PIL (no legacy scipy.misc required).

Default split for 2000 training images: 1250 train, 150 val, 600 test.
Defaults: raw root ``IMAGE_DATASETS_ROOT/ISIC2017``, output ``<project>/data/ISIC2017``.
Override with ``--data-dir`` / ``--out-dir`` or env ``QAMSEG_IMAGE_DATASETS_ROOT``.
Optionally also write into cwd when ``--also-save-cwd`` is set.
"""

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


def discover_isic2017_pairs(data_dir: str | Path) -> list[tuple[Path, Path]]:
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"ISIC 2017 directory not found: {root}. "
            "Expected training images/masks under --data-dir "
            "(e.g. ISIC-2017_Training_Data + ISIC-2017_Training_Part1_GroundTruth, "
            "or ISIC2017_Task1-2_Training_Input + ISIC2017_Task1_Training_GroundTruth). "
            "This script does not download datasets."
        )
    img_dirs = [
        root / sub
        for sub in (
            "ISIC-2017_Training_Data",
            "ISIC2017_Task1-2_Training_Input",
            "images",
            "input",
        )
        if (root / sub).is_dir()
    ] or [root]
    msk_dirs = [
        root / sub
        for sub in (
            "ISIC-2017_Training_Part1_GroundTruth",
            "ISIC2017_Task1_Training_GroundTruth",
            "masks",
            "groundtruth",
        )
        if (root / sub).is_dir()
    ] or [root]
    # Official dumps ship *_superpixels.png beside *.jpg; exclude those stems.
    return collect_paired_files(
        img_dirs,
        msk_dirs,
        exclude_stem_substrings=("_superpixels",),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ISIC 2017 into .npy splits.")
    parser.add_argument("--data-dir", default=default_raw_dir("ISIC2017"))
    parser.add_argument("--out-dir", default=default_out_dir("ISIC2017"))
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument("--dtype", default="uint8", choices=("uint8", "float32", "float64"))
    parser.add_argument("--also-save-cwd", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pairs = discover_isic2017_pairs(args.data_dir)
        t_cnt, v_cnt, te_cnt = args.train_count, args.val_count, args.test_count
        if (t_cnt is None and v_cnt is None and te_cnt is None) and len(pairs) == 2000:
            t_cnt, v_cnt, te_cnt = 1250, 150, 600
        summary = build_and_save_splits(
            pairs,
            args.out_dir,
            height=int(args.height),
            width=int(args.width),
            train_count=t_cnt,
            val_count=v_cnt,
            test_count=te_cnt,
            ratios=(0.625, 0.075, 0.30),
            dtype=str(args.dtype),
            also_save_cwd=bool(args.also_save_cwd),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"prepare_ISIC2017: {exc}")
        return 2

    print(f"Prepared ISIC 2017 ({len(pairs)} paired samples) -> {args.out_dir}")
    for name, shape in summary.items():
        print(f"  {name}.npy: {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
