#!/usr/bin/env python3
"""Prepare a custom paired image/mask dataset into .npy splits using PIL."""

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
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare custom images/ and masks/ into .npy splits.")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--mask-dir", default="masks")
    parser.add_argument("--out-dir", default=default_out_dir("custom"))
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dtype", default="uint8", choices=("uint8", "float32", "float64"))
    parser.add_argument("--also-save-cwd", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pairs = collect_paired_files([args.image_dir], [args.mask_dir])
        summary = build_and_save_splits(
            pairs,
            args.out_dir,
            height=int(args.height),
            width=int(args.width),
            train_count=args.train_count,
            val_count=args.val_count,
            test_count=args.test_count,
            ratios=(0.7, 0.1, 0.2),
            eval_only=bool(args.eval_only),
            dtype=str(args.dtype),
            also_save_cwd=bool(args.also_save_cwd),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"prepare_custom_dataset: {exc}")
        return 2

    print(f"Prepared custom dataset ({len(pairs)} paired samples) -> {args.out_dir}")
    for name, shape in summary.items():
        print(f"  {name}.npy: {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
