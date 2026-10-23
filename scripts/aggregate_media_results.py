#!/usr/bin/env python3
"""Aggregate QAM-Seg ``metrics.json`` files into a mean ± std table.

    python scripts/aggregate_media_results.py results/media_protocol

Writes ``summary.csv`` and ``summary.md`` under ``--out-dir`` (default:
the results directory). Dice, mIoU, HD95, and ASSD use the sample standard
deviation across seeds. This does not train and does not download data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qamseg.aggregate import aggregate_records, load_metrics_tree, render_csv, render_markdown  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Mean ± std Dice / mIoU / HD95 / ASSD from metrics.json.")
    parser.add_argument("root", help="Directory that contains one metrics.json per run.")
    parser.add_argument("--out-dir", default="", help="Defaults to the results directory.")
    parser.add_argument("--digits", type=int, default=4)
    args = parser.parse_args(argv)
    root = Path(args.root)
    try:
        records = load_metrics_tree(root)
    except FileNotFoundError as exc:
        print(f"QAM-Seg: {exc}")
        return 2
    if not records:
        print(f"QAM-Seg: no metrics.json files under {root}.")
        print("Run scripts/run_media_protocol.py first. This script does not download data.")
        return 2
    rows = aggregate_records(records, digits=int(args.digits))
    out_dir = Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"
    csv_path.write_text(render_csv(rows), encoding="utf-8")
    markdown = render_markdown(rows)
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print("Rows mix only within the same recipe, fraction, backbone, data tag, and eval split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
