"""Deterministic 5% / 10% / 20% labeled subsets of a train id list.

The unlabeled pool for a fraction is the complement of that labeled subset
inside the same train list. Validation and test ids are never sampled.
This module does not read images and does not download ISIC, HAM10000, or PH2.

Sampling rule (``nested_prefix``):

1. Require a non-empty list of unique train ids. Row ``i`` is the position
   of that id in the list you pass in (for ``data_train.npy``, row ``i`` is
   ``"i"``).
2. Draw one permutation with Fisher-Yates over
   ``numpy.random.PCG64(seed).random_raw`` (uint64, unbiased rejection).
   This does not call ``Generator.permutation``, whose stream has changed
   between NumPy releases.
3. For each fraction, ``n_labeled = min(n, max(1, floor(fraction * n + 0.5)))``.
   The labeled ids are the prefix of that permutation. A smaller fraction is
   therefore a prefix of a larger one when the same seed is used.
4. The unlabeled ids are the rest of the permutation. Masks for those ids
   stay out of the unsupervised loss; the loader drops them.

``dataprepare/prepare_ISIC2017.py`` and ``prepare_ISIC2018.py`` glob an input
folder and then slice train/val/test by position. That glob is not sorted, and
the folder still contains the images that become ``data_val.npy`` and
``data_test.npy``. A directory listing of that folder is not the official
train split. Use ``--train-npy`` on ``data_train.npy``, or an id file written
in train-row order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


DEFAULT_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.20)
COUNT_RULE = "n_labeled = min(n, max(1, floor(fraction * n + 0.5)))"
RNG_NAME = (
    "Fisher-Yates; each index is numpy.random.PCG64(seed).random_raw() "
    "reduced with unbiased uint64 rejection sampling"
)
_U64 = 1 << 64
NO_DOWNLOAD = "This tool does not download ISIC, HAM10000, or PH2."
FORMAT_NAME = "qamseg-labeled-fraction-v1"

# Basenames produced by dataprepare/*.py that are not the train image array.
HELD_OUT_NPY_NAMES = frozenset(
    {
        "data_test.npy",
        "data_val.npy",
        "mask_test.npy",
        "mask_val.npy",
        "mask_train.npy",
    }
)


def seeded_permutation(n: int, seed: int) -> list[int]:
    """Permutation of ``range(n)`` from PCG64's raw 64-bit stream.

    Fisher-Yates walks from the end of the list. Each swap index is
    ``random_raw() % (i + 1)`` after rejecting values in the incomplete
    top bucket of ``2**64``, so the draw is unbiased and does not depend
    on ``Generator.permutation``.
    """
    if n < 1:
        raise ValueError(f"permutation length must be positive, got {n}")
    order = list(range(n))
    if n == 1:
        return order
    bitgen = np.random.PCG64(int(seed))
    for i in range(n - 1, 0, -1):
        limit = i + 1
        cutoff = _U64 - (_U64 % limit)
        while True:
            raw = int(bitgen.random_raw())
            if raw < cutoff:
                j = raw % limit
                break
        order[i], order[j] = order[j], order[i]
    return order


def labeled_count(n: int, fraction: float) -> int:
    """How many train ids are labeled at ``fraction``.

    Half-integers round away from zero via ``floor(fraction * n + 0.5)``,
    then the count is clamped to ``[1, n]`` for a non-empty train list.
    ``fraction`` must be in ``(0, 1]``.
    """
    if n < 1:
        raise ValueError(f"train list length must be positive, got {n}")
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    raw = int(math.floor(float(fraction) * int(n) + 0.5))
    return min(int(n), max(1, raw))


def percent_token(fraction: float) -> str:
    """``0.05`` -> ``"5"``. Non-integers use a ``p`` in place of the dot."""
    percent = float(fraction) * 100.0
    nearest = round(percent)
    if abs(percent - nearest) < 1e-6:
        return str(int(nearest))
    text = f"{percent:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _as_id_list(ids: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    seen: dict[str, int] = {}
    for raw in ids:
        item = str(raw).strip()
        if not item:
            raise ValueError("Train ids contain an empty entry.")
        if item in seen:
            raise ValueError(
                f"Duplicate train id {item!r} at positions {seen[item]} and {len(cleaned)}."
            )
        seen[item] = len(cleaned)
        cleaned.append(item)
    if not cleaned:
        raise ValueError("Train id list is empty. " + NO_DOWNLOAD)
    return cleaned


def assert_train_excludes_heldout(train_ids: Sequence[str], heldout_ids: Sequence[str]) -> None:
    """Refuse a train list that already contains validation or test ids."""
    overlap = sorted(set(train_ids).intersection(heldout_ids))
    if not overlap:
        return
    preview = ", ".join(overlap[:8])
    raise ValueError(
        f"Refusing to build a split: {len(overlap)} id(s) are in both the train list "
        f"and the held-out list ({preview}). "
        "Validation and test images must not enter the labeled subset or the unlabeled pool. "
        + NO_DOWNLOAD
    )


def make_labeled_fraction_split(
    train_ids: Sequence[str],
    *,
    seed: int = 0,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    source: Optional[Mapping] = None,
    heldout_checked: bool = False,
    n_heldout: int = 0,
) -> dict:
    """Return the manifest and one record per fraction. Does not write files.

    ``labeled_rows`` are positions in ``train_ids``, which is the index a
    loader should use on ``data_train.npy`` when that array was built in the
    same order.
    """
    if int(seed) < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")
    ids = _as_id_list(train_ids)
    fracs = tuple(float(item) for item in fractions)
    if not fracs:
        raise ValueError("At least one labeled fraction is required.")
    if len(set(fracs)) != len(fracs):
        raise ValueError(f"Duplicate fractions: {fracs}")
    ordered = tuple(sorted(fracs))
    n = len(ids)
    order = seeded_permutation(n, int(seed))
    permuted = [ids[index] for index in order]

    by_token: dict[str, dict] = {}
    for fraction in ordered:
        count = labeled_count(n, fraction)
        labeled_rows = order[:count]
        unlabeled_rows = order[count:]
        labeled_ids = permuted[:count]
        unlabeled_ids = permuted[count:]
        token = percent_token(fraction)
        if token in by_token:
            raise ValueError(f"Fractions collide on file token {token!r}: {ordered}")
        by_token[token] = {
            "format": FORMAT_NAME,
            "seed": int(seed),
            "rng": RNG_NAME,
            "numpy_version": np.__version__,
            "rule": "nested_prefix",
            "count_rule": COUNT_RULE,
            "fraction": float(fraction),
            "percent_token": token,
            "n_train": n,
            "n_labeled": count,
            "n_unlabeled": n - count,
            "labeled_ids": labeled_ids,
            "unlabeled_ids": unlabeled_ids,
            "labeled_rows": labeled_rows,
            "unlabeled_rows": unlabeled_rows,
            "test_ids_included": False,
            "masks_for_unlabeled": "withheld",
            "note": (
                "Labeled ids are a prefix of one seeded permutation of the train list. "
                "Unlabeled ids are the complement inside that train list. "
                "Do not append validation or test images. "
                "Do not use this file as a Dice or HD95 result."
            ),
        }

    tokens = [percent_token(fraction) for fraction in ordered]
    for earlier, later in zip(tokens, tokens[1:]):
        early_ids = by_token[earlier]["labeled_ids"]
        late_ids = by_token[later]["labeled_ids"]
        if late_ids[: len(early_ids)] != early_ids:
            raise RuntimeError("Nested prefix invariant failed.")

    return {
        "format": FORMAT_NAME,
        "seed": int(seed),
        "rng": RNG_NAME,
        "numpy_version": np.__version__,
        "rule": "nested_prefix",
        "count_rule": COUNT_RULE,
        "fractions": [float(fraction) for fraction in ordered],
        "n_train": n,
        "source": dict(source or {}),
        "heldout_checked": bool(heldout_checked),
        "n_heldout": int(n_heldout),
        "test_ids_included": False,
        "permutation_rows": order,
        "note": (
            "Train ids only. The unlabeled pool is the complement of the labeled "
            "subset. Validation and test images are not in these files. "
            + NO_DOWNLOAD
        ),
        "splits": {token: by_token[token] for token in tokens},
    }


def _write_role_csv(path: Path, ids: Sequence[str], rows: Sequence[int], role: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "row", "role"])
        for image_id, row in zip(ids, rows):
            writer.writerow([image_id, int(row), role])


def write_split_files(manifest: Mapping, out_dir: Path | str) -> Path:
    """Write ``manifest.json``, per-fraction JSON, and CSV id lists."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    splits = manifest["splits"]
    file_index = {}
    for token, record in splits.items():
        fraction_name = f"fraction_{token}.json"
        labeled_name = f"labeled_{token}.csv"
        unlabeled_name = f"unlabeled_{token}.csv"
        fraction_path = destination / fraction_name
        fraction_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        _write_role_csv(destination / labeled_name, record["labeled_ids"], record["labeled_rows"], "labeled")
        _write_role_csv(
            destination / unlabeled_name,
            record["unlabeled_ids"],
            record["unlabeled_rows"],
            "unlabeled",
        )
        file_index[token] = {
            "fraction": record["fraction"],
            "n_labeled": record["n_labeled"],
            "n_unlabeled": record["n_unlabeled"],
            "fraction_json": fraction_name,
            "labeled_csv": labeled_name,
            "unlabeled_csv": unlabeled_name,
        }

    membership_name = "membership.csv"
    tokens = list(splits.keys())
    with (destination / membership_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "row"] + [f"labeled_{token}" for token in tokens])
        id_to_row = {}
        for record in splits.values():
            for image_id, row in zip(record["labeled_ids"], record["labeled_rows"]):
                id_to_row[image_id] = int(row)
            for image_id, row in zip(record["unlabeled_ids"], record["unlabeled_rows"]):
                id_to_row.setdefault(image_id, int(row))
        labeled_sets = {token: set(splits[token]["labeled_ids"]) for token in tokens}
        # Original train order is the row order, not the permutation order.
        ordered_ids = sorted(id_to_row, key=lambda image_id: id_to_row[image_id])
        for image_id in ordered_ids:
            flags = ["1" if image_id in labeled_sets[token] else "0" for token in tokens]
            writer.writerow([image_id, id_to_row[image_id], *flags])

    public = {
        key: manifest[key]
        for key in (
            "format",
            "seed",
            "rng",
            "numpy_version",
            "rule",
            "count_rule",
            "fractions",
            "n_train",
            "source",
            "heldout_checked",
            "n_heldout",
            "test_ids_included",
            "note",
        )
    }
    public["files"] = file_index
    public["membership_csv"] = membership_name
    (destination / "manifest.json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    return destination


def load_fraction_record(path: str) -> dict:
    """Load one ``fraction_*.json``. A missing path is an error, not a download."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"Split file not found: {path}. "
            "Generate it with scripts/make_labeled_fraction_splits.py from a local train id list. "
            + NO_DOWNLOAD
        )
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict) or "labeled_rows" not in record or "unlabeled_rows" not in record:
        raise ValueError(
            f"{path} has no labeled_rows/unlabeled_rows. "
            "Pass fraction_5.json, fraction_10.json, or fraction_20.json, not manifest.json."
        )
    return record


def rows_cover_train(record: Mapping, n_rows: int) -> tuple[list[int], list[int]]:
    """Validate that the fraction partition matches a train array of length ``n_rows``."""
    if int(record.get("n_train", -1)) != int(n_rows):
        raise ValueError(
            f"Split n_train={record.get('n_train')} but the array has {n_rows} rows. "
            "Refusing to slice. The split and the train array are a pair; "
            + NO_DOWNLOAD
        )
    labeled = [int(row) for row in record["labeled_rows"]]
    unlabeled = [int(row) for row in record["unlabeled_rows"]]
    if len(set(labeled)) != len(labeled) or len(set(unlabeled)) != len(unlabeled):
        raise ValueError("Split rows contain duplicates.")
    if set(labeled).intersection(unlabeled):
        raise ValueError("Labeled and unlabeled rows overlap.")
    combined = set(labeled).union(unlabeled)
    expected = set(range(int(n_rows)))
    if combined != expected:
        missing = len(expected - combined)
        extra = len(combined - expected)
        raise ValueError(
            f"Split rows are not a partition of the train array (missing={missing}, extra={extra})."
        )
    for row in labeled + unlabeled:
        if row < 0 or row >= int(n_rows):
            raise ValueError(f"Split row {row} is outside [0, {n_rows}).")
    return labeled, unlabeled


def read_id_list(path: str) -> list[str]:
    """Read ids from a text file, a JSON list, or a one-column CSV.

    Text files: one id per line. Blank lines and ``#`` comments are skipped.
    JSON: a list of strings, or an object with ``ids`` or ``train_ids``.
    CSV: a column named ``id``, ``image_id``, ``filename``, ``name``, or ``stem``;
    otherwise the first column, including the first row.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"Id list not found: {path}. " + NO_DOWNLOAD + " Write a local train-id file or pass --train-npy."
        )
    text = Path(path).read_text(encoding="utf-8")
    suffix = Path(path).suffix.lower()
    if suffix == ".json" or text.lstrip().startswith(("[", "{")):
        payload = json.loads(text)
        if isinstance(payload, list):
            raw = payload
        elif isinstance(payload, dict):
            for key in ("ids", "train_ids", "labeled_ids"):
                if key in payload:
                    raw = payload[key]
                    break
            else:
                raise ValueError(f"{path} JSON object has no ids or train_ids list.")
        else:
            raise ValueError(f"{path} JSON must be a list or an object with an id list.")
        return _as_id_list(raw)

    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            raise ValueError(f"{path} has no rows.")
        header = [cell.strip().lower() for cell in rows[0]]
        known = {"id", "image_id", "filename", "name", "stem"}
        if any(cell in known for cell in header):
            column = next(index for index, cell in enumerate(header) if cell in known)
            body = rows[1:]
        else:
            column = 0
            body = rows
        values = []
        for row in body:
            if not row or column >= len(row) or not row[column].strip():
                continue
            values.append(row[column].strip())
        return _as_id_list(values)

    values = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values.append(stripped.split()[0])
    return _as_id_list(values)


def list_train_directory(directory: str, pattern: str = "*.jpg") -> list[str]:
    """Sorted file stems under ``directory``.

    The order is lexicographic on the relative path. It is not the unsorted
    ``glob.glob`` order used by ``dataprepare/prepare_ISIC2017.py`` and
    ``prepare_ISIC2018.py``, and it does not apply those scripts' train/val/test
    slices. Point this at a folder that already contains only training images.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Train directory not found: {directory}. "
            + NO_DOWNLOAD
            + " Pass a local folder of training images, or use --train-npy / --ids-file."
        )
    matches = sorted(
        (path for path in root.glob(pattern) if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not matches:
        raise FileNotFoundError(
            f"No files matched {pattern!r} under {directory}. " + NO_DOWNLOAD
        )
    stems: list[str] = []
    seen: dict[str, str] = {}
    for path in matches:
        stem = path.stem
        if stem in seen:
            raise ValueError(
                f"Duplicate stem {stem!r} in {directory}: {seen[stem]} and {path.name}."
            )
        seen[stem] = path.name
        stems.append(stem)
    return _as_id_list(stems)


def train_ids_from_npy(path: str) -> list[str]:
    """Row ids ``\"0\" .. \"N-1\"`` for a local train array. Pixels are not copied."""
    name = os.path.basename(path).lower() if path else ""
    if name in HELD_OUT_NPY_NAMES:
        raise ValueError(
            f"Refusing to index {path}. "
            "Pass data_train.npy (the train slice), not a val/test array and not mask_train.npy. "
            "Test images must not enter the unlabeled pool."
        )
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"Train array not found: {path}. "
            + NO_DOWNLOAD
            + " Expected a local data_train.npy whose leading axis is the train set."
        )
    try:
        array = np.load(path, mmap_mode="r")
    except Exception as exc:
        raise ValueError(f"Could not read train array {path}: {type(exc).__name__}: {exc}. " + NO_DOWNLOAD) from exc
    try:
        if getattr(array, "ndim", 0) < 1:
            raise ValueError(f"Train array {path} has no leading axis.")
        n = int(array.shape[0])
    finally:
        mapped = getattr(array, "_mmap", None)
        if mapped is not None:
            mapped.close()
        del array
    if n < 1:
        raise ValueError(f"Train array {path} has 0 rows.")
    return [str(index) for index in range(n)]


def build_and_write(
    train_ids: Sequence[str],
    out_dir: Path | str,
    *,
    seed: int,
    fractions: Sequence[float],
    source: Mapping,
    heldout_ids: Optional[Sequence[str]] = None,
) -> Path:
    heldout = list(heldout_ids or [])
    if heldout:
        assert_train_excludes_heldout(train_ids, heldout)
    manifest = make_labeled_fraction_split(
        train_ids,
        seed=seed,
        fractions=fractions,
        source=source,
        heldout_checked=bool(heldout),
        n_heldout=len(heldout),
    )
    return write_split_files(manifest, out_dir)


def _print_summary(destination: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Wrote labeled-fraction splits to {destination}")
    print(f"seed: {manifest['seed']}")
    print(f"rng: {manifest['rng']}")
    print(f"count_rule: {manifest['count_rule']}")
    print(f"n_train: {manifest['n_train']}")
    for token, info in manifest["files"].items():
        print(
            f"{token}%: labeled {info['n_labeled']}, unlabeled {info['n_unlabeled']} "
            f"({info['fraction_json']})"
        )
    print("Train ids only. Validation and test images are not in these files.")
    print(NO_DOWNLOAD)
    print("These files are indices, not a Dice or HD95 result.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/make_labeled_fraction_splits.py",
        description=(
            "Write deterministic 5%/10%/20% labeled subsets and complementary "
            "unlabeled pools for one train id list. Does not download data."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--ids-file",
        help="Local text, JSON, or CSV list of train ids in data_train.npy row order.",
    )
    source.add_argument(
        "--train-dir",
        help=(
            "Directory of training images only. Stems are sorted lexicographically. "
            "This is not the unsorted glob in dataprepare/prepare_ISIC*.py."
        ),
    )
    source.add_argument(
        "--train-npy",
        help="Local train image array. Ids are row indices 0..N-1. Do not pass data_test.npy.",
    )
    parser.add_argument("--pattern", default="*.jpg", help="Glob used with --train-dir. Default: *.jpg")
    parser.add_argument(
        "--heldout-ids",
        default="",
        help="Optional val/test id file. Overlap with the train list is an error. Ids are not copied into the split.",
    )
    parser.add_argument("--name", default="train", help="Subdirectory name under --out-dir.")
    parser.add_argument("--seed", type=int, default=0, help="PCG64 seed recorded in every JSON file.")
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_FRACTIONS),
        help="Labeled fractions of the train list. Default: 0.05 0.10 0.20",
    )
    parser.add_argument("--out-dir", default="splits", help="Root directory for JSON and CSV files.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.ids_file:
            train_ids = read_id_list(args.ids_file)
            source = {"kind": "ids-file", "path": os.path.abspath(args.ids_file)}
        elif args.train_dir:
            train_ids = list_train_directory(args.train_dir, pattern=args.pattern)
            source = {
                "kind": "train-dir",
                "path": os.path.abspath(args.train_dir),
                "pattern": args.pattern,
                "order": "lexicographic relative path",
                "npy_alignment": (
                    "Not the unsorted glob order in dataprepare/prepare_ISIC2017.py "
                    "or prepare_ISIC2018.py. Do not use those input folders here; "
                    "they still contain the images later written to data_val.npy and data_test.npy."
                ),
            }
        else:
            train_ids = train_ids_from_npy(args.train_npy)
            source = {"kind": "train-npy", "path": os.path.abspath(args.train_npy), "ids": "row index"}
        heldout = read_id_list(args.heldout_ids) if args.heldout_ids else []
        destination = Path(args.out_dir) / str(args.name) / f"seed{int(args.seed)}"
        build_and_write(
            train_ids,
            destination,
            seed=int(args.seed),
            fractions=args.fractions,
            source=source,
            heldout_ids=heldout,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"QAM-Seg splits: {exc}")
        return 2
    _print_summary(destination, destination / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
