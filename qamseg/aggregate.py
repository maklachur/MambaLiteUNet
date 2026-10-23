"""Mean ± sample-std tables from ``metrics.json`` trees.

Dice, mIoU, HD95, and ASSD are aggregated per recipe, fraction, backbone,
and data tag. One seed leaves the std blank. Non-finite distances are
omitted from that column only. This does not download data and does not
treat a synthetic smoke file as a medical result.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


METRICS_FORMAT = "qamseg-media-metrics-v1"
SUMMARY_COLUMNS = (
    "recipe",
    "group",
    "fraction",
    "backbone",
    "data",
    "eval_split",
    "n",
    "dice_mean",
    "dice_std",
    "miou_mean",
    "miou_std",
    "hd95_mean",
    "hd95_std",
    "assd_mean",
    "assd_std",
    "dice",
    "miou",
    "hd95",
    "assd",
)


def json_float(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.startswith("n/a"):
            return None
        try:
            value = float(text.split()[0])
        except ValueError:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def metric_value(record: Mapping, name: str) -> Optional[float]:
    metrics = record.get("metrics") or {}
    if name == "dice":
        if "dice" in metrics:
            return json_float(metrics.get("dice"))
        return json_float(metrics.get("f1_or_dsc"))
    if name == "miou":
        return json_float(metrics.get("miou"))
    if name == "hd95":
        if "hd95_mean" in metrics:
            return json_float(metrics.get("hd95_mean"))
        return json_float(metrics.get("hd95"))
    if name == "assd":
        if "assd_mean" in metrics:
            return json_float(metrics.get("assd_mean"))
        return json_float(metrics.get("assd"))
    raise KeyError(name)


def mean_std(values: Iterable[Optional[float]]) -> tuple[Optional[float], Optional[float], int]:
    """Sample standard deviation (``ddof=1``). A single finite value has std ``None``."""
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None, None, 0
    arr = np.asarray(finite, dtype=np.float64)
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, None, int(arr.size)
    return mean, float(arr.std(ddof=1)), int(arr.size)


def format_mean_std(mean: Optional[float], std: Optional[float], digits: int = 4) -> str:
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _cell(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def group_key(record: Mapping) -> tuple:
    return (
        str(record.get("recipe") or record.get("method") or ""),
        str(record.get("group") or ""),
        str(record.get("fraction_token") or record.get("fraction") or ""),
        str(record.get("backbone") or ""),
        str(record.get("data") or ""),
        str(record.get("eval_split") or ""),
    )


def aggregate_records(records: Sequence[Mapping], digits: int = 4) -> list[dict]:
    buckets: dict[tuple, list[Mapping]] = {}
    order: list[tuple] = []
    for record in records:
        key = group_key(record)
        if key not in buckets:
            order.append(key)
            buckets[key] = []
        buckets[key].append(record)

    rows = []
    for key in order:
        group = buckets[key]
        recipe, group_name, fraction, backbone, data, eval_split = key
        dice_mean, dice_std, n_dice = mean_std(metric_value(item, "dice") for item in group)
        miou_mean, miou_std, n_miou = mean_std(metric_value(item, "miou") for item in group)
        hd_mean, hd_std, n_hd = mean_std(metric_value(item, "hd95") for item in group)
        assd_mean, assd_std, n_assd = mean_std(metric_value(item, "assd") for item in group)
        n = max(n_dice, n_miou, n_hd, n_assd, len(group))
        rows.append(
            {
                "recipe": recipe,
                "group": group_name,
                "fraction": fraction,
                "backbone": backbone,
                "data": data,
                "eval_split": eval_split,
                "n": n,
                "n_dice": n_dice,
                "n_miou": n_miou,
                "n_hd95": n_hd,
                "n_assd": n_assd,
                "dice_mean": dice_mean,
                "dice_std": dice_std,
                "miou_mean": miou_mean,
                "miou_std": miou_std,
                "hd95_mean": hd_mean,
                "hd95_std": hd_std,
                "assd_mean": assd_mean,
                "assd_std": assd_std,
                "dice": format_mean_std(dice_mean, dice_std, digits=digits),
                "miou": format_mean_std(miou_mean, miou_std, digits=digits),
                "hd95": format_mean_std(hd_mean, hd_std, digits=digits),
                "assd": format_mean_std(assd_mean, assd_std, digits=digits),
            }
        )
    return rows


def load_metrics_tree(root: Path | str) -> list[dict]:
    """Read every ``metrics.json`` under ``root``. Unknown JSON is skipped."""
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"Results directory not found: {base}")
    records = []
    for path in sorted(base.rglob("metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("format") not in (None, METRICS_FORMAT):
            continue
        if "metrics" not in payload:
            continue
        payload = dict(payload)
        payload["_path"] = str(path)
        records.append(payload)
    return records


def write_metrics_file(path: Path | str, payload: Mapping) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["format"] = METRICS_FORMAT
    destination.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def render_csv(rows: Sequence[Mapping]) -> str:
    lines = []
    # csv module wants a file; build in memory for tests.
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(SUMMARY_COLUMNS), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        exported = {column: row.get(column, "") for column in SUMMARY_COLUMNS}
        for key in (
            "dice_mean",
            "dice_std",
            "miou_mean",
            "miou_std",
            "hd95_mean",
            "hd95_std",
            "assd_mean",
            "assd_std",
        ):
            exported[key] = _cell(row.get(key))
        exported["n"] = row.get("n", "")
        writer.writerow(exported)
    lines.append(buffer.getvalue())
    return "".join(lines)


def render_markdown(rows: Sequence[Mapping]) -> str:
    header = (
        "| recipe | fraction | backbone | data | n | Dice | mIoU | HD95 | ASSD |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    body = []
    for row in rows:
        body.append(
            "| {recipe} | {fraction} | {backbone} | {data} | {n} | {dice} | {miou} | {hd95} | {assd} |\n".format(
                recipe=row.get("recipe", ""),
                fraction=row.get("fraction", ""),
                backbone=row.get("backbone", ""),
                data=row.get("data", ""),
                n=row.get("n", ""),
                dice=row.get("dice", ""),
                miou=row.get("miou", ""),
                hd95=row.get("hd95", ""),
                assd=row.get("assd", ""),
            )
        )
    note = (
        "\nMean ± sample std (ddof=1) over the seeds in each cell. "
        "A single seed has no std. Blank distance cells had no finite values. "
        "Synthetic rows are wiring checks, not MedIA numbers. "
        "HD95 and ASSD are pixels at mask resolution, not millimetres.\n"
    )
    return header + "".join(body) + note
