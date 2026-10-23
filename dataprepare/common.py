"""Shared PIL/NumPy dataset preparation helpers (no legacy scipy.misc required).

Replaces ``scipy.misc.imread`` and ``scipy.misc.imresize`` (removed in SciPy 1.3)
with ``PIL.Image`` using deterministic lexicographical stem sorting, bilinear
interpolation for RGB images, and nearest-neighbor interpolation for binary
segmentation masks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image


# Central raw-image store on this lab machine. Override with env var for other hosts.
IMAGE_DATASETS_ROOT = Path(
    os.environ.get(
        "QAMSEG_IMAGE_DATASETS_ROOT",
        "/home/icdmlab/Downloads/Image_Datasets",
    )
).expanduser()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_raw_dir(dataset_name: str) -> str:
    """Absolute path under IMAGE_DATASETS_ROOT for a named dataset folder."""
    return str(IMAGE_DATASETS_ROOT / dataset_name)


def default_out_dir(dataset_name: str) -> str:
    """Absolute path under the project ``data/<dataset>/`` for processed .npy files."""
    return str(PROJECT_ROOT / "data" / dataset_name)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_SUFFIXES = ("_segmentation", "_lesion", "_mask", "")


def load_rgb_image(path: str | Path, height: int = 256, width: int = 256) -> np.ndarray:
    """Read an image as RGB and resize to ``(height, width, 3)`` using bilinear interpolation."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if rgb.size != (width, height):
            resample = getattr(Image, "Resampling", Image).BILINEAR
            rgb = rgb.resize((width, height), resample=resample)
        return np.asarray(rgb, dtype=np.uint8)


def load_binary_mask(path: str | Path, height: int = 256, width: int = 256) -> np.ndarray:
    """Read a segmentation mask as grayscale and resize to ``(height, width)`` with nearest-neighbor.

    Returns values in ``{0, 255}`` (uint8) so both ``loader.py`` (``mask > 0``)
    and ``qamseg.data`` (``masks / 255.0 > 0``) interpret foreground identically.
    """
    with Image.open(path) as img:
        gray = img.convert("L")
        if gray.size != (width, height):
            resample = getattr(Image, "Resampling", Image).NEAREST
            gray = gray.resize((width, height), resample=resample)
        arr = np.asarray(gray, dtype=np.uint8)
        return np.where(arr >= 128, 255, 0).astype(np.uint8)


def find_matching_mask(
    stem: str,
    mask_dirs: Sequence[Path],
    extensions: Sequence[str] = (".png", ".bmp", ".jpg", ".jpeg", ".tif"),
    suffixes: Sequence[str] = MASK_SUFFIXES,
) -> Optional[Path]:
    """Locate the ground-truth mask file corresponding to image ``stem``."""
    for mask_dir in mask_dirs:
        if not mask_dir.is_dir():
            continue
        for suffix in suffixes:
            for ext in extensions:
                candidate = mask_dir / f"{stem}{suffix}{ext}"
                if candidate.is_file():
                    return candidate
                candidate_upper = mask_dir / f"{stem}{suffix}{ext.upper()}"
                if candidate_upper.is_file():
                    return candidate_upper
    return None


def collect_paired_files(
    image_dirs: Sequence[str | Path],
    mask_dirs: Sequence[str | Path],
    *,
    extensions: Sequence[str] = IMAGE_EXTENSIONS,
    exclude_stem_substrings: Sequence[str] = (),
) -> list[tuple[Path, Path]]:
    """Discover and deterministically sort ``(image_path, mask_path)`` pairs by image stem."""
    img_paths: dict[str, Path] = {}
    for folder in image_dirs:
        p = Path(folder)
        if not p.is_dir():
            continue
        for entry in sorted(p.iterdir()):
            if entry.is_file() and entry.suffix.lower() in extensions:
                if any(sub in entry.stem for sub in exclude_stem_substrings):
                    continue
                img_paths.setdefault(entry.stem, entry)

    mask_dir_paths = [Path(d) for d in mask_dirs]
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for stem in sorted(img_paths):
        img_path = img_paths[stem]
        mask_path = find_matching_mask(stem, mask_dir_paths)
        if mask_path is None:
            missing.append(stem)
        else:
            pairs.append((img_path, mask_path))

    if missing and not pairs:
        raise FileNotFoundError(
            f"Found {len(img_paths)} images but could not match any masks in {[str(d) for d in mask_dir_paths]}. "
            f"First missing stem: {missing[0]}"
        )
    if missing:
        raise FileNotFoundError(
            f"Missing ground-truth masks for {len(missing)} image(s) (first: {missing[0]}) "
            f"in {[str(d) for d in mask_dir_paths]}."
        )
    return pairs


def compute_split_counts(
    total: int,
    train_count: Optional[int] = None,
    val_count: Optional[int] = None,
    test_count: Optional[int] = None,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> tuple[int, int, int]:
    """Compute ``(n_train, n_val, n_test)`` summing to ``total``."""
    if total < 1:
        raise ValueError(f"total must be >= 1, got {total}")
    if train_count is not None and val_count is not None and test_count is not None:
        if train_count + val_count + test_count > total:
            raise ValueError(
                f"Requested split counts ({train_count} + {val_count} + {test_count} = "
                f"{train_count + val_count + test_count}) exceed dataset size {total}."
            )
        return int(train_count), int(val_count), int(test_count)

    r_train, r_val, r_test = ratios
    r_sum = r_train + r_val + r_test
    n_train = int(round(total * (r_train / r_sum)))
    n_val = int(round(total * (r_val / r_sum)))
    if n_train + n_val > total:
        n_val = max(0, total - n_train)
    n_test = total - n_train - n_val
    return n_train, n_val, n_test


def build_and_save_splits(
    pairs: Sequence[tuple[Path, Path]],
    out_dir: str | Path,
    *,
    height: int = 256,
    width: int = 256,
    train_count: Optional[int] = None,
    val_count: Optional[int] = None,
    test_count: Optional[int] = None,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    eval_only: bool = False,
    save_full_test_copy: bool = False,
    dtype: str = "uint8",
    also_save_cwd: bool = False,
) -> dict[str, tuple[int, ...]]:
    """Load paired images/masks, slice into train/val/test, and save ``.npy`` files."""
    if not pairs:
        raise ValueError("No (image, mask) pairs provided.")
    n_total = len(pairs)
    np_dtype = np.uint8 if dtype == "uint8" else (np.float32 if dtype == "float32" else np.float64)

    images = np.zeros((n_total, height, width, 3), dtype=np_dtype)
    masks = np.zeros((n_total, height, width), dtype=np_dtype)

    for idx, (img_path, mask_path) in enumerate(pairs):
        images[idx] = load_rgb_image(img_path, height=height, width=width).astype(np_dtype)
        masks[idx] = load_binary_mask(mask_path, height=height, width=width).astype(np_dtype)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    destinations = [out_path]
    if also_save_cwd and out_path.resolve() != Path.cwd().resolve():
        destinations.append(Path.cwd())

    if eval_only:
        for dest in destinations:
            np.save(dest / "data_test.npy", images)
            np.save(dest / "mask_test.npy", masks)
        return {
            "data_test": images.shape,
            "mask_test": masks.shape,
        }

    n_train, n_val, n_test = compute_split_counts(
        n_total,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        ratios=ratios,
    )
    train_img = images[:n_train]
    val_img = images[n_train : n_train + n_val]
    test_img = images[n_train + n_val : n_train + n_val + n_test]

    train_mask = masks[:n_train]
    val_mask = masks[n_train : n_train + n_val]
    test_mask = masks[n_train + n_val : n_train + n_val + n_test]

    for dest in destinations:
        np.save(dest / "data_train.npy", train_img)
        np.save(dest / "data_val.npy", val_img)
        np.save(dest / "data_test.npy", test_img)
        np.save(dest / "mask_train.npy", train_mask)
        np.save(dest / "mask_val.npy", val_mask)
        np.save(dest / "mask_test.npy", test_mask)
        if save_full_test_copy:
            np.save(dest / "data_test_full.npy", images)
            np.save(dest / "mask_test_full.npy", masks)

    summary = {
        "data_train": train_img.shape,
        "data_val": val_img.shape,
        "data_test": test_img.shape,
        "mask_train": train_mask.shape,
        "mask_val": val_mask.shape,
        "mask_test": test_mask.shape,
    }
    if save_full_test_copy:
        summary["data_test_full"] = images.shape
        summary["mask_test_full"] = masks.shape
    return summary
