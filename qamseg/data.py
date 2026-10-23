"""Labeled + unlabeled batch scaffolding.

``loader.isic_loader`` is not modified and is not imported here. Supervised
``python train.py`` keeps using that class alone.

The smoke path uses synthetic blobs. A local train ``.npy`` pair can be
sliced with a fraction JSON from ``qamseg.splits`` (labeled rows keep masks,
unlabeled rows do not). A missing path raises ``FileNotFoundError``. This
module does not download ISIC, HAM10000, or PH2, and it does not treat
``data_val.npy`` or ``data_test.npy`` as an unlabeled pool.
"""

from __future__ import annotations

import itertools
import os
from typing import Iterator, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


UNLABELED_TODO = (
    "Point --ssl-split at a fraction JSON from scripts/make_labeled_fraction_splits.py "
    "and pass local train .npy files, or pass a local unlabeled .npy. "
    "Remaining train images only; do not use validation or test images. "
    "This code does not download ISIC, HAM10000, or PH2."
)

_HELD_OUT_ARRAY_NAMES = frozenset(
    {"data_test.npy", "data_val.npy", "mask_test.npy", "mask_val.npy"}
)


class SyntheticLesionDataset(Dataset):
    """Deterministic blobs on noise.

    Not a medical dataset and not a stand-in for ISIC, HAM, or PH2.
    Labeled items are ``(image, mask)`` with shapes ``(3, H, W)`` and
    ``(1, H, W)``. Unlabeled items are the image only.
    """

    def __init__(self, length: int = 4, size: int = 32, labeled: bool = True, seed: int = 0):
        if length < 1:
            raise ValueError(f"length must be positive, got {length}")
        if size < 8:
            raise ValueError(f"size must be >= 8, got {size}")
        self.length = int(length)
        self.size = int(size)
        self.labeled = bool(labeled)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if index < 0 or index >= self.length:
            raise IndexError(index)
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(index))
        image = torch.rand(3, self.size, self.size, generator=generator) * 0.3
        low = self.size // 4
        high = max(low + 1, (3 * self.size) // 4)
        cy = int(torch.randint(low, high, (1,), generator=generator))
        cx = int(torch.randint(low, high, (1,), generator=generator))
        radius_low = max(2, self.size // 8)
        radius_high = max(radius_low + 1, self.size // 3)
        radius = int(torch.randint(radius_low, radius_high, (1,), generator=generator))
        mask2d = _circle_mask(self.size, cy, cx, radius)
        image = image.clone()
        image[:, mask2d > 0.5] = 0.85
        if self.labeled:
            return image, mask2d.unsqueeze(0)
        return image


def _circle_mask(size: int, cy: int, cx: int, radius: int) -> Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        indexing="ij",
    )
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2).float()


class ArrayImageDataset(Dataset):
    """In-memory images, optional masks. Images are ``(N, C, H, W)`` float.

    When ``augment=True``, applies random 90-degree rotations and flips to
    ``(image, mask)`` pairs (or ``image`` alone) using a seeded generator.
    """

    def __init__(
        self,
        images: Tensor,
        masks: Optional[Tensor] = None,
        *,
        augment: bool = False,
        seed: int = 0,
    ):
        if images.ndim != 4:
            raise ValueError(f"images must be NCHW, got shape {tuple(images.shape)}")
        self.images = images.float()
        self.masks = None if masks is None else masks.float()
        if self.masks is not None and self.masks.shape[0] != self.images.shape[0]:
            raise ValueError(
                f"mask count {self.masks.shape[0]} != image count {self.images.shape[0]}"
            )
        self.augment = bool(augment)
        self.seed = int(seed)
        self._call_count = 0

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int):
        image = self.images[index]
        mask = None if self.masks is None else self.masks[index]
        if self.augment:
            gen = torch.Generator()
            gen.manual_seed(self.seed * 100003 + int(index) * 97 + self._call_count)
            self._call_count += 1
            k = int(torch.randint(0, 4, (1,), generator=gen).item())
            if k > 0:
                image = torch.rot90(image, k, dims=(-2, -1))
                if mask is not None:
                    mask = torch.rot90(mask, k, dims=(-2, -1))
            if bool(torch.randint(0, 2, (1,), generator=gen).item()):
                image = torch.flip(image, dims=(-1,))
                if mask is not None:
                    mask = torch.flip(mask, dims=(-1,))
            if bool(torch.randint(0, 2, (1,), generator=gen).item()):
                image = torch.flip(image, dims=(-2,))
                if mask is not None:
                    mask = torch.flip(mask, dims=(-2,))
        if mask is None:
            return image
        return image, mask


def _to_nchw(array: np.ndarray) -> Tensor:
    if array.ndim != 4:
        raise ValueError(f"Expected a 4D array, got shape {array.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    # Channel-last if the last axis looks like RGB and the second does not.
    if array.shape[-1] in (1, 3) and array.shape[1] not in (1, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    elif array.shape[1] not in (1, 3) and array.shape[-1] in (1, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    tensor = tensor.float()
    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def _refuse_heldout_array(path: str) -> None:
    name = os.path.basename(path).lower() if path else ""
    if name in _HELD_OUT_ARRAY_NAMES:
        raise ValueError(
            f"Refusing to build an SSL pool from {path}. "
            "Validation and test arrays must not be used as labeled or unlabeled data. "
            "Point --ssl-labeled-npy at data_train.npy and --ssl-mask-npy at mask_train.npy. "
            + UNLABELED_TODO
        )


def load_unlabeled_array(path: str) -> Tensor:
    """Load a local unlabeled image array.

    Accepts ``(N, H, W, C)`` or ``(N, C, H, W)``. Missing files raise
    ``FileNotFoundError`` and do not trigger a download.
    """
    _refuse_heldout_array(path)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Unlabeled array not found: {path}. {UNLABELED_TODO}")
    array = np.load(path)
    images = _to_nchw(array)
    if images.shape[1] not in (1, 3):
        raise ValueError(f"Unlabeled array channel axis is unclear: shape {tuple(images.shape)}")
    return images


def _read_image_mask_files(image_path: str, mask_path: str) -> tuple[Tensor, Tensor]:
    if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
        missing = image_path if not os.path.isfile(image_path) else mask_path
        raise FileNotFoundError(f"Array not found: {missing}. {UNLABELED_TODO}")
    images = _to_nchw(np.load(image_path))
    masks = torch.from_numpy(np.ascontiguousarray(np.load(mask_path))).float()
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    elif masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks.permute(0, 3, 1, 2)
    if masks.max() > 1.5:
        masks = masks / 255.0
    masks = (masks > 0).float()
    if masks.shape[0] != images.shape[0]:
        raise ValueError(
            f"Count mismatch: {images.shape[0]} images vs {masks.shape[0]} masks"
        )
    if masks.shape[-2:] != images.shape[-2:]:
        raise ValueError(
            f"Spatial mismatch: images {tuple(images.shape)} masks {tuple(masks.shape)}"
        )
    return images, masks


def load_labeled_arrays(image_path: str, mask_path: str) -> tuple[Tensor, Tensor]:
    """Load a local labeled pair. Same message as the unlabeled path: no download."""
    _refuse_heldout_array(image_path)
    _refuse_heldout_array(mask_path)
    return _read_image_mask_files(image_path, mask_path)


def load_eval_arrays(image_path: str, mask_path: str) -> tuple[Tensor, Tensor]:
    """Load a held-out eval pair. Never an unlabeled training pool.

    ``data_val.npy`` and ``data_test.npy`` are allowed here and still refused
    by ``load_labeled_arrays`` / ``load_unlabeled_array``. A missing path
    raises ``FileNotFoundError`` and does not download anything.
    """
    if not image_path or not mask_path:
        raise ValueError("Pass both --ssl-eval-npy and --ssl-eval-mask-npy, or neither.")
    return _read_image_mask_files(image_path, mask_path)


def datasets_from_split(
    image_path: str,
    mask_path: str,
    split_path: str,
    *,
    augment_labeled: bool = False,
    seed: int = 0,
) -> tuple[Dataset, Dataset]:
    """Slice a local train pair with a ``fraction_*.json`` file.

    Unlabeled rows are returned as images only. The split must be a partition
    of this train array. Missing files raise ``FileNotFoundError``.
    """
    from qamseg.splits import load_fraction_record, rows_cover_train

    _refuse_heldout_array(image_path)
    _refuse_heldout_array(mask_path)
    record = load_fraction_record(split_path)
    images, masks = load_labeled_arrays(image_path, mask_path)
    labeled_rows, unlabeled_rows = rows_cover_train(record, int(images.shape[0]))
    if not labeled_rows:
        raise ValueError(f"Split {split_path} has no labeled rows.")
    if not unlabeled_rows:
        raise ValueError(
            f"Split {split_path} has an empty unlabeled pool. "
            "The complement of the labeled subset is empty, so there is nothing to train on. "
            + UNLABELED_TODO
        )
    labeled = ArrayImageDataset(
        images[labeled_rows],
        masks[labeled_rows],
        augment=augment_labeled,
        seed=seed,
    )
    unlabeled = ArrayImageDataset(images[unlabeled_rows], masks=None)
    return labeled, unlabeled


def build_ssl_datasets(
    length: int = 4,
    size: int = 32,
    seed: int = 0,
    unlabeled_npy: str = "",
    labeled_npy: str = "",
    mask_npy: str = "",
    split_json: str = "",
    augment_labeled: bool = False,
) -> tuple[Dataset, Dataset]:
    """Synthetic datasets by default.

    A non-empty ``split_json`` slices ``labeled_npy`` / ``mask_npy`` into a
    labeled subset and an unlabeled complement. A non-empty ``unlabeled_npy``
    replaces the unlabeled half when no split is passed. Labeled ``.npy``
    paths are used only when both the image array and the mask array are
    provided. ``isic_loader`` is never constructed here.
    """
    if split_json:
        if unlabeled_npy:
            raise ValueError(
                "--ssl-split selects the unlabeled pool from the train arrays "
                "(masks withheld). Do not also pass --ssl-unlabeled-npy."
            )
        if not labeled_npy or not mask_npy:
            raise ValueError(
                "Pass --ssl-labeled-npy and --ssl-mask-npy with --ssl-split. "
                "Missing paths are an error. " + UNLABELED_TODO
            )
        return datasets_from_split(
            labeled_npy,
            mask_npy,
            split_json,
            augment_labeled=augment_labeled,
            seed=seed,
        )
    if labeled_npy or mask_npy:
        if not labeled_npy or not mask_npy:
            raise ValueError("Pass both --ssl-labeled-npy and --ssl-mask-npy, or neither.")
        images, masks = load_labeled_arrays(labeled_npy, mask_npy)
        labeled: Dataset = ArrayImageDataset(
            images, masks, augment=augment_labeled, seed=seed
        )
    else:
        labeled = SyntheticLesionDataset(length=length, size=size, labeled=True, seed=seed)

    if unlabeled_npy:
        unlabeled: Dataset = ArrayImageDataset(load_unlabeled_array(unlabeled_npy))
    else:
        unlabeled = SyntheticLesionDataset(
            length=length,
            size=size,
            labeled=False,
            seed=seed + 1000,
        )
    return labeled, unlabeled


def build_dual_loaders(
    labeled: Dataset,
    unlabeled: Dataset,
    batch_size: int = 2,
    shuffle: bool = False,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Two loaders. ``num_workers=0`` so the smoke path needs no fork."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    gen_l = torch.Generator()
    gen_l.manual_seed(int(seed))
    gen_u = torch.Generator()
    gen_u.manual_seed(int(seed) + 997)
    labeled_loader = DataLoader(
        labeled,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True,
        generator=gen_l if shuffle else None,
    )
    unlabeled_loader = DataLoader(
        unlabeled,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True,
        generator=gen_u if shuffle else None,
    )
    return labeled_loader, unlabeled_loader


def iter_dual_batches(
    labeled_loader: Sequence,
    unlabeled_loader: Sequence,
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """Yield ``(x_l, y_l, x_u)`` across the full span of the larger loader.

    Fixes the unlabeled starvation bug where iterating only over
    ``labeled_loader`` truncated each pass to ``len(labeled_loader)`` steps
    and discarded the remaining 80%–95% of ``unlabeled_loader``. Now runs
    ``max(len(labeled_loader), len(unlabeled_loader))`` steps per epoch,
    cycling whichever loader is shorter.
    """
    if len(unlabeled_loader) == 0:
        raise ValueError(f"Unlabeled loader is empty. {UNLABELED_TODO}")
    if len(labeled_loader) == 0:
        raise ValueError("Labeled loader is empty.")
    total_batches = max(len(labeled_loader), len(unlabeled_loader))
    labeled_cycle = itertools.cycle(labeled_loader)
    unlabeled_cycle = itertools.cycle(unlabeled_loader)
    for _ in range(total_batches):
        labeled_batch = next(labeled_cycle)
        unlabeled_batch = next(unlabeled_cycle)
        images_l, masks_l = labeled_batch
        images_u = (
            unlabeled_batch[0]
            if isinstance(unlabeled_batch, (tuple, list))
            else unlabeled_batch
        )
        count = min(int(images_l.shape[0]), int(images_u.shape[0]))
        yield images_l[:count], masks_l[:count], images_u[:count]
