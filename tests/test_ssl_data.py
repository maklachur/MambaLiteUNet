"""Dual-batch scaffolding. Synthetic tensors only; isic_loader is not used."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import numpy as np

from qamseg.data import (
    UNLABELED_TODO,
    build_dual_loaders,
    build_ssl_datasets,
    iter_dual_batches,
    load_unlabeled_array,
)
from require_mamba_cuda import skip_unless_mamba_cuda


class DualLoaderTests(unittest.TestCase):
    def test_synthetic_batches_have_labeled_and_unlabeled_tensors(self):
        labeled, unlabeled = build_ssl_datasets(length=4, size=16, seed=0)
        labeled_loader, unlabeled_loader = build_dual_loaders(
            labeled, unlabeled, batch_size=2, shuffle=False
        )
        batches = list(iter_dual_batches(labeled_loader, unlabeled_loader))
        self.assertEqual(len(batches), 2)
        images_l, masks_l, images_u = batches[0]
        self.assertEqual(tuple(images_l.shape), (2, 3, 16, 16))
        self.assertEqual(tuple(masks_l.shape), (2, 1, 16, 16))
        self.assertEqual(tuple(images_u.shape), (2, 3, 16, 16))
        self.assertGreaterEqual(float(masks_l.min()), 0.0)
        self.assertLessEqual(float(masks_l.max()), 1.0)
        self.assertTrue(set(masks_l.unique().tolist()).issubset({0.0, 1.0}))

    def test_missing_unlabeled_npy_names_the_todo(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_unlabeled_array("/tmp/qamseg-missing-unlabeled.npy")
        message = str(ctx.exception)
        self.assertIn("does not download", message)
        self.assertIn("ISIC", message)
        self.assertIn("--ssl-split", UNLABELED_TODO)

    def test_local_npy_roundtrip_without_a_dataset_download(self):
        array = np.zeros((2, 8, 8, 3), dtype=np.float32)
        array[0, 2:5, 2:5, :] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "unlabeled.npy")
            np.save(path, array)
            images = load_unlabeled_array(path)
        self.assertEqual(tuple(images.shape), (2, 3, 8, 8))
        self.assertAlmostEqual(float(images[0, :, 3, 3].mean()), 1.0)
        self.assertAlmostEqual(float(images[0, :, 0, 0].mean()), 0.0)

    def test_module_does_not_expose_isic_loader(self):
        import qamseg.data as data

        self.assertFalse(hasattr(data, "isic_loader"))

    def test_split_json_slices_train_rows_and_drops_unlabeled_masks(self):
        from qamseg.data import datasets_from_split
        from qamseg.splits import make_labeled_fraction_split, write_split_files

        images = np.zeros((20, 8, 8, 3), dtype=np.float32)
        masks = np.zeros((20, 8, 8), dtype=np.float32)
        for index in range(20):
            images[index, 0, 0, 0] = (index + 1) / 100.0
            masks[index, 1:3, 1:3] = 1.0
        ids = [str(index) for index in range(20)]
        manifest = make_labeled_fraction_split(ids, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            destination = write_split_files(manifest, tmp)
            image_path = os.path.join(tmp, "data_train.npy")
            mask_path = os.path.join(tmp, "mask_train.npy")
            np.save(image_path, images)
            np.save(mask_path, masks)
            labeled, unlabeled = datasets_from_split(
                image_path,
                mask_path,
                str(destination / "fraction_20.json"),
            )
            self.assertEqual(len(labeled), 4)
            self.assertEqual(len(unlabeled), 16)
            image_l, mask_l = labeled[0]
            self.assertEqual(tuple(image_l.shape), (3, 8, 8))
            self.assertEqual(tuple(mask_l.shape), (1, 8, 8))
            record = json.loads((destination / "fraction_20.json").read_text(encoding="utf-8"))
            first_row = int(record["labeled_rows"][0])
            self.assertAlmostEqual(float(image_l[0, 0, 0]), (first_row + 1) / 100.0)
            self.assertEqual(float(mask_l.max()), 1.0)
            image_u = unlabeled[0]
            self.assertIsInstance(image_u, type(image_l))
            self.assertNotIsInstance(unlabeled[0], tuple)
            missing = os.path.join(tmp, "missing-fraction.json")
            with self.assertRaises(FileNotFoundError) as ctx:
                datasets_from_split(image_path, mask_path, missing)
            self.assertIn("does not download", str(ctx.exception))

    def test_val_and_test_array_names_are_refused(self):
        from qamseg.data import load_labeled_arrays

        with tempfile.TemporaryDirectory() as tmp:
            test_images = os.path.join(tmp, "data_test.npy")
            test_masks = os.path.join(tmp, "mask_test.npy")
            np.save(test_images, np.zeros((2, 4, 4, 3), dtype=np.float32))
            np.save(test_masks, np.zeros((2, 4, 4), dtype=np.float32))
            with self.assertRaises(ValueError) as ctx:
                load_labeled_arrays(test_images, test_masks)
            self.assertIn("test", str(ctx.exception).lower())

    @skip_unless_mamba_cuda
    def test_train_py_split_flag_uses_local_rows_only(self):
        from qamseg.splits import make_labeled_fraction_split, write_split_files

        images = np.zeros((20, 32, 32, 3), dtype=np.float32)
        masks = np.zeros((20, 32, 32), dtype=np.float32)
        manifest = make_labeled_fraction_split([str(index) for index in range(20)], seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            destination = write_split_files(manifest, os.path.join(tmp, "splits"))
            image_path = os.path.join(tmp, "data_train.npy")
            mask_path = os.path.join(tmp, "mask_train.npy")
            np.save(image_path, images)
            np.save(mask_path, masks)
            completed = subprocess.run(
                [
                    sys.executable,
                    "train.py",
                    "--ssl",
                    "--ssl-split",
                    str(destination / "fraction_20.json"),
                    "--ssl-labeled-npy",
                    image_path,
                    "--ssl-mask-npy",
                    mask_path,
                    "--ssl-steps",
                    "1",
                    "--ssl-batch-size",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        self.assertIn("Student: MambaLiteUNet", completed.stdout)
        self.assertNotIn("TinySegStudent", completed.stdout)
        self.assertIn("not a test score", completed.stdout)
        self.assertIn("gate_keep_fraction:", completed.stdout)
        self.assertIn("Not experimentally validated", completed.stdout)


if __name__ == "__main__":
    unittest.main()
