"""Unit tests for HAM10000, PH2, ISIC2017, and ISIC2018 dataset preparation scripts."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from dataprepare.prepare_HAM10000 import main as ham_main
from dataprepare.prepare_ISIC2017 import main as isic17_main
from dataprepare.prepare_ISIC2018 import main as isic18_main
from dataprepare.prepare_PH2 import main as ph2_main
from qamseg.data import load_eval_arrays, load_labeled_arrays


def _write_rgb(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (40, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = color
    Image.fromarray(arr, mode="RGB").save(path)


def _write_mask(path: Path, box: tuple[int, int, int, int] = (5, 5, 20, 20), size: tuple[int, int] = (40, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    x0, y0, x1, y1 = box
    arr[y0:y1, x0:x1] = 255
    Image.fromarray(arr, mode="L").save(path)


class DataPrepareTests(unittest.TestCase):
    def test_ham10000_two_part_folders_and_deterministic_sorting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HAM10000"
            out = Path(tmp) / "out_ham"
            stems = [f"ISIC_{1000 + i:07d}" for i in (5, 1, 9, 3, 7, 2, 8, 4, 6, 0)]
            for idx, stem in enumerate(stems):
                part = "HAM10000_images_part_1" if idx < 5 else "HAM10000_images_part_2"
                stem_num = int(stem.split("_")[1]) - 1000
                _write_rgb(root / part / f"{stem}.png", (stem_num * 20, 50, 100))
                _write_mask(
                    root / "HAM10000_segmentations_lesion_tschandl" / f"{stem}_segmentation.png"
                )

            rc = ham_main(
                [
                    "--data-dir",
                    str(root),
                    "--out-dir",
                    str(out),
                    "--height",
                    "16",
                    "--width",
                    "16",
                ]
            )
            self.assertEqual(rc, 0)
            train_x = np.load(out / "data_train.npy")
            val_x = np.load(out / "data_val.npy")
            test_x = np.load(out / "data_test.npy")
            train_y = np.load(out / "mask_train.npy")
            self.assertEqual(train_x.shape, (7, 16, 16, 3))
            self.assertEqual(val_x.shape, (1, 16, 16, 3))
            self.assertEqual(test_x.shape, (2, 16, 16, 3))
            self.assertEqual(train_y.shape, (7, 16, 16))
            reds = [int(train_x[i, 0, 0, 0]) for i in range(7)]
            self.assertEqual(reds, [0, 20, 40, 60, 80, 100, 120])

            imgs_t, msks_t = load_labeled_arrays(
                str(out / "data_train.npy"), str(out / "mask_train.npy")
            )
            self.assertEqual(tuple(imgs_t.shape), (7, 3, 16, 16))
            self.assertEqual(tuple(msks_t.shape), (7, 1, 16, 16))

    def test_ph2_official_nested_bmp_layout_and_eval_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "PH2"
            out_split = Path(tmp) / "out_ph2_split"
            out_eval = Path(tmp) / "out_ph2_eval"
            for i in range(10):
                case = f"IMD{i + 1:03d}"
                base = root / "PH2 Dataset images" / case
                _write_rgb(base / f"{case}_Dermoscopic_Image" / f"{case}.bmp", (i * 15, 30, 60))
                _write_mask(base / f"{case}_lesion" / f"{case}_lesion.bmp")

            rc_split = ph2_main(
                [
                    "--data-dir",
                    str(root),
                    "--out-dir",
                    str(out_split),
                    "--height",
                    "16",
                    "--width",
                    "16",
                ]
            )
            self.assertEqual(rc_split, 0)
            self.assertEqual(np.load(out_split / "data_train.npy").shape, (7, 16, 16, 3))
            self.assertEqual(np.load(out_split / "data_val.npy").shape, (1, 16, 16, 3))
            self.assertEqual(np.load(out_split / "data_test.npy").shape, (2, 16, 16, 3))
            self.assertEqual(np.load(out_split / "data_test_full.npy").shape, (10, 16, 16, 3))

            rc_eval = ph2_main(
                [
                    "--data-dir",
                    str(root),
                    "--out-dir",
                    str(out_eval),
                    "--height",
                    "16",
                    "--width",
                    "16",
                    "--eval-only",
                ]
            )
            self.assertEqual(rc_eval, 0)
            eval_x, eval_y = load_eval_arrays(
                str(out_eval / "data_test.npy"), str(out_eval / "mask_test.npy")
            )
            self.assertEqual(tuple(eval_x.shape), (10, 3, 16, 16))
            self.assertEqual(tuple(eval_y.shape), (10, 1, 16, 16))

    def test_isic2017_and_isic2018_scripts_run_without_scipy_misc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root17 = Path(tmp) / "ISIC2017"
            out17 = Path(tmp) / "out17"
            for i in range(8):
                stem = f"ISIC_{i:07d}"
                _write_rgb(root17 / "ISIC2017_Task1-2_Training_Input" / f"{stem}.jpg", (100, 100, 100))
                _write_mask(
                    root17 / "ISIC2017_Task1_Training_GroundTruth" / f"{stem}_segmentation.png"
                )
            self.assertEqual(
                isic17_main(["--data-dir", str(root17), "--out-dir", str(out17), "--height", "16", "--width", "16"]),
                0,
            )
            self.assertTrue((out17 / "data_train.npy").is_file())

            root18 = Path(tmp) / "ISIC2018"
            out18 = Path(tmp) / "out18"
            for i in range(10):
                stem = f"ISIC_{i:07d}"
                _write_rgb(root18 / "ISIC2018_Task1-2_Training_Input" / f"{stem}.jpg", (120, 80, 60))
                _write_mask(
                    root18 / "ISIC2018_Task1_Training_GroundTruth" / f"{stem}_segmentation.png"
                )
            self.assertEqual(
                isic18_main(["--data-dir", str(root18), "--out-dir", str(out18), "--height", "16", "--width", "16"]),
                0,
            )
            self.assertEqual(np.load(out18 / "data_train.npy").shape, (7, 16, 16, 3))

    def test_missing_directory_exits_2_without_downloading(self):
        completed = subprocess.run(
            [
                sys.executable,
                "dataprepare/prepare_HAM10000.py",
                "--data-dir",
                "/tmp/nonexistent_ham10000_dir_xyz",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("does not download", completed.stdout)


if __name__ == "__main__":
    unittest.main()
