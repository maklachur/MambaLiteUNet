"""Sampling math for 5%/10%/20% train splits. Synthetic ids only."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from qamseg.splits import (
    COUNT_RULE,
    DEFAULT_FRACTIONS,
    labeled_count,
    list_train_directory,
    load_fraction_record,
    make_labeled_fraction_split,
    read_id_list,
    rows_cover_train,
    train_ids_from_npy,
    write_split_files,
)


class CountTests(unittest.TestCase):
    def test_half_up_counts_for_the_protocol_fractions(self):
        expected = {
            1: (1, 1, 1),
            3: (1, 1, 1),
            10: (1, 1, 2),
            20: (1, 2, 4),
            100: (5, 10, 20),
            1250: (63, 125, 250),
            1815: (91, 182, 363),
        }
        for n, counts in expected.items():
            got = tuple(labeled_count(n, fraction) for fraction in DEFAULT_FRACTIONS)
            self.assertEqual(got, counts, n)

    def test_count_rejects_empty_and_out_of_range(self):
        with self.assertRaises(ValueError):
            labeled_count(0, 0.1)
        with self.assertRaises(ValueError):
            labeled_count(10, 0.0)
        with self.assertRaises(ValueError):
            labeled_count(10, 1.5)


class SamplingTests(unittest.TestCase):
    def test_nested_prefixes_and_complements_on_a_synthetic_list(self):
        ids = [f"id_{index:04d}" for index in range(200)]
        manifest = make_labeled_fraction_split(ids, seed=0)
        self.assertEqual(manifest["seed"], 0)
        self.assertIn("PCG64", manifest["rng"])
        self.assertEqual(manifest["count_rule"], COUNT_RULE)
        self.assertFalse(manifest["test_ids_included"])
        splits = manifest["splits"]
        labeled = {token: splits[token]["labeled_ids"] for token in ("5", "10", "20")}
        unlabeled = {token: splits[token]["unlabeled_ids"] for token in ("5", "10", "20")}
        self.assertEqual([len(labeled[token]) for token in ("5", "10", "20")], [10, 20, 40])
        self.assertEqual(labeled["10"][:10], labeled["5"])
        self.assertEqual(labeled["20"][:20], labeled["10"])
        # The larger labeled prefix is taken from the front of the smaller unlabeled pool.
        self.assertEqual(labeled["10"][10:], unlabeled["5"][:10])
        self.assertEqual(labeled["20"][20:], unlabeled["10"][:20])
        self.assertEqual(unlabeled["5"][10:], unlabeled["10"])
        self.assertEqual(unlabeled["10"][20:], unlabeled["20"])
        for token in ("5", "10", "20"):
            self.assertEqual(len(set(labeled[token]).intersection(unlabeled[token])), 0)
            self.assertEqual(set(labeled[token]).union(unlabeled[token]), set(ids))
            self.assertEqual(
                len(labeled[token]) + len(unlabeled[token]),
                200,
            )

    def test_seed_zero_permutation_is_stable(self):
        ids = [str(index) for index in range(20)]
        manifest = make_labeled_fraction_split(ids, seed=0)
        again = make_labeled_fraction_split(ids, seed=0)
        self.assertEqual(manifest["permutation_rows"], again["permutation_rows"])
        self.assertEqual(
            manifest["permutation_rows"],
            [10, 5, 8, 12, 1, 6, 15, 18, 13, 0, 17, 3, 2, 7, 4, 19, 9, 14, 16, 11],
        )
        self.assertEqual(manifest["splits"]["5"]["labeled_ids"], ["10"])
        self.assertEqual(manifest["splits"]["10"]["labeled_ids"], ["10", "5"])
        self.assertEqual(manifest["splits"]["20"]["labeled_ids"], ["10", "5", "8", "12"])
        other = make_labeled_fraction_split(ids, seed=1)
        self.assertNotEqual(other["splits"]["20"]["labeled_ids"], manifest["splits"]["20"]["labeled_ids"])

    def test_heldout_overlap_is_refused(self):
        from qamseg.splits import assert_train_excludes_heldout

        with self.assertRaises(ValueError) as ctx:
            assert_train_excludes_heldout(["a", "b", "c"], ["c", "d"])
        self.assertIn("test", str(ctx.exception).lower())
        assert_train_excludes_heldout(["a", "b"], ["c"])

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            make_labeled_fraction_split(["a", "a"], seed=0)


class FileTests(unittest.TestCase):
    def test_json_and_csv_roundtrip_records_the_seed(self):
        ids = [f"img_{index}" for index in range(40)]
        manifest = make_labeled_fraction_split(
            ids,
            seed=7,
            source={"kind": "ids-file", "path": "local-train-ids.txt"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            destination = write_split_files(manifest, tmp)
            written = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(written["seed"], 7)
            self.assertEqual(written["n_train"], 40)
            self.assertNotIn("permutation_rows", written)
            record = load_fraction_record(str(destination / "fraction_10.json"))
            self.assertEqual(record["seed"], 7)
            self.assertEqual(record["n_labeled"], labeled_count(40, 0.10))
            with open(destination / "labeled_10.csv", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["id"] for row in rows], record["labeled_ids"])
            self.assertEqual([int(row["row"]) for row in rows], record["labeled_rows"])
            self.assertTrue(all(row["role"] == "labeled" for row in rows))
            labeled_rows, unlabeled_rows = rows_cover_train(record, 40)
            self.assertEqual(len(labeled_rows) + len(unlabeled_rows), 40)

    def test_id_file_directory_and_npy_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            text_path = os.path.join(tmp, "ids.txt")
            with open(text_path, "w", encoding="utf-8") as handle:
                handle.write("# train only\n\nalpha\nbeta\n")
            self.assertEqual(read_id_list(text_path), ["alpha", "beta"])

            json_path = os.path.join(tmp, "ids.json")
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump({"train_ids": ["a", "b", "c"]}, handle)
            self.assertEqual(read_id_list(json_path), ["a", "b", "c"])

            image_dir = os.path.join(tmp, "images")
            os.makedirs(image_dir)
            for name in ("c.jpg", "a.jpg", "b.jpg"):
                open(os.path.join(image_dir, name), "wb").close()
            self.assertEqual(list_train_directory(image_dir, "*.jpg"), ["a", "b", "c"])

            npy_path = os.path.join(tmp, "data_train.npy")
            np.save(npy_path, np.zeros((6, 2, 2, 3), dtype=np.uint8))
            self.assertEqual(train_ids_from_npy(npy_path), ["0", "1", "2", "3", "4", "5"])

            missing = os.path.join(tmp, "nope.npy")
            with self.assertRaises(FileNotFoundError) as ctx:
                train_ids_from_npy(missing)
            self.assertIn("does not download", str(ctx.exception))

            with self.assertRaises(FileNotFoundError) as ctx:
                list_train_directory(os.path.join(tmp, "absent"), "*.jpg")
            self.assertIn("does not download", str(ctx.exception))

            test_path = os.path.join(tmp, "data_test.npy")
            np.save(test_path, np.zeros((2, 2, 2, 3), dtype=np.uint8))
            with self.assertRaises(ValueError) as ctx:
                train_ids_from_npy(test_path)
            self.assertIn("unlabeled", str(ctx.exception).lower())

    def test_cli_writes_splits_and_missing_npy_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            npy_path = os.path.join(tmp, "data_train.npy")
            out_dir = os.path.join(tmp, "splits")
            np.save(npy_path, np.zeros((20, 2, 2, 3), dtype=np.uint8))
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/make_labeled_fraction_splits.py",
                    "--train-npy",
                    npy_path,
                    "--name",
                    "synthetic",
                    "--seed",
                    "0",
                    "--out-dir",
                    out_dir,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            self.assertIn("seed: 0", completed.stdout)
            self.assertIn("does not download", completed.stdout)
            self.assertIn("not a Dice or HD95 result", completed.stdout)
            fraction_path = os.path.join(out_dir, "synthetic", "seed0", "fraction_20.json")
            with open(fraction_path, encoding="utf-8") as handle:
                fraction = json.load(handle)
            self.assertEqual(fraction["labeled_ids"], ["10", "5", "8", "12"])

            missing = subprocess.run(
                [
                    sys.executable,
                    "scripts/make_labeled_fraction_splits.py",
                    "--train-npy",
                    os.path.join(tmp, "missing-data_train.npy"),
                    "--name",
                    "synthetic",
                    "--out-dir",
                    out_dir,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(missing.returncode, 2)
            self.assertIn("does not download", missing.stdout)


if __name__ == "__main__":
    unittest.main()
