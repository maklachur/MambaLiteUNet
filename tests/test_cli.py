"""Flag behavior: --ssl is accepted, ablation flags still exit 2."""

import subprocess
import sys
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from qamseg.cli import parse_train_args, refuse_reserved_flags
from require_mamba_cuda import skip_unless_mamba_cuda


class ParseTests(unittest.TestCase):
    def test_no_flags_leave_ssl_off(self):
        args, unknown, reserved = parse_train_args([])
        self.assertFalse(args.ssl)
        self.assertEqual(unknown, [])
        self.assertEqual(reserved, [])

    def test_ssl_is_not_a_reserved_refusal(self):
        args, _unknown, reserved = parse_train_args(["--ssl", "--ssl-steps", "1"])
        self.assertTrue(args.ssl)
        self.assertEqual(args.ssl_steps, 1)
        self.assertEqual(reserved, [])

    def test_ablation_flags_still_exit_2(self):
        for flag in ("--no-amf", "--no-lgfm", "--no-cga"):
            _args, _unknown, reserved = parse_train_args([flag])
            with self.assertRaises(SystemExit) as ctx:
                refuse_reserved_flags(reserved)
            self.assertEqual(ctx.exception.code, 2)


class SubprocessTests(unittest.TestCase):
    @skip_unless_mamba_cuda
    def test_train_py_ssl_smoke_exits_0(self):
        completed = subprocess.run(
            [
                sys.executable,
                "train.py",
                "--ssl",
                "--ssl-steps",
                "1",
                "--ssl-image-size",
                "32",
                "--ssl-batch-size",
                "2",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("gate_keep_fraction:", completed.stdout)
        self.assertIn("unsup:", completed.stdout)
        self.assertIn("hd95:", completed.stdout)
        self.assertIn("Not experimentally validated", completed.stdout)

    @skip_unless_mamba_cuda
    def test_module_entry_exits_0(self):
        completed = subprocess.run(
            [sys.executable, "-m", "qamseg.train_ssl", "--ssl-steps", "1", "--ssl-image-size", "32"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("gate_keep_fraction:", completed.stdout)

    def test_ablation_still_refuses_before_training(self):
        completed = subprocess.run(
            [sys.executable, "train.py", "--no-amf"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("not wired", completed.stdout)

    def test_ssl_plus_ablation_still_refuses(self):
        completed = subprocess.run(
            [sys.executable, "train.py", "--ssl", "--no-cga"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertNotIn("gate_keep_fraction:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
