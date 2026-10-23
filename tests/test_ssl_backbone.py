"""MambaLiteUNet is the only SSL student. The real class is optional in CI."""

import contextlib
import io
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import unittest
from unittest import mock

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import torch
from torch import nn

from qamseg.backbones import BackboneUnavailable, build_mamba_student, read_c_list
from qamseg.cli import parse_train_args
from qamseg.train_ssl import run_from_args
from require_mamba_cuda import skip_unless_mamba_cuda


class _FakeMamba(nn.Module):
    built = 0

    def __init__(self, num_classes=1, input_channels=3, c_list=None):
        super().__init__()
        type(self).built += 1
        self.c_list = list(c_list or [])
        self.net = nn.Sequential(nn.Conv2d(input_channels, num_classes, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class _CpuDeviceLoader:
    """Stand-in so a reported CUDA device keeps batches on CPU in unit tests."""

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        yield from self.loader


@contextlib.contextmanager
def _cuda_reported_without_a_gpu():
    """Treat CUDA as available and keep the fake student on CPU."""

    def _stay_on_cpu(self, *args, **kwargs):
        return self

    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
        "torch.nn.Module.cuda", _stay_on_cpu
    ), mock.patch("qamseg.train_ssl._DeviceLoader", _CpuDeviceLoader):
        yield


class BackboneTests(unittest.TestCase):
    def test_default_parser_is_mamba_on_cuda(self):
        args, _unknown, reserved = parse_train_args(["--ssl"])
        self.assertEqual(args.ssl_backbone, "mamba")
        self.assertEqual(args.ssl_device, "cuda")
        self.assertEqual(args.ssl_split, "")
        self.assertEqual(reserved, [])

    def test_removed_backbone_choice_is_rejected(self):
        with self.assertRaises(SystemExit) as ctx, redirect_stderr(io.StringIO()):
            parse_train_args(["--ssl", "--ssl-backbone", "tiny"])
        self.assertEqual(ctx.exception.code, 2)

    def test_c_list_matches_the_supervised_config(self):
        self.assertEqual(read_c_list(), [16, 32, 48, 64, 96, 128])

    def test_cuda_unavailable_exits_before_import(self):
        args, _unknown, _reserved = parse_train_args(
            ["--ssl", "--ssl-steps", "1", "--ssl-image-size", "32", "--ssl-batch-size", "2"]
        )
        buffer = io.StringIO()
        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
            "qamseg.backbones.import_mamba_lite_unet"
        ) as importer, redirect_stdout(buffer):
            code = run_from_args(args)
        text = buffer.getvalue()
        self.assertEqual(code, 2)
        importer.assert_not_called()
        self.assertIn("CUDA is not available", text)
        self.assertIn("No other student is substituted", text)
        self.assertNotIn("TinySegStudent", text)
        self.assertNotIn("gate_keep_fraction:", text)

    def test_cpu_device_is_refused(self):
        args, _unknown, _reserved = parse_train_args(
            ["--ssl", "--ssl-device", "cpu", "--ssl-steps", "1", "--ssl-image-size", "32"]
        )
        buffer = io.StringIO()
        with mock.patch("qamseg.backbones.import_mamba_lite_unet") as importer, redirect_stdout(buffer):
            code = run_from_args(args)
        text = buffer.getvalue()
        self.assertEqual(code, 2)
        importer.assert_not_called()
        self.assertIn("--ssl-device cpu was requested", text)
        self.assertIn("No other student is substituted", text)
        self.assertNotIn("TinySegStudent", text)

    def test_synthetic_image_below_32_exits(self):
        args, _unknown, _reserved = parse_train_args(
            ["--ssl", "--ssl-steps", "1", "--ssl-image-size", "16"]
        )
        buffer = io.StringIO()
        with mock.patch("qamseg.backbones.import_mamba_lite_unet") as importer, redirect_stdout(buffer):
            code = run_from_args(args)
        self.assertEqual(code, 2)
        importer.assert_not_called()
        self.assertIn("at least 32", buffer.getvalue())

    def test_requested_mamba_uses_the_imported_class(self):
        _FakeMamba.built = 0
        args, _unknown, _reserved = parse_train_args(
            [
                "--ssl",
                "--ssl-backbone",
                "mamba",
                "--ssl-device",
                "cuda",
                "--ssl-steps",
                "1",
                "--ssl-image-size",
                "32",
                "--ssl-batch-size",
                "2",
            ]
        )
        buffer = io.StringIO()
        with _cuda_reported_without_a_gpu(), mock.patch(
            "qamseg.backbones.import_mamba_lite_unet", return_value=_FakeMamba
        ), redirect_stdout(buffer):
            code = run_from_args(args)
        text = buffer.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("Student: MambaLiteUNet", text)
        self.assertNotIn("TinySegStudent", text)
        self.assertIn("gate_keep_fraction:", text)
        self.assertGreaterEqual(_FakeMamba.built, 1)
        student = build_mamba_student(importer=lambda: _FakeMamba)
        self.assertIsInstance(student, _FakeMamba)
        self.assertEqual(student.c_list, [16, 32, 48, 64, 96, 128])
        self.assertEqual(tuple(student(torch.zeros(1, 3, 8, 8)).shape), (1, 1, 8, 8))

    def test_import_failure_exits_without_a_substitute(self):
        def explode():
            raise ImportError("No module named 'mamba_ssm'")

        args, _unknown, _reserved = parse_train_args(
            ["--ssl", "--ssl-backbone", "mamba", "--ssl-steps", "1", "--ssl-image-size", "32"]
        )
        buffer = io.StringIO()
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "qamseg.backbones.import_mamba_lite_unet", side_effect=explode
        ), redirect_stdout(buffer):
            code = run_from_args(args)
        text = buffer.getvalue()
        self.assertEqual(code, 2)
        self.assertIn("mamba_ssm", text)
        self.assertIn("No other student is substituted", text)
        self.assertNotIn("TinySegStudent", text)
        with self.assertRaises(BackboneUnavailable) as ctx:
            build_mamba_student(importer=explode)
        message = str(ctx.exception)
        self.assertIn("No other student is substituted", message)
        self.assertIn("mamba_ssm", message)
        self.assertNotIn("tiny", message.lower())

    def test_product_sources_do_not_offer_another_ssl_student(self):
        roots = [Path("qamseg"), Path("scripts"), Path("docs"), Path("README.md"), Path("train.py")]
        needles = ("TinySegStudent", "ConvLite", "--ssl-backbone tiny", "'tiny'", '"tiny"')
        hits = []
        for root in roots:
            paths = [root] if root.is_file() else list(root.rglob("*"))
            for path in paths:
                if not path.is_file() or path.suffix not in {".py", ".md", ""}:
                    continue
                if path.suffix == "" and path.name != "README.md":
                    continue
                text = path.read_text(encoding="utf-8")
                for needle in needles:
                    if needle in text:
                        hits.append(f"{path}: {needle}")
        self.assertEqual(hits, [])

    def test_train_py_ssl_exits_when_cuda_is_missing(self):
        if torch.cuda.is_available():
            self.skipTest("CUDA is available; the missing-CUDA path is covered in-process")
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
            timeout=120,
        )
        blob = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 2, blob)
        self.assertIn("CUDA is not available", blob)
        self.assertIn("No other student is substituted", blob)
        self.assertNotIn("TinySegStudent", blob)
        self.assertNotIn("gate_keep_fraction:", completed.stdout)

    @skip_unless_mamba_cuda
    def test_train_py_ssl_smoke_uses_mamba_on_cuda(self):
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
            timeout=180,
        )
        blob = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, blob)
        self.assertIn("Student: MambaLiteUNet", completed.stdout)
        self.assertNotIn("TinySegStudent", blob)
        self.assertIn("gate_keep_fraction:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
