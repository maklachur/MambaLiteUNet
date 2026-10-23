"""Skip MambaLiteUNet smokes when CUDA or mamba_ssm is missing."""

from __future__ import annotations

import importlib
import unittest


def mamba_cuda_skip_reason() -> str | None:
    try:
        import torch
    except Exception as exc:
        return f"PyTorch is not importable ({type(exc).__name__}: {exc})"
    if not torch.cuda.is_available():
        return "CUDA is not available; MambaLiteUNet selective scan is CUDA-only"
    for name in ("timm", "mamba_ssm"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            return (
                f"{name} is not importable ({type(exc).__name__}: {exc}); "
                "MambaLiteUNet smokes are not run"
            )
    return None


def skip_unless_mamba_cuda(test_item):
    reason = mamba_cuda_skip_reason()
    return unittest.skipIf(reason is not None, reason or "MambaLiteUNet requires CUDA")(test_item)
