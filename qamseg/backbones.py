"""SSL student factory.

The SSL student is ``models.MambaLiteUNet``. A failed import or construction
is fatal. Nothing is substituted. The supervised ``python train.py`` path
does not use this module. MambaLiteUNet is the backbone, not a second
quality model.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_C_LIST = [16, 32, 48, 64, 96, 128]


class BackboneUnavailable(Exception):
    """MambaLiteUNet did not import or construct. Nothing is substituted."""

    def __init__(self, cause: BaseException):
        self.cause = cause
        super().__init__(self.format_message())

    def format_message(self) -> str:
        return (
            "QAM-Seg: MambaLiteUNet could not be imported or constructed.\n"
            f"Reason: {type(self.cause).__name__}: {self.cause}\n"
            "The SSL student is MambaLiteUNet. No other student is substituted.\n"
            "This process does not install packages and does not download datasets.\n"
            "A successful import is still not a medical training run and does not report CVPR Dice or HD95."
        )


def read_c_list(config_path: Path | None = None) -> list[int]:
    """Read ``c_list`` without importing ``configs/config_setting.py``.

    That module builds a loss and a timestamped work directory on import.
    """
    path = config_path or (ROOT / "configs" / "config_setting.py")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return list(DEFAULT_C_LIST)
    match = re.search(r"""['"]c_list['"]\s*:\s*\[([0-9,\s]+)\]""", text)
    if not match:
        return list(DEFAULT_C_LIST)
    values = [int(part) for part in match.group(1).split(",") if part.strip()]
    return values or list(DEFAULT_C_LIST)


def import_mamba_lite_unet():
    """Import the released class. Missing ``mamba_ssm`` or ``timm`` raises."""
    from models.MambaLiteUNet import MambaLiteUNet

    return MambaLiteUNet


def build_mamba_student(
    importer=None,
    *,
    use_amf: bool = True,
    use_lgfm: bool = True,
    use_cga: bool = True,
    scan_mode: str = "four_way",
    global_mode: str = "attn",
):
    """Construct MambaLiteUNet with the config ``c_list`` and optional ablation flags.

    Inspects the constructor signature so simple mock classes in unit tests
    that only accept ``(num_classes, input_channels, c_list)`` continue to work.
    """
    load = importer or import_mamba_lite_unet
    try:
        cls = load()
        kwargs = {
            "num_classes": 1,
            "input_channels": 3,
            "c_list": read_c_list(),
        }
        try:
            sig = inspect.signature(cls)
            optional = {
                "use_amf": bool(use_amf),
                "use_lgfm": bool(use_lgfm),
                "use_cga": bool(use_cga),
                "scan_mode": str(scan_mode),
                "global_mode": str(global_mode),
            }
            for key, value in optional.items():
                if key in sig.parameters:
                    kwargs[key] = value
        except (TypeError, ValueError):
            pass
        return cls(**kwargs)
    except BackboneUnavailable:
        raise
    except Exception as exc:
        raise BackboneUnavailable(exc) from exc
