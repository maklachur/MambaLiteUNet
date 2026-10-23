#!/usr/bin/env python3
"""Print parameter count and, when a profiler imports, FLOPs at 256x256.

Profiles the MambaLiteUNet class in this fork, including optional ablation
bypasses (--no-amf, --no-lgfm, --no-cga) and scan/mixing modes (--scan-mode,
--global-mode). It does not download data or checkpoints, does not train, and
does not report the CVPR paper's complexity table.
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_C_LIST = [16, 32, 48, 64, 96, 128]


def _read_c_list():
    """Read model_config c_list without importing the training config."""
    path = ROOT / 'configs' / 'config_setting.py'
    text = path.read_text(encoding='utf-8')
    match = re.search(r"""['"]c_list['"]\s*:\s*\[([0-9,\s]+)\]""", text)
    if not match:
        print(
            'warning: could not parse c_list from configs/config_setting.py; '
            f'using {DEFAULT_C_LIST}'
        )
        return list(DEFAULT_C_LIST)
    values = [int(part) for part in match.group(1).split(',') if part.strip()]
    if not values:
        print(f'warning: empty c_list parse; using {DEFAULT_C_LIST}')
        return list(DEFAULT_C_LIST)
    return values


def _stub(reason):
    print('status: stub')
    print(f'reason: {reason}')
    print()
    print('This script did not measure params or FLOPs.')
    print('It does not download datasets or checkpoints.')
    print()
    print('When the supervised environment is installed, re-run:')
    print('  pip install torch torchvision')
    print('  pip install timm einops')
    print('  pip install causal_conv1d mamba_ssm')
    print('  pip install thop')
    print('  # optional alternative profiler: pip install fvcore')
    print('  python scripts/count_params_flops.py')
    print()
    print('The successful run prints params for MambaLiteUNet and FLOPs')
    print('for one forward of a 1x3x256x256 tensor (override with --height/--width).')
    print('mamba_ssm\'s selective scan often requires a CUDA forward.')
    print('Parameter counting itself does not run the model.')
    print('Do not treat the output as the CVPR paper complexity table.')


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'params_total: {total}')
    print(f'params_trainable: {trainable}')
    print(f'params_million: {total / 1e6:.6f}')
    return total, trainable


def _try_thop(model, dummy):
    try:
        from thop import profile
    except ImportError:
        print('thop: not installed')
        return False
    try:
        macs, params = profile(model, inputs=(dummy,), verbose=False)
    except Exception as exc:
        print(f'thop: failed ({type(exc).__name__}: {exc})')
        return False
    print(f'thop_macs: {macs}')
    print(f'thop_gmacs: {macs / 1e9:.6f}')
    print(f'thop_params: {params}')
    print(
        'note: thop\'s first return is multiply-accumulate count. '
        'This script does not rescale it and does not equate it with a paper GFLOP number.'
    )
    return True


def _try_fvcore(model, dummy):
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print('fvcore: not installed')
        return False
    try:
        flop_count = FlopCountAnalysis(model, dummy)
        total = flop_count.total()
    except Exception as exc:
        print(f'fvcore: failed ({type(exc).__name__}: {exc})')
        return False
    print(f'fvcore_flops: {total}')
    print(f'fvcore_gflops: {total / 1e9:.6f}')
    print(
        'note: fvcore FlopCountAnalysis counts FLOPs under its own op set. '
        'Unsupported Mamba kernels can be omitted. This is not a paper-table figure.'
    )
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Count MambaLiteUNet params and FLOPs at 256x256 without training.'
    )
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument(
        '--device',
        default='cpu',
        help='cpu (default) or cuda. FLOP tools run a forward; params do not.',
    )
    parser.add_argument('--no-amf', action='store_true', help='Bypass AMF blocks.')
    parser.add_argument('--no-lgfm', action='store_true', help='Bypass LGFM blocks.')
    parser.add_argument('--no-cga', action='store_true', help='Bypass CGA skip gates.')
    parser.add_argument(
        '--scan-mode',
        default='four_way',
        choices=('four_way', 'unidirectional'),
    )
    parser.add_argument(
        '--global-mode',
        default='attn',
        choices=('attn', 'spectral'),
    )
    args = parser.parse_args(argv)

    c_list = _read_c_list()
    print('QAM-Seg complexity probe')
    print('backbone: MambaLiteUNet (supervised code in this fork)')
    print(f'c_list: {c_list}')
    print(
        f'config: use_amf={not args.no_amf}, use_lgfm={not args.no_lgfm}, '
        f'use_cga={not args.no_cga}, scan_mode={args.scan_mode}, global_mode={args.global_mode}'
    )
    print(f'input: ({args.batch}, 3, {args.height}, {args.width})')
    print('paper_table: not reported by this script')

    try:
        import torch
    except ImportError:
        _stub('PyTorch is not installed.')
        return 0

    try:
        from models.MambaLiteUNet import MambaLiteUNet
    except ImportError as exc:
        _stub(
            'Could not import models.MambaLiteUNet '
            f'({exc}). timm and mamba_ssm are required to construct the class.'
        )
        return 0

    try:
        model = MambaLiteUNet(
            num_classes=1,
            input_channels=3,
            c_list=c_list,
            use_amf=not args.no_amf,
            use_lgfm=not args.no_lgfm,
            use_cga=not args.no_cga,
            scan_mode=args.scan_mode,
            global_mode=args.global_mode,
        )
        model.eval()
        device_name = args.device
        if device_name.startswith('cuda') and not torch.cuda.is_available():
            print('cuda requested but torch.cuda.is_available() is false; using cpu')
            device_name = 'cpu'
        device = torch.device(device_name)
        model.to(device)
        _count_params(model)
        dummy = torch.randn(args.batch, 3, args.height, args.width, device=device)
        thop_ok = _try_thop(model, dummy)
        fvcore_ok = _try_fvcore(model, dummy)
        if not thop_ok and not fvcore_ok:
            print('flops: unavailable')
            print('Install thop or fvcore and re-run. Params above are still valid.')
            print(
                'If a profiler imports but the forward fails, mamba_ssm may be '
                'CUDA-only on this machine. Retry with --device cuda on a GPU box.'
            )
        print('status: ok')
    except Exception as exc:
        print('status: error')
        print(f'error: {type(exc).__name__}: {exc}')
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
