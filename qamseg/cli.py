"""CLI for supervised training and the quality-aware SSL path.

``python train.py`` with no flags does not enter SSL. ``--ssl`` dispatches
to ``qamseg.train_ssl`` before the supervised stack is imported.

``--no-amf``, ``--no-lgfm``, and ``--no-cga`` stay refused unless
``--allow-ablation`` is also passed (or use ``--ablate-amf``,
``--ablate-lgfm``, ``--ablate-cga`` on the SSL path).
"""

import argparse


RESERVED_FLAGS = (
    ('no_amf', '--no-amf', 'AMF ablation is not wired.'),
    ('no_lgfm', '--no-lgfm', 'LGFM ablation is not wired.'),
    ('no_cga', '--no-cga', 'CGA ablation is not wired.'),
)


def build_parser():
    parser = argparse.ArgumentParser(
        prog='train.py',
        description=(
            'Supervised MambaLiteUNet training used as the QAM-Seg backbone. '
            '--ssl runs the quality-aware SSL path (short CUDA smoke on synthetic blobs by default). '
            'Ablation flags are still refused unless --allow-ablation is set.'
        ),
    )
    parser.add_argument(
        '--ssl',
        action='store_true',
        help=(
            'Enable MQE + QGC on MambaLiteUNet. Default is a short CUDA run on synthetic blobs, '
            'not an ISIC/HAM/PH2 experiment. Requires CUDA. See docs/QAM_SEG_SSL.md.'
        ),
    )
    parser.add_argument(
        '--no-amf',
        action='store_true',
        help='Reserved ablation flag (requires --allow-ablation or scripts/count_params_flops.py).',
    )
    parser.add_argument(
        '--no-lgfm',
        action='store_true',
        help='Reserved ablation flag (requires --allow-ablation or scripts/count_params_flops.py).',
    )
    parser.add_argument(
        '--no-cga',
        action='store_true',
        help='Reserved ablation flag (requires --allow-ablation or scripts/count_params_flops.py).',
    )
    parser.add_argument(
        '--allow-ablation',
        action='store_true',
        help='Permit --no-amf, --no-lgfm, and --no-cga to configure MambaLiteUNet ablations.',
    )
    parser.add_argument(
        '--ablate-amf',
        action='store_true',
        help='Bypass Adaptive Multi-branch Mamba Feature Fusion in MambaLiteUNet.',
    )
    parser.add_argument(
        '--ablate-lgfm',
        action='store_true',
        help='Bypass Local-Global Feature Mixing in MambaLiteUNet.',
    )
    parser.add_argument(
        '--ablate-cga',
        action='store_true',
        help='Bypass Cross-Gated Attention in MambaLiteUNet.',
    )
    parser.add_argument(
        '--ssl-scan-mode',
        default='four_way',
        choices=('four_way', 'unidirectional'),
        help='2D scan mode for MambaLiteUNet branches (default: four_way).',
    )
    parser.add_argument(
        '--ssl-global-mode',
        default='attn',
        choices=('attn', 'spectral'),
        help='Global branch inside LGFM: attn (MultiheadAttention) or spectral (2D FFT).',
    )
    parser.add_argument('--ssl-steps', type=int, default=2, help='SSL optimizer steps. Default is a smoke length.')
    parser.add_argument(
        '--ssl-device',
        default='cuda',
        choices=('cpu', 'cuda'),
        help=(
            'Device for the SSL student. Default cuda. mamba_ssm selective scan is CUDA-only. '
            'cpu is refused. If CUDA is unavailable the process exits and does not substitute another student.'
        ),
    )
    parser.add_argument('--ssl-batch-size', type=int, default=2)
    parser.add_argument(
        '--ssl-image-size',
        type=int,
        default=32,
        help='Synthetic spatial size. MambaLiteUNet needs at least 32 (five stride-2 pools).',
    )
    parser.add_argument(
        '--ssl-gate',
        default='gated',
        choices=('hard', 'soft', 'gated', 'ungated'),
        help=(
            'QGC mode. ungated is uniform consistency (plain mean teacher when '
            'quality reweight is also off). Default gated keeps the previous objective.'
        ),
    )
    parser.add_argument('--ssl-level', default='both', choices=('image', 'pixel', 'both'))
    parser.add_argument(
        '--ssl-tau0',
        type=float,
        default=0.35,
        help='MQE image-quality gate start. Scale is fused MQE q in [0,1], not pixel confidence.',
    )
    parser.add_argument(
        '--ssl-tau1',
        type=float,
        default=0.55,
        help=(
            'MQE image-quality gate end. Default 0.55 matches typical MQE q (~0.45-0.75). '
            '0.8 locked the gate shut on ISIC (q rarely reaches 0.8) and collapsed SSL to labeled-only.'
        ),
    )
    parser.add_argument(
        '--ssl-warm-steps',
        type=int,
        default=1,
        help=(
            'Steps to ramp tau0→tau1. Default 1 is the smoke sentinel; for ssl-steps>=10 '
            'the loop expands it to ~30% of steps unless set explicitly.'
        ),
    )
    parser.add_argument('--ssl-lambda-max', type=float, default=1.0)
    parser.add_argument('--ssl-gamma', type=float, default=1.0)
    parser.add_argument('--ssl-seed', type=int, default=0)
    parser.add_argument(
        '--ssl-unlabeled-npy',
        default='',
        help='Optional local unlabeled image .npy. Missing paths error. No dataset download.',
    )
    parser.add_argument(
        '--ssl-labeled-npy',
        default='',
        help='Optional local labeled image .npy. Requires --ssl-mask-npy. No dataset download.',
    )
    parser.add_argument('--ssl-mask-npy', default='', help='Mask .npy paired with --ssl-labeled-npy.')
    parser.add_argument(
        '--ssl-split',
        default='',
        help=(
            'Fraction JSON from scripts/make_labeled_fraction_splits.py '
            '(fraction_5.json, fraction_10.json, or fraction_20.json). '
            'Requires local train .npy paths. Missing files error. No download.'
        ),
    )
    parser.add_argument(
        '--ssl-backbone',
        default='mamba',
        choices=('mamba',),
        help=(
            'SSL student. mamba (the only choice, and the default) is models.MambaLiteUNet. '
            'Import or construction failure exits non-zero. No other student is substituted. '
            'Ignored unless --ssl is set. Not a CVPR run.'
        ),
    )
    parser.add_argument(
        '--use-learned-quality-head',
        action='store_true',
        help='Acknowledge the MQE head stub. It is not trained and its fusion weight stays 0.',
    )
    parser.add_argument(
        '--ssl-baq',
        action='store_true',
        help='Multiply pixel weights by the optional boundary-aware prior. Off by default.',
    )
    parser.add_argument(
        '--ssl-cutmix',
        action='store_true',
        help='Apply Quality-Guided CutMix across unlabeled batches when batch_size >= 2.',
    )
    parser.add_argument(
        '--ssl-qar',
        default='off',
        choices=('off', 'prototype'),
        help=(
            'Quality-aware regularizer mode. off (default) skips QAR. '
            'prototype enables Quality-Guided Feature Prototype Alignment (QG-PA) on boundary features.'
        ),
    )
    parser.add_argument(
        '--ssl-lambda-qar',
        type=float,
        default=0.1,
        help='Weight on the prototype QAR term when --ssl-qar prototype is active.',
    )
    parser.add_argument(
        '--quality-reweight',
        default='off',
        choices=('off', 'sample', 'pixel'),
        help=(
            'MQE reweight of the unlabeled term (QG-SSL PL-QW analogue, no g_phi). '
            'off (default) keeps QGC. sample weights each image by the ramped MQE score. '
            'pixel weights each pixel by the ramped MQE map. See docs/QAM_SEG_SSL.md.'
        ),
    )
    parser.add_argument(
        '--ssl-lambda-qw',
        type=float,
        default=1.0,
        help=(
            'Coefficient on the reweighted unlabeled term. Ignored when --quality-reweight off. '
            'Default 1 keeps the unlabeled scale aligned with --ssl-lambda-max. '
            'QG-SSL used 0.25; pass 0.25 to match that coefficient.'
        ),
    )
    parser.add_argument(
        '--ssl-qw-ramp',
        default='gaussian',
        choices=('gaussian', 'linear', 'none'),
        help=(
            'Ramp of the MQE weight from uniform (step 0) to the quality score (last step). '
            'gaussian matches the lambda_u shape. none uses the quality score immediately.'
        ),
    )
    parser.add_argument(
        '--ssl-method',
        default='qam',
        choices=('qam', 'mt', 'uamt', 'pl', 'supervised'),
        help=(
            'Recipe preset. qam (default) honors the gate, BAQ, and reweight flags. '
            'mt, uamt, pl, and supervised replace those with a fair baseline on the same student. '
            'uamt uses teacher entropy, not dropout inside MambaLiteUNet.'
        ),
    )
    parser.add_argument(
        '--ssl-uncertainty',
        default='off',
        choices=('off', 'entropy'),
        help='UA-MT weight from teacher predictive entropy. Off unless --ssl-method uamt sets it.',
    )
    parser.add_argument(
        '--ssl-uncertainty-tau',
        type=float,
        default=0.75,
        help='Drop pixels whose normalized entropy exceeds this value. 1 keeps every pixel. Placeholder.',
    )
    parser.add_argument(
        '--ssl-pl-threshold',
        type=float,
        default=0.0,
        help='Confidence-threshold pseudo-labels. 0 disables. --ssl-method pl uses 0.95 unless this is set.',
    )
    parser.add_argument(
        '--ssl-recipe',
        default='',
        help='Name stored in metrics.json (for example qam-gated-baq-qw). Does not change the loss.',
    )
    parser.add_argument(
        '--ssl-group',
        default='',
        help='Table group stored in metrics.json (ablation, baseline, or supervised). Does not change the loss.',
    )
    parser.add_argument(
        '--ssl-fraction',
        default='',
        help='Fraction token stored in metrics.json (5, 10, 20, or full). Does not select the split by itself.',
    )
    parser.add_argument(
        '--ssl-out-dir',
        default='',
        help='If set, write metrics.json (Dice, mIoU, HD95, ASSD) for scripts/aggregate_media_results.py.',
    )
    parser.add_argument(
        '--ssl-eval-npy',
        default='',
        help=(
            'Optional held-out image .npy used only for eval. data_val.npy and data_test.npy are allowed here. '
            'They are still refused as unlabeled training data. No download.'
        ),
    )
    parser.add_argument(
        '--ssl-eval-mask-npy',
        default='',
        help='Mask .npy paired with --ssl-eval-npy. Eval only.',
    )
    return parser


def parse_train_args(argv=None):
    """Return (args, unknown, reserved_names).

    `reserved_names` lists ablation flags that were actually passed without
    `--allow-ablation`.
    `unknown` is ignored so extra tokens do not change the supervised run.
    """
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if getattr(args, 'allow_ablation', False):
        reserved = []
    else:
        reserved = [flag for attr, flag, _reason in RESERVED_FLAGS if getattr(args, attr)]
    return args, unknown, reserved


def resolve_ssl_method(args):
    """Apply a baseline preset.

    ``qam`` leaves the flags alone. The other methods force a shared-student
    baseline so a gate or MQE reweight cannot leak into MT / UA-MT / PL /
    supervised. Hybrids use ``--ssl-method qam`` plus the low-level flags.
    """
    import copy

    method = str(args.ssl_method)
    if method == 'qam':
        return args
    resolved = copy.copy(args)
    resolved.ssl_gate = 'ungated'
    resolved.quality_reweight = 'off'
    resolved.ssl_baq = False
    resolved.ssl_cutmix = False
    resolved.ssl_qar = 'off'
    resolved.ssl_uncertainty = 'off'
    resolved.ssl_pl_threshold = 0.0
    if method == 'mt':
        return resolved
    if method == 'uamt':
        resolved.ssl_uncertainty = 'entropy'
        resolved.ssl_uncertainty_tau = float(args.ssl_uncertainty_tau)
        return resolved
    if method == 'pl':
        threshold = float(args.ssl_pl_threshold)
        resolved.ssl_pl_threshold = 0.95 if threshold <= 0.0 else threshold
        return resolved
    if method == 'supervised':
        resolved.ssl_lambda_max = 0.0
        return resolved
    raise ValueError(f'Unknown SSL method: {method}')


def refuse_reserved_flags(reserved):
    """Exit 2 when an ablation flag was passed without --allow-ablation."""
    if not reserved:
        return
    reasons = {flag: reason for _attr, flag, reason in RESERVED_FLAGS}
    print('QAM-Seg: ablation flags are scaffolds only and are not wired into training.')
    print('Pass --allow-ablation (or --ablate-amf / --ablate-lgfm / --ablate-cga) to enable them.')
    print('--ssl is separate and does not enable these bypasses without --allow-ablation.')
    print('See docs/QAM_SEG_SSL.md.')
    for flag in reserved:
        print(f'  {flag}: {reasons[flag]}')
    print('Refusing to start. Requested: ' + ' '.join(reserved))
    raise SystemExit(2)
