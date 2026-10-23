# QAM-Seg quality-aware SSL

Status: **MQE, QGC, Mamba 4-way directional branch disagreement, QG-CutMix, and
Quality-Guided Feature Prototype Alignment (QG-PA) are implemented in code and
covered by synthetic unit tests (`tests/`).**

## Key Fixes & Journal-Level Upgrades in This Version

1. **MambaLiteUNet Architecture Fixes (`models/MambaLiteUNet.py`)**:
   - **Protected Mamba S4D/HiPPO Initialization**: `MambaSSMLayer` marks internal `mamba_ssm.Mamba` submodules with `_is_mamba_internal = True` so `self.apply(self._init_weights)` no longer overwrites `dt_proj.bias` (inverse-softplus $\Delta$) or `A_log` / `D`.
   - **4-Way Multi-Directional Scanning (`scan_2d` / `unscan_2d`)**: The 4 parallel branches inside `AdaptiveMultiBranchMambaFeatureFusion` (`AMF`) and `CrossGatedAttentionGate` (`CGA`) now scan along 4 complementary 2D trajectories (row-major forward, row-major backward, column-major forward, column-major backward) and restore canonical spatial order at zero extra parameter cost.
   - **State-Space Epistemic Disagreement (`last_branch_disagreement`)**: `AMF` computes the per-pixel variance across the 4 directional Mamba branches and exposes an aggregated `(B, 1, H, W)` map on `model.last_aux["branch_disagreement"]`, feeding directly into `MQE` (`QualityWeights.mamba`).
   - **Spectral Global Mixing & Non-Square Support (`LocalGlobalFeatureMixing`)**: Supports explicit `(H, W)` dimensions and an $O(N \log N)$ 2D FFT spectral gating mode (`--ssl-global-mode spectral`), plus active affine parameters in `DepthwiseConvolution`.
   - **Wired Ablation Bypasses**: `--no-amf`, `--no-lgfm`, and `--no-cga` are wired in `MambaLiteUNet`, `scripts/count_params_flops.py`, and `train.py` (when `--allow-ablation` is set, or via `--ablate-amf`, `--ablate-lgfm`, `--ablate-cga`).

2. **SSL Training Pipeline Fixes (`qamseg/`)**:
   - **Weak/Strong Coordinate Alignment (`qamseg/augment.py`)**: Fixed the $180^\circ$ geometric mismatch where `weak_augment` flipped horizontally while `strong_augment` flipped vertically without inverse-warping `teacher_weak`. Both views now share a common geometric frame while `strong_augment` applies photometric contrast/brightness perturbation, cutout, and optional Gaussian noise.
   - **Full Unlabeled Pool Iteration & Labeled Augmentation (`qamseg/data.py`)**: `iter_dual_batches` now iterates `max(len(labeled_loader), len(unlabeled_loader))` steps per pass so 100% of the unlabeled pool is visited every epoch, with seeded shuffling (`shuffle=True`) and geometric augmentations (`rot90` + flips) on labeled training data.
   - **Uncancelled Image-Level Quality Scaling (`qamseg/qgc.py`, `qamseg/reweight.py`)**: Fixed the `weights.sum()` denominator bug that canceled out scalar image quality weights $q^\gamma$ when `batch_size == 1` or only one sample passed $q \ge \tau$.
   - **Quality-Guided Feature Prototype Alignment (`qamseg/qar.py`, `--ssl-qar prototype`)**: Replaces the skipped output-probability QAR with a non-collapsing representation-level boundary regularizer (`prototype_quality_regularization`) that pulls ambiguous contour-band student features toward high-reliability teacher foreground/background prototypes.
   - **Quality-Guided CutMix (`qamseg/augment.py`, `--ssl-cutmix`)**: Mixes pairs of unlabeled images and warps teacher pseudo-masks and MQE pixel/image quality weights consistently.
   - **Adaptive Threshold Warmup & Cosine LR (`qamseg/train_ssl.py`)**: Default `--ssl-tau0/--ssl-tau1` are `0.35/0.55` on the **MQE q** scale (not pixel confidence); `0.8` locked the image gate shut on ISIC. Automatically scales `warm_steps` to `0.3 * steps` on multi-step training runs (`steps >= 10`) when left at the 1-step smoke default, adds `CosineAnnealingLR`, and logs EMA teacher evaluation metrics (`teacher_metrics`).

## CLI Summary

| Flag | Meaning |
| --- | --- |
| `--ssl` | Run the QAM-Seg SSL training loop (`qamseg.train_ssl`) on MambaLiteUNet |
| `--ssl-backbone` | `mamba` only (`models.MambaLiteUNet`). This is the default student. A failed import exits. No other student is substituted. |
| `--ssl-device {cpu,cuda}` | `cuda` (default). `mamba_ssm` selective scan is CUDA-only. `cpu`, or `cuda` when CUDA is missing, exits. No other student is substituted. |
| `--ssl-image-size` | Synthetic spatial size. Default 32. MambaLiteUNet needs at least 32 (five stride-2 pools). |
| `--ssl-split PATH` | Slice local train `.npy` arrays with `fraction_5.json`, `fraction_10.json`, or `fraction_20.json` |
| `--ssl-gate {gated,hard,soft,ungated}` | QGC gating mode |
| `--quality-reweight {off,sample,pixel}` | MQE reweighting of the unlabeled consistency loss |
| `--ssl-baq` | Smoothed boundary-aware pixel prior using `(1 - entropy)` |
| `--ssl-cutmix` | Enable Quality-Guided CutMix across unlabeled batches |
| `--ssl-qar {off,prototype}` | `off` (default) or `prototype` (Quality-Guided Feature Prototype Alignment) |
| `--ssl-scan-mode {four_way,unidirectional}` | 4-way directional scanning (default) or unidirectional scan in MambaLiteUNet |
| `--ssl-global-mode {attn,spectral}` | Attention or 2D FFT spectral global mixing inside LGFM |
| `--allow-ablation` | Enable `--no-amf`, `--no-lgfm`, and `--no-cga` bypasses during training |

## Running Unit Tests

```bash
python -m unittest discover -s tests -v
```

MambaLiteUNet smokes (`python train.py --ssl`, protocol `--smoke`) require CUDA. The suite skips those smokes when CUDA or `mamba_ssm` is missing. MQE, QGC, and reweight tensor tests do not construct the student and still run.
