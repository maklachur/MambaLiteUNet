# MedIA notes for QAM-Seg

Status: protocol, training hooks, architectural fixes, and representation-level
SSL upgrades are implemented and verified by unit tests (`tests/`).

## Summary of Fixes & Upgrades for MedIA Submission

1. **Backbone Bug Fixes & 4-Way State-Space Disagreement (`models/MambaLiteUNet.py`)**:
   - Protects `mamba_ssm.Mamba` internal initialization (`dt_proj.bias`, `A_log`, `D`) from being overwritten by `_init_weights`.
   - Implements 4-way directional cross-scanning ($\rightarrow, \leftarrow, \downarrow, \uparrow$) across the 4 branches of `AMF` and `CGA`.
   - Extracts per-pixel directional branch variance (`last_branch_disagreement`) and feeds it into `MQE` (`QualityWeights.mamba`), directly coupling the Mamba state-space representation with pseudo-label quality estimation.
   - Wires `--no-amf`, `--no-lgfm`, and `--no-cga` in `MambaLiteUNet` and `scripts/count_params_flops.py`.

2. **SSL Pipeline Bug Fixes (`qamseg/`)**:
   - **Weak/Strong Coordinate Alignment**: `strong_augment` now shares `weak_augment`'s geometric frame and applies photometric contrast/brightness scaling + cutout + optional noise, fixing the $180^\circ$ flip mismatch.
   - **Full Unlabeled Pool Coverage**: `iter_dual_batches` iterates across `max(len(labeled_loader), len(unlabeled_loader))` batches per epoch with `shuffle=True` and labeled spatial augmentations (`augment_labeled_batch`).
   - **Uncancelled Image-Level Quality Weights**: `quality_gated_consistency_loss` normalizes spatial weights per active image while preserving the continuous image quality weight $q^\gamma$.
   - **Quality-Guided Feature Prototype Alignment (`--ssl-qar prototype`) & CutMix (`--ssl-cutmix`)**: Adds representation-level boundary regularization (`prototype_quality_regularization`) and region mixing (`quality_guided_cutmix`), available via recipes `qam-qar` and `qam-full`.

The protocol runner (`scripts/run_media_protocol.py`) defaults to MambaLiteUNet (`--backbone mamba`) and `--device cuda`. `--smoke` is a short synthetic wiring check on that same student. `mamba_ssm` selective scan is CUDA-only, so a machine without CUDA exits instead of substituting another student.

## Recipes in `qamseg/protocol.py`

| Recipe | Description |
| --- | --- |
| `supervised` | Supervised baseline on 5%, 10%, 20%, and 100% (`full`) splits |
| `mt` | Plain Mean Teacher (`ungated`, uniform consistency) |
| `uamt` | Entropy-weighted Mean Teacher (`uncertainty=entropy`, `tau=0.75`) |
| `pl` | Confidence-thresholded Pseudo-Labeling (`pl_threshold=0.95`) |
| `qam-ungated` | QAM-Seg ungated consistency |
| `qam-gated` | QAM-Seg quality-gated consistency (`QGC`) |
| `qam-gated-baq` | `QGC` + smoothed Boundary-Aware Quality prior (`BAQ`) |
| `qam-gated-baq-qw` | `QGC` + `BAQ` + sample-level MQE reweighting |
| `qam-qw` | Ungated sample-level MQE reweighting (direct PL-QW analogue) |
| `qam-qw-pixel` | Ungated pixel-level MQE reweighting |
| `qam-qar` | `qam-gated-baq-qw` + Quality-Guided Feature Prototype Alignment (`--ssl-qar prototype`) |
| `qam-full` | `qam-qar` + Quality-Guided CutMix (`--ssl-cutmix`) |

## Gate threshold defaults (MQE q scale)

Defaults are `--ssl-tau0 0.35` → `--ssl-tau1 0.55` with warm ≈ `0.3 * steps` when `--ssl-warm-steps` is left at the smoke value `1`. An older `0.4→0.8` schedule shut the hard image gate on ISIC (MQE `q` typically stays ~0.45–0.75), so unlabeled loss stayed at 0 for most steps. `metrics.json` records `mean_gate_keep` / `frac_gate_open`; per-step rows go to `train_log.jsonl`.
