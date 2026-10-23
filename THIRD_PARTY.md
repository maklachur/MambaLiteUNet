# Third-party notice

## MambaLiteUNet (upstream prior)

This repository is a fork of **MambaLiteUNet** and keeps that project's
supervised skin-lesion segmentation code as the backbone for QAM-Seg
(Quality-Aware Mamba Segmentation), a separate quality-aware
semi-supervised learning effort.

QAM-Seg uses the upstream modules as released. It does **not** claim
coauthorship of:

- Adaptive Multi-branch Mamba Feature Fusion (AMF)
- Local-Global Feature Mixing (LGFM)
- Cross-Gated Attention (CGA)

and it does **not** claim the CVPR 2026 metrics as QAM-Seg results.

| | |
| --- | --- |
| Paper | MambaLiteUNet: Cross-Gated Adaptive Feature Fusion for Robust Skin Lesion Segmentation |
| Authors | Md Maklachur Rahman, Soon Ki Jung, and Tracy Hammond |
| Venue | Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 8556–8565 |
| arXiv | [2604.20286](https://arxiv.org/abs/2604.20286) |
| Code | https://github.com/maklachur/MambaLiteUNet |
| Open access | https://openaccess.thecvf.com/content/CVPR2026/html/Rahman_MambaLiteUNet_Cross-Gated_Adaptive_Feature_Fusion_for_Robust_Skin_Lesion_Segmentation_CVPR_2026_paper.html |

```bibtex
@inproceedings{rahman2026mambaliteunet,
  title={MambaLiteUNet: Cross-Gated Adaptive Feature Fusion for Robust Skin Lesion Segmentation},
  author={Rahman, Md Maklachur and Jung, Soon Ki and Hammond, Tracy},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8556--8565},
  year={2026}
}
```

Author names, title, pages, and year are taken from the citation block shipped
in the upstream README. Upstream ships no LICENSE file; see `LICENSE` in this
fork for how that is handled.

## Code acknowledged by the upstream README

The upstream README thanks the authors of these projects. They have their own
licenses. This fork does not relicense them.

- VMamba — https://github.com/MzeroMiko/VMamba
- Vision Mamba — https://github.com/hustvl/Vim
- VM-UNet — https://github.com/JCruan519/VM-UNet
- UltraLight-VM-UNet — https://github.com/wurenkai/UltraLight-VM-UNet

## Runtime packages

`requirements.txt` pins third-party packages, including `mamba_ssm` and
`causal_conv1d`. Those packages are not vendored here.

Checkpoints under `pre_trained_models/` are the weights shipped by the
upstream repository for its supervised model. They are not QAM-Seg results.
