"""QAM-Seg additions on top of the MambaLiteUNet supervised backbone.

MQE, QGC, MQE loss reweighting, labeled-fraction splits, and the
MambaLiteUNet SSL student live in this package. Tensor tests cover MQE,
QGC, and reweight without a GPU. MambaLiteUNet smokes require CUDA.
They are not experimentally validated. See docs/QAM_SEG_SSL.md and
docs/MEDIA_NOTES.md. There is no separate quality-predictor network.
"""
