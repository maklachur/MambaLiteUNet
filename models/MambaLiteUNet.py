import math
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm import Mamba  # Ensure mamba_ssm is installed and accessible


def scan_2d(x: torch.Tensor, H: int, W: int, direction: int = 0) -> torch.Tensor:
    """Reorder flattened tokens (B, H*W, C) along one of 4 canonical 2D scan paths.

    Directions:
        0: row-major forward (top-left -> bottom-right)
        1: row-major backward (bottom-right -> top-left)
        2: column-major forward (top-left -> bottom-right, column-first)
        3: column-major backward (bottom-right -> top-left, column-first)
    """
    B, N, C = x.shape
    if N != H * W:
        raise ValueError(f"scan_2d expects N=H*W, got N={N}, H*W={H * W}")
    direction = int(direction) % 4
    if direction == 0:
        return x
    if direction == 1:
        return torch.flip(x, dims=(1,))
    grid = x.view(B, H, W, C)
    if direction == 2:
        return grid.transpose(1, 2).contiguous().view(B, N, C)
    return torch.flip(grid.transpose(1, 2).contiguous().view(B, N, C), dims=(1,))


def unscan_2d(y: torch.Tensor, H: int, W: int, direction: int = 0) -> torch.Tensor:
    """Inverse of scan_2d: restore canonical row-major (B, H*W, C) token order."""
    B, N, C = y.shape
    if N != H * W:
        raise ValueError(f"unscan_2d expects N=H*W, got N={N}, H*W={H * W}")
    direction = int(direction) % 4
    if direction == 0:
        return y
    if direction == 1:
        return torch.flip(y, dims=(1,))
    if direction == 2:
        grid = y.view(B, W, H, C)
        return grid.transpose(1, 2).contiguous().view(B, N, C)
    grid = torch.flip(y, dims=(1,)).view(B, W, H, C)
    return grid.transpose(1, 2).contiguous().view(B, N, C)


# Depth-Wise Convolution Module
class DepthwiseConvolution(nn.Module):
    def __init__(self, dim=None):
        super(DepthwiseConvolution, self).__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim
        )
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x, H, W):
        B, N, C = x.shape
        if N != H * W:
            raise ValueError(
                f"DepthwiseConvolution expects N=H*W, but got N={N}, HxW={H}x{W}"
            )
        x = x.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)
        x = F.layer_norm(x, [H, W])             # LN over spatial dims
        # Apply learnable channel-wise affine scale and bias (previously dead parameters)
        x = x * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        x = self.dwconv(x)                      # Depthwise conv
        x = x.flatten(2).transpose(1, 2)        # (B, N, C)
        return x


# Mamba SSM-based Layer Wrapper
class MambaSSMLayer(nn.Module):
    """
    A wrapper around the Mamba SSM-based layer with optional 2D directional scanning
    and protection for Mamba's internal HiPPO / delta initialization.
    Expects input as either (B, C, H, W) or (B, N, C).
    """

    def __init__(self, d_model, d_state, d_conv, expand, direction: int = 0):
        super(MambaSSMLayer, self).__init__()
        self.direction = int(direction) % 4
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # Protect Mamba's specialized S4D/HiPPO A_log, D, and inverse-softplus dt_proj.bias
        # from being overwritten by generic _init_weights calls.
        for submodule in self.mamba.modules():
            submodule._is_mamba_internal = True
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if getattr(m, "_is_mamba_internal", False):
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H: Optional[int] = None, W: Optional[int] = None):
        reshaped_4d = (x.dim() == 4)
        if reshaped_4d:
            B, C, H_in, W_in = x.shape
            H = H_in if H is None else H
            W = W_in if W is None else W
            x = x.view(B, C, H * W).transpose(1, 2)  # (B, N, C)
        elif x.dim() != 3:
            raise ValueError(
                f"MambaSSMLayer expects input with 3 or 4 dimensions, got {x.dim()}"
            )

        B, N, C = x.shape
        if H is not None and W is not None and H * W == N and self.direction != 0:
            x = scan_2d(x, H, W, self.direction)
            x = self.mamba(x)
            x = unscan_2d(x, H, W, self.direction)
        else:
            x = self.mamba(x)
        return x


class SpectralGating2D(nn.Module):
    """O(N log N) 2D FFT spectral global mixing alternative to O(N^2) MultiheadAttention."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.complex_weight = nn.Parameter(torch.ones(dim, 1, 1, 2) * 0.5)
        self.pw = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        spatial = x.transpose(1, 2).view(B, C, H, W).float()
        fft_feat = torch.fft.rfft2(spatial, norm="ortho")
        weight = torch.view_as_complex(self.complex_weight.float())
        fft_feat = fft_feat * weight.unsqueeze(0)
        out = torch.fft.irfft2(fft_feat, s=(H, W), norm="ortho")
        out = out.to(dtype=x.dtype).flatten(2).transpose(1, 2)
        return self.pw(out)


# Local-Global Feature Mixing Module
class LocalGlobalFeatureMixing(nn.Module):
    def __init__(self, input_dim, num_heads=8, global_mode: str = "attn"):
        super(LocalGlobalFeatureMixing, self).__init__()
        if global_mode not in ("attn", "spectral"):
            raise ValueError(f"Unknown global_mode: {global_mode}")
        self.global_mode = global_mode
        self.local_conv = nn.Conv1d(
            input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim
        )
        if global_mode == "attn":
            self.global_attn = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=num_heads, batch_first=True
            )
            self.spectral_gate = None
        else:
            self.global_attn = None
            self.spectral_gate = SpectralGating2D(input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim * 2, input_dim)
        self.act = nn.GELU()
        self.dwconv = DepthwiseConvolution(input_dim)

    def forward(self, x, H: Optional[int] = None, W: Optional[int] = None):
        # x: (B, N, C)
        B, N, C = x.shape
        if H is None or W is None:
            side = int(math.isqrt(N))
            if side * side != N:
                raise ValueError(
                    f"Feature map is not square and (H, W) were not provided. Got N={N}."
                )
            H = W = side
        elif H * W != N:
            raise ValueError(f"Expected N=H*W, got N={N}, H={H}, W={W}.")

        # Local branch
        x_local = x.transpose(1, 2)  # (B, C, N)
        x_local = self.local_conv(x_local)
        x_local = x_local.transpose(1, 2)  # (B, N, C)

        # Global branch
        if self.global_mode == "spectral" and self.spectral_gate is not None:
            x_global = self.spectral_gate(x, H, W)
        else:
            x_global, _ = self.global_attn(x, x, x)

        # Combine local and global
        x_cat = torch.cat([x_local, x_global], dim=-1)  # (B, N, 2C)
        x_out = self.fc(x_cat)                          # (B, N, C)
        x_out = self.act(self.norm(x_out))

        # Apply DepthwiseConvolution for enhanced spatial feature extraction
        x_out = self.dwconv(x_out, H, W)  # (B, N, C)

        return x_out


# Adaptive Multi-Branch Mamba Feature Fusion
class AdaptiveMultiBranchMambaFeatureFusion(nn.Module):
    """
    Takes (B, C, H, W) as input and internally reshapes to (B, H*W, C),
    splits channels across multiple branches with 4-way directional scanning,
    passes them through MambaSSMLayer, records inter-branch state-space disagreement,
    merges again, and outputs (B, C_out, H, W).
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_branches=4,
        d_state=16,
        d_conv=4,
        expand=2,
        use_lgfm: bool = True,
        scan_mode: str = "four_way",
        global_mode: str = "attn",
    ):
        super(AdaptiveMultiBranchMambaFeatureFusion, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_branches = num_branches
        self.use_lgfm = bool(use_lgfm)
        self.scan_mode = scan_mode
        self.norm = nn.LayerNorm(input_dim)
        self.last_branch_disagreement: Optional[torch.Tensor] = None

        assert input_dim % num_branches == 0, (
            f"input_dim({input_dim}) must be divisible by num_branches({num_branches})"
        )
        self.mamba_layers = nn.ModuleList([
            MambaSSMLayer(
                d_model=input_dim // num_branches,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                direction=(i % 4) if scan_mode == "four_way" else 0,
            )
            for i in range(num_branches)
        ])

        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

        # Adaptive Gating Mechanisms
        # Q-branch gating
        self.conv_q_dw = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_branches,
            kernel_size=3,
            padding=1,
            groups=num_branches,
        )
        self.conv_q_pw = nn.Conv1d(
            in_channels=num_branches,
            out_channels=num_branches,
            kernel_size=1,
        )
        # R-branch gating
        self.conv_r_dw = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=3,
            padding=1,
            groups=input_dim,
        )
        self.conv_r_pw = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=1,
        )
        # Final depthwise conv
        self.dw_conv_final_dw = nn.Conv1d(
            in_channels=2 * input_dim,
            out_channels=2 * input_dim,
            kernel_size=3,
            padding=1,
            groups=2 * input_dim,
        )
        self.dw_conv_final_pw = nn.Conv1d(
            in_channels=2 * input_dim,
            out_channels=input_dim,
            kernel_size=1,
        )
        self.sigmoid = nn.Sigmoid()

        # Local-Global Feature Mixing
        if self.use_lgfm:
            self.local_global_mixing = LocalGlobalFeatureMixing(
                input_dim, global_mode=global_mode
            )
        else:
            self.local_global_mixing = None
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W

        # Flatten => (B, N, C)
        x = x.view(B, C, N).transpose(1, 2)
        x_norm = self.norm(x)

        # Split into multiple branches along channel dimension
        x_chunks = torch.chunk(x_norm, self.num_branches, dim=2)

        # Pass each chunk through directional MambaSSMLayer + skip
        x_mamba_chunks = [
            self.mamba_layers[i](x_chunk, H=H, W=W) + self.skip_scale * x_chunk
            for i, x_chunk in enumerate(x_chunks)
        ]

        # Compute directional Mamba branch disagreement map (B, 1, H, W) in [0, 1]
        with torch.no_grad():
            branch_probs = torch.stack(
                [torch.sigmoid(chunk.detach()).mean(dim=-1) for chunk in x_mamba_chunks],
                dim=1,
            )  # (B, num_branches, N)
            # Max variance of Bernoulli variables in [0, 1] is 0.25; scale by 4 to [0, 1]
            disagreement = (branch_probs.var(dim=1, unbiased=False) * 4.0).clamp(0.0, 1.0)
            self.last_branch_disagreement = disagreement.view(B, 1, H, W)

        # Concatenate => (B, N, C)
        x_concat = torch.cat(x_mamba_chunks, dim=2)

        # Adaptive Gating: Q
        x_concat_t = x_concat.transpose(1, 2)  # (B, C, N)
        Q_dw = self.conv_q_dw(x_concat_t)      # (B, num_branches, N)
        Q = self.conv_q_pw(Q_dw)               # (B, num_branches, N)
        g = self.sigmoid(Q)                    # (B, num_branches, N)

        # Gate each chunk
        g_chunks = torch.chunk(g, self.num_branches, dim=1)
        x_mamba_chunks_gated = [
            g_chunk.permute(0, 2, 1) * x_mamba_chunk
            for g_chunk, x_mamba_chunk in zip(g_chunks, x_mamba_chunks)
        ]
        x_gated = torch.cat(x_mamba_chunks_gated, dim=2)  # (B, N, C)

        # Adaptive Gating: R
        x_gated_t = x_gated.transpose(1, 2)    # (B, C, N)
        x_gated_t_residual = x_gated_t
        R_dw = self.conv_r_dw(x_gated_t)       # (B, C, N)
        R = self.conv_r_pw(R_dw)               # (B, C, N)
        R = R + x_gated_t_residual             # (B, C, N)
        R = R.transpose(1, 2)                  # (B, N, C)

        # Final depthwise conv
        concat_NR = torch.cat([x, R], dim=2)               # (B, N, 2C)
        concat_NR_t = concat_NR.transpose(1, 2)            # (B, 2C, N)
        x_mamba_t_dw = self.dw_conv_final_dw(concat_NR_t)  # (B, 2C, N)
        x_mamba_t = self.dw_conv_final_pw(x_mamba_t_dw)    # (B, C, N)
        # Residual from only the original x portion (first C channels)
        x_mamba_t = x_mamba_t + concat_NR_t[:, :C, :]
        x_mamba = x_mamba_t.transpose(1, 2)                # (B, N, C)

        # Local-Global Feature Mixing (or bypass when --no-lgfm is set)
        if self.local_global_mixing is not None:
            x_mixed = self.local_global_mixing(x_mamba, H=H, W=W)
        else:
            x_mixed = x_mamba

        # Final LN + projection
        x_mixed = self.norm(x_mixed)
        x_mixed = self.proj(x_mixed)

        # Reshape back to (B, output_dim, H, W)
        out = x_mixed.transpose(1, 2).view(B, self.output_dim, H, W)
        return out


# Cross-Gated Attention Gate using Depthwise Convolutions
class CrossGatedAttentionGate(nn.Module):
    """
    A refined attention gate that applies cross-gating between g and x
    on a per-branch basis with 4-way directional scanning, then merges
    the results to produce an attention mask applied to x (the skip connection).
    """

    def __init__(
        self,
        F_g,
        F_l,
        F_int,
        num_branches=4,
        d_state=16,
        d_conv=4,
        expand=2,
        scan_mode: str = "four_way",
        use_final_bn: bool = True,
    ):
        super(CrossGatedAttentionGate, self).__init__()
        self.F_g = F_g
        self.F_l = F_l
        self.F_int = F_int
        self.num_branches = num_branches

        # MambaSSMLayer for gating signal g
        assert F_g % num_branches == 0, (
            f"F_g={F_g} must be divisible by num_branches={num_branches}"
        )
        self.mamba_g = nn.ModuleList([
            MambaSSMLayer(
                d_model=F_g // num_branches,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                direction=(i % 4) if scan_mode == "four_way" else 0,
            )
            for i in range(num_branches)
        ])

        # MambaSSMLayer for skip connection x
        assert F_l % num_branches == 0, (
            f"F_l={F_l} must be divisible by num_branches={num_branches}"
        )
        self.mamba_x = nn.ModuleList([
            MambaSSMLayer(
                d_model=F_l // num_branches,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                direction=(i % 4) if scan_mode == "four_way" else 0,
            )
            for i in range(num_branches)
        ])

        # Replace lightweight 1x1 conv with depthwise conv for each branch
        self.conv_g_dw = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    F_g // num_branches,
                    F_g // num_branches,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=F_g // num_branches,
                ),
                nn.BatchNorm2d(F_g // num_branches),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_branches)
        ])
        self.conv_x_dw = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    F_l // num_branches,
                    F_l // num_branches,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=F_l // num_branches,
                ),
                nn.BatchNorm2d(F_l // num_branches),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_branches)
        ])

        # Project cross_i => F_int using depthwise then pointwise conv => BN => ReLU
        self.branch_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    F_g // num_branches,
                    F_g // num_branches,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=F_g // num_branches,
                ),
                nn.BatchNorm2d(F_g // num_branches),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    F_g // num_branches, F_int, kernel_size=1, stride=1, padding=0
                ),
                nn.BatchNorm2d(F_int),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_branches)
        ])

        # Final combination: (B, num_branches*F_int, H, W) -> (B, F_int, H, W) -> (B, 1, H, W)
        final_norm = nn.BatchNorm2d(1) if use_final_bn else nn.Identity()
        self.combine = nn.Sequential(
            nn.Conv2d(F_int * num_branches, F_int, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            final_norm,
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        """
        g: (B, F_g, H, W)
        x: (B, F_l, H, W)
        returns: x * attention_mask
        """
        B, _, H, W = g.shape

        # Split gating signal and skip connection features into branches
        g_chunks = torch.chunk(g, self.num_branches, dim=1)
        x_chunks = torch.chunk(x, self.num_branches, dim=1)

        # For merging cross-gated branches.
        cross_branch_list = []

        for i in range(self.num_branches):
            # -- Gating branch --
            g_i = g_chunks[i]  # (B, F_g/num_branches, H, W)
            g_i_mamba = self.mamba_g[i](g_i, H=H, W=W)
            _, n_g, c_g = g_i_mamba.shape
            if n_g != H * W:
                raise ValueError(f"G branch mismatch: N={n_g}, but H*W={H}*{W}={H*W}")
            g_i_mamba = g_i_mamba.transpose(1, 2).view(B, c_g, H, W)

            # -- Skip branch --
            x_i = x_chunks[i]  # (B, F_l/num_branches, H, W)
            x_i_mamba = self.mamba_x[i](x_i, H=H, W=W)
            _, n_x, c_x = x_i_mamba.shape
            if n_x != H * W:
                raise ValueError(f"X branch mismatch: N={n_x}, but H*W={H}*{W}={H*W}")
            x_i_mamba = x_i_mamba.transpose(1, 2).view(B, c_x, H, W)

            # Cross gating using depthwise convs:
            g_i_conv = self.conv_g_dw[i](g_i_mamba)  # (B, c_g, H, W)
            x_i_conv = self.conv_x_dw[i](x_i_mamba)  # (B, c_x, H, W)
            cross_i = x_i_mamba * torch.sigmoid(g_i_conv) + g_i_mamba * torch.sigmoid(
                x_i_conv
            )

            # Project cross_i => F_int with branch_proj[i]
            cross_i_proj = self.branch_proj[i](cross_i)  # (B, F_int, H, W)
            cross_branch_list.append(cross_i_proj)

        # Combine across branches => (B, F_int * num_branches, H, W)
        cross_cat = torch.cat(cross_branch_list, dim=1)

        # Final attention mask
        psi = self.combine(cross_cat)  # (B, 1, H, W)

        # Apply attention mask to skip connection x
        return x * psi


class _ConvFallbackBlock(nn.Module):
    """Lightweight depthwise-separable convolutional replacement when --no-amf is active."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.last_branch_disagreement = None
        self.net = nn.Sequential(
            nn.Conv2d(
                input_dim,
                input_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=input_dim,
            ),
            nn.GELU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# MambaLiteUNet Model
class MambaLiteUNet(nn.Module):
    def __init__(
        self,
        num_classes=1,
        input_channels=3,
        c_list=[16, 32, 48, 64, 96, 128],
        use_amf: bool = True,
        use_lgfm: bool = True,
        use_cga: bool = True,
        scan_mode: str = "four_way",
        global_mode: str = "attn",
        cga_final_bn: bool = True,
        return_logits: bool = False,
        return_aux: bool = False,
    ):
        super(MambaLiteUNet, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.c_list = list(c_list)
        self.use_amf = bool(use_amf)
        self.use_lgfm = bool(use_lgfm)
        self.use_cga = bool(use_cga)
        self.scan_mode = scan_mode
        self.global_mode = global_mode
        self.return_logits = bool(return_logits)
        self.return_aux = bool(return_aux)
        self.last_aux: dict = {}

        def _make_amf(in_dim: int, out_dim: int) -> nn.Module:
            if not self.use_amf:
                return _ConvFallbackBlock(in_dim, out_dim)
            return AdaptiveMultiBranchMambaFeatureFusion(
                input_dim=in_dim,
                output_dim=out_dim,
                num_branches=4,
                d_state=16,
                d_conv=4,
                expand=2,
                use_lgfm=self.use_lgfm,
                scan_mode=self.scan_mode,
                global_mode=self.global_mode,
            )

        # Encoder layers
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], kernel_size=3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], kernel_size=3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], kernel_size=3, stride=1, padding=1),
        )
        self.encoder4 = _make_amf(c_list[2], c_list[3])
        self.encoder5 = _make_amf(c_list[3], c_list[4])
        self.encoder6 = _make_amf(c_list[4], c_list[5])

        # Decoder layers
        self.decoder1 = _make_amf(c_list[5], c_list[4])
        self.decoder2 = _make_amf(c_list[4], c_list[3])
        self.decoder3 = _make_amf(c_list[3], c_list[2])
        self.decoder4 = nn.Conv2d(
            c_list[2], c_list[1], kernel_size=3, stride=1, padding=1
        )
        self.decoder5 = nn.Conv2d(
            c_list[1], c_list[0], kernel_size=3, stride=1, padding=1
        )

        # Cross-Gated Attention Gates (with Cross-Gating)
        if self.use_cga:
            self.attention5 = CrossGatedAttentionGate(
                F_g=c_list[4],
                F_l=c_list[4],
                F_int=c_list[4] // 2,
                num_branches=4,
                d_state=16,
                d_conv=4,
                expand=2,
                scan_mode=self.scan_mode,
                use_final_bn=cga_final_bn,
            )
            self.attention4 = CrossGatedAttentionGate(
                F_g=c_list[3],
                F_l=c_list[3],
                F_int=c_list[3] // 2,
                num_branches=4,
                d_state=16,
                d_conv=4,
                expand=2,
                scan_mode=self.scan_mode,
                use_final_bn=cga_final_bn,
            )
            self.attention3 = CrossGatedAttentionGate(
                F_g=c_list[2],
                F_l=c_list[2],
                F_int=c_list[2] // 2,
                num_branches=4,
                d_state=16,
                d_conv=4,
                expand=2,
                scan_mode=self.scan_mode,
                use_final_bn=cga_final_bn,
            )
            self.attention2 = CrossGatedAttentionGate(
                F_g=c_list[1],
                F_l=c_list[1],
                F_int=c_list[1] // 2,
                num_branches=4,
                d_state=16,
                d_conv=4,
                expand=2,
                scan_mode=self.scan_mode,
                use_final_bn=cga_final_bn,
            )
        else:
            self.attention5 = None
            self.attention4 = None
            self.attention3 = None
            self.attention2 = None

        # Normalizations
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])

        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        # Final convolution
        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        # Initialize weights while preserving internal Mamba S4D/HiPPO initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if getattr(m, "_is_mamba_internal", False):
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _collect_branch_disagreement(self, H: int, W: int) -> Optional[torch.Tensor]:
        maps = []
        for block in (
            self.encoder4,
            self.encoder5,
            self.encoder6,
            self.decoder1,
            self.decoder2,
            self.decoder3,
        ):
            d_map = getattr(block, "last_branch_disagreement", None)
            if d_map is not None:
                maps.append(
                    F.interpolate(
                        d_map, size=(H, W), mode="bilinear", align_corners=False
                    )
                )
        if not maps:
            return None
        return torch.stack(maps, dim=0).mean(dim=0).clamp(0.0, 1.0)

    def forward(self, x):
        _, _, H_in, W_in = x.shape

        # Encoder
        out = self.ebn1(self.encoder1(x))
        out = F.gelu(F.max_pool2d(out, 2, 2))
        t1 = out

        out = self.ebn2(self.encoder2(out))
        out = F.gelu(F.max_pool2d(out, 2, 2))
        t2 = out

        out = self.ebn3(self.encoder3(out))
        out = F.gelu(F.max_pool2d(out, 2, 2))
        t3 = out

        out = self.ebn4(self.encoder4(out))
        out = F.gelu(F.max_pool2d(out, 2, 2))
        t4 = out

        out = self.ebn5(self.encoder5(out))
        out = F.gelu(F.max_pool2d(out, 2, 2))
        t5 = out

        # Bottleneck part
        out = self.encoder6(out)
        out = F.gelu(out)

        # Decoder with Cross-Gated Attention (or direct skip if --no-cga)
        out5 = self.decoder1(out)
        out5 = F.gelu(self.dbn1(out5))
        att_t5 = self.attention5(g=out5, x=t5) if self.attention5 is not None else t5
        out5 = out5 + att_t5

        out4 = self.decoder2(out5)
        out4 = F.gelu(self.dbn2(out4))
        out4 = F.interpolate(out4, scale_factor=2, mode='bilinear', align_corners=True)
        att_t4 = self.attention4(g=out4, x=t4) if self.attention4 is not None else t4
        out4 = out4 + att_t4

        out3 = self.decoder3(out4)
        out3 = F.gelu(self.dbn3(out3))
        out3 = F.interpolate(out3, scale_factor=2, mode='bilinear', align_corners=True)
        att_t3 = self.attention3(g=out3, x=t3) if self.attention3 is not None else t3
        out3 = out3 + att_t3

        out2 = self.dbn4(self.decoder4(out3))
        out2 = F.gelu(out2)
        out2 = F.interpolate(out2, scale_factor=2, mode='bilinear', align_corners=True)
        att_t2 = self.attention2(g=out2, x=t2) if self.attention2 is not None else t2
        out2 = out2 + att_t2

        out1 = self.dbn5(self.decoder5(out2))
        out1 = F.gelu(out1)
        out1 = F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=True)
        out1 = out1 + t1

        # Final prediction
        logits = F.interpolate(
            self.final(out1), size=(H_in, W_in), mode='bilinear', align_corners=True
        )
        probs = torch.sigmoid(logits)

        self.last_aux = {
            "logits": logits,
            "features": out1,
            "branch_disagreement": self._collect_branch_disagreement(H_in, W_in),
        }

        out_main = logits if self.return_logits else probs
        if self.return_aux:
            return out_main, self.last_aux
        return out_main
