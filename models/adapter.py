# models/adapter.py

from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock1x1x1(nn.Module):
    """
    Simple residual block with 1x1x1 convs + GroupNorm + SiLU.
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=1)

        self.norm2 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return residual + h


class HunyuanToKVAEAdapter(nn.Module):
    """
    Tiny 3-block ResNet mapping Hunyuan 3D latents -> KVAE-3D latents.

    Input:  [B, 16, T', H', W']
    Output: [B, 16, T', H', W']
    """

    def __init__(self, channels: int = 16, num_blocks: int = 3, groups: int = 8):
        super().__init__()
        self.in_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResBlock1x1x1(channels=channels, groups=groups) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, z_h: torch.Tensor) -> torch.Tensor:
        # z_h: [B, C, T', H', W']
        h = self.in_proj(z_h)
        for block in self.blocks:
            h = block(h)
        z_k_hat = self.out_proj(h)
        return z_k_hat

