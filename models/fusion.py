"""
SpikeFusion-GI 2.0 — Lightweight Fusion Module
===============================================
Fuses SNN edge features and CNN colour features into tokens.

Replaces expensive MultiheadAttention with lightweight channel-wise attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightFusion(nn.Module):
    """
    Fuses SNN and CNN features via:
        1. Spatial alignment
        2. Concatenation + 1x1 projection
        3. Channel attention (SE-style but lightweight)
        4. Patchify to tokens
    """

    def __init__(self, snn_channels, cnn_channels, embed_dim=128,
                 num_patches=49, img_size=224):
        super().__init__()
        self.embed_dim = embed_dim
        self.target_size = int(num_patches ** 0.5)

        # Fusion projection
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(snn_channels + cnn_channels, embed_dim,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

        # Lightweight channel attention (SE bottleneck: embed_dim -> 4 -> embed_dim)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(embed_dim, embed_dim // 4, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim // 4, embed_dim, bias=False),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, snn_feat, cnn_feat):
        """
        Args:
            snn_feat: [B, snn_ch, H1, W1]
            cnn_feat: [B, cnn_ch, H2, W2]
        Returns:
            tokens:   [B, num_patches, embed_dim]
            fused_map: [B, embed_dim, target_size, target_size]
        """
        # Align
        snn_feat = F.adaptive_avg_pool2d(snn_feat, (self.target_size, self.target_size))
        cnn_feat = F.adaptive_avg_pool2d(cnn_feat, (self.target_size, self.target_size))

        # Concat + project
        fused = torch.cat([snn_feat, cnn_feat], dim=1)
        fused = self.fusion_conv(fused)          # [B, embed_dim, 7, 7]

        # Channel attention
        attn = self.se(fused).view(-1, self.embed_dim, 1, 1)
        fused = fused * attn
        fused_map = fused

        # Patchify
        b, c, h, w = fused.shape
        tokens = fused.view(b, c, h * w).permute(0, 2, 1)  # [B, 49, 128]
        tokens = self.norm(tokens)
        return tokens, fused_map
