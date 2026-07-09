"""
SpikeFusion-GI 2.0 — CNN Branch with ECA Attention
====================================================
Lightweight RGB-CNN with Efficient Channel Attention (ECA).
ECA adds ~0.1% parameters but significantly boosts representational power.

Reference:
- Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional
  Neural Networks" (CVPR 2020)
"""

import math
import torch
import torch.nn as nn


class ECA(nn.Module):
    """
    Efficient Channel Attention.
    Uses 1D conv with adaptive kernel size instead of SE's MLP bottleneck.
    """

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            x: [B, C, H, W] with channel-wise attention applied
        """
        y = self.avg_pool(x)                # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class InvertedResidualECA(nn.Module):
    """
    MobileNet-v2 inverted residual + ECA attention.
    """

    def __init__(self, in_ch, out_ch, expansion=4, stride=1):
        super().__init__()
        import math
        self.use_res = stride == 1 and in_ch == out_ch
        hidden = int(round(in_ch * expansion))

        layers = []
        if expansion != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)
        self.eca = ECA(out_ch)

    def forward(self, x):
        out = self.conv(x)
        out = self.eca(out)
        if self.use_res:
            return x + out
        return out


class CNNBranch(nn.Module):
    """
    Lightweight CNN for colour + texture with ECA attention.

    Input:  [B, 3, 224, 224]
    Output: [B, out_channels, 7, 7]
    """

    def __init__(self, out_channels=48, base_width=24):
        super().__init__()
        self.out_channels = out_channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.SiLU(inplace=True),
        )

        self.stage1 = self._make_stage(base_width, base_width, 2, stride=1)
        self.stage2 = self._make_stage(base_width, base_width * 2, 2, stride=2)
        self.stage3 = self._make_stage(base_width * 2, base_width * 2, 2, stride=1)
        self.stage4 = self._make_stage(base_width * 2, base_width * 4, 2, stride=2)
        self.stage5 = self._make_stage(base_width * 4, out_channels, 2, stride=2)
        self.stage6 = self._make_stage(out_channels, out_channels, 2, stride=2)

        self._init_weights()

    def _make_stage(self, in_ch, out_ch, num_blocks, stride):
        layers = []
        layers.append(InvertedResidualECA(in_ch, out_ch, expansion=4, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(InvertedResidualECA(out_ch, out_ch, expansion=4, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feature_maps = []
        x = self.stem(x)           # [B, 24, 112, 112]
        feature_maps.append(x)
        x = self.stage1(x)         # [B, 24, 112, 112]
        feature_maps.append(x)
        x = self.stage2(x)         # [B, 48, 56, 56]
        feature_maps.append(x)
        x = self.stage3(x)         # [B, 48, 56, 56]
        feature_maps.append(x)
        x = self.stage4(x)         # [B, 96, 28, 28]
        feature_maps.append(x)
        x = self.stage5(x)         # [B, 48, 14, 14]
        feature_maps.append(x)
        x = self.stage6(x)         # [B, 48, 7, 7]
        feature_maps.append(x)
        return x, feature_maps
