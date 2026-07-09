"""
SpikeFusion-GI 2.0 — Retinal Encoder
=====================================
Biologically-inspired preprocessing mimicking the human retina → LGN → V1 pathway.

Components:
    1. Center-Surround (DoG) — detects blob-like lesions / polyps at multiple scales
    2. Color Opponency — R-G (inflammation) and B-Y (necrosis/mucus) channels
    3. Orientation-Selective (Gabor) — V1 simple-cell edge detection at 4 angles
    4. Luminance — grayscale intensity
    5. Sobel Magnitude — overall edge strength

All filters are fixed (no learnable parameters) — purely biological prior.
Output channels: 10 (1 luminance + 2 DoG + 2 color + 4 Gabor + 1 Sobel)

References:
- Hubel & Wiesel, "Receptive fields of single neurones in the cat's striate cortex"
  (J. Physiology, 1959)
- Daugman, "Uncertainty relation for resolution in space, spatial frequency,
  and orientation optimized by two-dimensional visual cortical filters"
  (JOSA A, 1985)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RetinalEncoder(nn.Module):
    """
    Parameter-free retinal encoder inspired by early visual system.

    Input:  RGB [B, 3, H, W]
    Output: [B, 10, H, W] retinal feature maps
    """

    def __init__(self):
        super().__init__()

        # ---------- 1. Center-Surround (Difference of Gaussians) ----------
        # DoG approximates retinal ganglion cell / LGN receptive fields
        self.dog_scales = [1.0, 2.0]  # 2 spatial scales
        self.dog_kernels = nn.ParameterList()
        for sigma in self.dog_scales:
            k = self._make_dog_kernel(sigma, size=7)
            self.dog_kernels.append(nn.Parameter(k, requires_grad=False))

        # ---------- 2. Color Opponency ----------
        # R-G and B-Y opponent channels (like parvocellular pathway)
        rg_kernel = torch.tensor([
            [1.0, -1.0, 0.0],   # R - G
        ], dtype=torch.float32).view(1, 3, 1, 1)
        by_kernel = torch.tensor([
            [0.0, -1.0, 1.0],   # B - Y (approx)
        ], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("rg_kernel", rg_kernel)
        self.register_buffer("by_kernel", by_kernel)

        # ---------- 3. Orientation-Selective Gabor Filters ----------
        # 4 orientations mimicking V1 simple cells
        self.gabor_orientations = [0, 45, 90, 135]
        self.gabor_kernels = nn.ParameterList()
        for theta_deg in self.gabor_orientations:
            k = self._make_gabor_kernel(theta_deg, sigma=2.0, gamma=1.0,
                                          size=7, wavelength=4.0)
            self.gabor_kernels.append(nn.Parameter(k, requires_grad=False))

        # ---------- 4. Sobel Edge Magnitude ----------
        sobel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # Luminance weights (ITU-R BT.601)
        self.register_buffer("luma_weights",
                             torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def _make_dog_kernel(self, sigma, size=7):
        """Difference of Gaussians kernel (on-center / off-center)."""
        x = torch.arange(size, dtype=torch.float32) - size // 2
        xx = x.view(1, -1).expand(size, size)
        yy = x.view(-1, 1).expand(size, size)
        r2 = xx.pow(2) + yy.pow(2)

        sigma1 = sigma
        sigma2 = sigma * 1.6
        g1 = torch.exp(-r2 / (2 * sigma1 ** 2))
        g2 = torch.exp(-r2 / (2 * sigma2 ** 2))
        dog = g1 / g1.sum() - g2 / g2.sum()
        return dog.view(1, 1, size, size)

    def _make_gabor_kernel(self, theta_deg, sigma, gamma, size, wavelength):
        """2D Gabor filter kernel."""
        theta = math.radians(theta_deg)
        x = torch.arange(size, dtype=torch.float32) - size // 2
        xx = x.view(1, -1).expand(size, size)
        yy = x.view(-1, 1).expand(size, size)

        x_theta = xx * math.cos(theta) + yy * math.sin(theta)
        y_theta = -xx * math.sin(theta) + yy * math.cos(theta)

        gb = torch.exp(-(x_theta.pow(2) + gamma ** 2 * y_theta.pow(2))
                       / (2 * sigma ** 2))
        gb = gb * torch.cos(2 * math.pi * x_theta / wavelength)
        gb = gb - gb.mean()
        return gb.view(1, 1, size, size)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            retinal: [B, 10, H, W] where channels are:
                0: Luminance
                1-2: DoG scale 1 (on-center), scale 2 (off-center)
                3-4: R-G opponency, B-Y opponency
                5-8: Gabor 0°, 45°, 90°, 135°
                9: Sobel edge magnitude
        """
        feats = []

        # 0. Luminance
        luma = (x * self.luma_weights).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        feats.append(luma)

        # 1-2. DoG at multiple scales
        for k in self.dog_kernels:
            dog = F.conv2d(luma, k, padding=k.shape[-1] // 2)
            feats.append(dog)

        # 3-4. Color opponency
        rg = F.conv2d(x, self.rg_kernel, padding=0)
        by = F.conv2d(x, self.by_kernel, padding=0)
        feats.append(rg)
        feats.append(by)

        # 5-8. Gabor orientation filters
        for k in self.gabor_kernels:
            gab = F.conv2d(luma, k, padding=k.shape[-1] // 2)
            feats.append(gab)

        # 9. Sobel magnitude
        gx = F.conv2d(luma, self.sobel_x, padding=1)
        gy = F.conv2d(luma, self.sobel_y, padding=1)
        sobel = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)
        feats.append(sobel)

        retinal = torch.cat(feats, dim=1)  # [B, 10, H, W]
        # Normalise per-channel to [0, 1] using tanh for stability
        retinal = torch.tanh(retinal)
        return retinal
