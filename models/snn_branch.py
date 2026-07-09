"""
SpikeFusion-GI 2.0 — SNN Branch with Retinal Encoder
======================================================
SNN now processes rich retinal features (DoG, Gabor, colour opponency)
instead of raw Sobel edges.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .retinal_encoder import RetinalEncoder


def atan_surrogate(x, alpha=2.0):
    """Arctan surrogate gradient."""
    return alpha / 2.0 / (1.0 + (math.pi / 2.0 * alpha * x).pow(2))


def fast_sigmoid_surrogate(x, alpha=10.0):
    """Fast sigmoid surrogate gradient."""
    return alpha / (1.0 + alpha * x.abs()).pow(2)


SURROGATES = {
    "atan": atan_surrogate,
    "fast_sigmoid": fast_sigmoid_surrogate,
}


class LIFNeuron(nn.Module):
    """LIF neuron with learnable tau and vth."""

    def __init__(self, channels, tau_init=2.0, vth_init=1.0,
                 surrogate="atan", surrogate_alpha=2.0):
        super().__init__()
        self.channels = channels
        self.surrogate_fn = SURROGATES[surrogate]
        self.surrogate_alpha = surrogate_alpha
        self.tau = nn.Parameter(torch.ones(channels) * tau_init)
        self.vth = nn.Parameter(torch.ones(channels) * vth_init)

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        tau = self.tau.view(1, -1, 1, 1).clamp_min(1.01)
        vth = self.vth.view(1, -1, 1, 1).clamp_min(0.1)
        mem = mem + (x - mem) / tau
        spike = (mem >= vth).float()
        if self.training:
            sg = self.surrogate_fn(mem - vth, self.surrogate_alpha)
            spike = spike + sg - sg.detach()
        mem = mem - spike * vth
        return spike, mem


class SNNBranch(nn.Module):
    """
    SNN branch with retinal preprocessing.
    Architecture: Retinal Encoder → Conv3x3 → LIF → Conv3x3 → LIF
    """

    def __init__(self, retinal_channels=10, out_channels=32, timesteps=4,
                 tau=2.0, vth=1.0, surrogate="atan"):
        super().__init__()
        self.timesteps = timesteps
        self.out_channels = out_channels

        self.retinal_encoder = RetinalEncoder()

        self.conv1 = nn.Conv2d(retinal_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = LIFNeuron(out_channels, tau_init=tau, vth_init=vth,
                              surrogate=surrogate)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = LIFNeuron(out_channels, tau_init=tau, vth_init=vth,
                              surrogate=surrogate)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            spike_rate: [B, out_channels, H, W]
            retinal_map: [B, 10, H, W] for XAI
            spikes: [B, T, out_channels, H, W]
        """
        retinal_map = self.retinal_encoder(x)  # [B, 10, H, W]

        mem1 = None
        mem2 = None
        spikes = []

        for t in range(self.timesteps):
            c1 = self.bn1(self.conv1(retinal_map))
            s1, mem1 = self.lif1(c1, mem1)
            c2 = self.bn2(self.conv2(s1))
            s2, mem2 = self.lif2(c2, mem2)
            spikes.append(s2)

        spikes = torch.stack(spikes, dim=1)
        spike_rate = spikes.mean(dim=1)
        return spike_rate, retinal_map, spikes
