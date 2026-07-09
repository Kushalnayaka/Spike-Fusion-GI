"""
SpikeFusion-GI 2.0 — XAI: SNN & Retinal Visualisations
=======================================================
Spike raster, firing rate, and retinal channel visualisations.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_spike_raster(spikes, save_path=None, max_neurons=64):
    """Plot spike raster for a single sample."""
    if spikes.dim() == 4:
        t, c, h, w = spikes.shape
        spikes = spikes.reshape(t, -1)
    spikes = spikes.detach().cpu().numpy()
    t, n = spikes.shape
    n = min(n, max_neurons)
    spikes = spikes[:, :n]

    fig, ax = plt.subplots(figsize=(10, 6))
    for neuron in range(n):
        times = np.where(spikes[:, neuron] > 0)[0]
        ax.scatter(times, [neuron] * len(times), s=2, c="black")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Neuron Index")
    ax.set_title("SNN Spike Raster")
    ax.set_xlim(0, t)
    ax.set_ylim(-1, n)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_spike_rate_map(spikes, save_path=None):
    """Average firing rate as spatial heatmap."""
    rate = spikes.mean(dim=0).mean(dim=0)
    rate = rate.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(rate, cmap="hot", interpolation="nearest")
    ax.set_title("SNN Average Firing Rate")
    plt.colorbar(im, ax=ax, label="Firing Rate")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_retinal_channels(retinal_map, save_path=None):
    """
    Visualise all 10 retinal encoder channels.

    Args:
        retinal_map: [10, H, W] tensor
    """
    if retinal_map.dim() == 4:
        retinal_map = retinal_map[0]
    rm = retinal_map.detach().cpu().numpy()
    c, h, w = rm.shape

    channel_names = [
        "Luminance",
        "DoG On-Center (s=1)",
        "DoG Off-Center (s=2)",
        "R-G Opponency",
        "B-Y Opponency",
        "Gabor 0°",
        "Gabor 45°",
        "Gabor 90°",
        "Gabor 135°",
        "Sobel Magnitude",
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i in range(c):
        im = axes[i].imshow(rm[i], cmap="viridis", aspect="auto")
        axes[i].set_title(channel_names[i])
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    plt.suptitle("Retinal Encoder Output Channels", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
