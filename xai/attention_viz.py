"""
SpikeFusion-GI 2.0 — XAI: Mamba Attention / State Visualisation
=================================================================
Visualise the hidden states and token evolution through Mamba blocks.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_token_heatmap(tokens, save_path=None, title="Mamba Token States"):
    """
    Plot a heatmap of token representations.

    Args:
        tokens: [L, D] or [1, L, D] tensor
    """
    if tokens.dim() == 3:
        tokens = tokens.squeeze(0)
    tok = tokens.detach().cpu().numpy()  # [L, D]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(tok, aspect="auto", cmap="viridis")
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Token Index (spatial)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_mamba_state_evolution(states, save_path=None):
    """
    Plot the evolution of token norms across Mamba blocks.

    Args:
        states: list of [1, L, D] tensors (one per block)
    """
    norms = []
    for s in states:
        s = s.squeeze(0)
        norms.append(s.norm(dim=1).detach().cpu().numpy())  # [L]
    norms = np.stack(norms, axis=0)  # [depth, L]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(norms, aspect="auto", cmap="plasma")
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Mamba Block Depth")
    ax.set_title("Token Norm Evolution Across Mamba Blocks")
    plt.colorbar(im, ax=ax, label="L2 Norm")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_fusion_map(fused_map, save_path=None):
    """
    Visualise the fused feature map as averaged channels.

    Args:
        fused_map: [C, H, W] tensor
    """
    fm = fused_map.mean(dim=0).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(fm, cmap="inferno")
    ax.set_title("Fusion Feature Map (Channel Average)")
    ax.axis("off")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
