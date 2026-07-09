"""
SpikeFusion-GI 2.0 — XAI: Unified Explainability Pipeline
==========================================================
Upgraded with SpikeCAM (SNN attention maps) + retinal visualisations.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .gradcam import GradCAM
from .spikecam import SpikeCAM
from .spike_viz import (
    plot_spike_raster, plot_spike_rate_map,
    plot_retinal_channels,
)
from .attention_viz import (
    plot_token_heatmap, plot_mamba_state_evolution, plot_fusion_map,
)


def explain_single_image(model, image_tensor, image_np, class_names,
                          output_dir="./xai_outputs", device="cpu"):
    """Generate comprehensive XAI report for one image."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    image_tensor = image_tensor.to(device)

    # Forward pass (no grad for XAI collection)
    with torch.no_grad():
        logits, aux = model(image_tensor)
    probs = torch.softmax(logits, dim=1)
    pred_class = logits.argmax(dim=1).item()
    pred_prob = probs[0, pred_class].item()

    # 1. Grad-CAM (CNN branch)
    gradcam = GradCAM(model, target_layer_name="cnn_branch.stage6")
    cam_cnn, _ = gradcam.generate(image_tensor, target_class=pred_class)
    overlay_cnn = gradcam.overlay(image_np, cam_cnn)
    gradcam.remove_hooks()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np.astype(np.uint8))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(cam_cnn, cmap="jet")
    axes[1].set_title("Grad-CAM (CNN)")
    axes[1].axis("off")
    axes[2].imshow(overlay_cnn)
    axes[2].set_title(f"CNN Overlay: {class_names[pred_class]}")
    axes[2].axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "gradcam.png"), dpi=150)
    plt.close(fig)

    # 2. SpikeCAM (SNN branch)
    spikecam = SpikeCAM(model)
    cam_snn, _ = spikecam.generate(image_tensor, target_class=pred_class)
    overlay_snn = spikecam.overlay(image_np, cam_snn)
    spikecam.remove_hooks()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np.astype(np.uint8))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(cam_snn, cmap="hot")
    axes[1].set_title("SpikeCAM (SNN)")
    axes[1].axis("off")
    axes[2].imshow(overlay_snn)
    axes[2].set_title(f"SNN Overlay: {class_names[pred_class]}")
    axes[2].axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spikecam.png"), dpi=150)
    plt.close(fig)

    # 3. SNN / Retinal visualisations
    plot_retinal_channels(aux["retinal_map"][0],
                          save_path=os.path.join(output_dir, "retinal_channels.png"))
    plot_spike_raster(aux["snn_spikes"][0],
                       save_path=os.path.join(output_dir, "spike_raster.png"))
    plot_spike_rate_map(aux["snn_spikes"][0],
                        save_path=os.path.join(output_dir, "spike_rate.png"))

    # 4. Mamba visualisations
    plot_token_heatmap(aux["fused_tokens"][0],
                       save_path=os.path.join(output_dir, "tokens_input.png"),
                       title="Fused Tokens (Input to Mamba)")
    plot_mamba_state_evolution(aux["mamba_states"],
                               save_path=os.path.join(output_dir, "mamba_evolution.png"))
    plot_fusion_map(aux["fused_map"][0],
                    save_path=os.path.join(output_dir, "fusion_map.png"))

    # 5. Prediction confidence bar
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(class_names))
    scores = probs[0].cpu().numpy()
    colors = ["red" if i == pred_class else "skyblue" for i in range(len(class_names))]
    ax.barh(y_pos, scores, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Probability")
    ax.set_title("Prediction Confidence")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "prediction_bar.png"), dpi=150)
    plt.close(fig)

    print(f"[XAI] Saved visualisations to {output_dir}")
    print(f"[XAI] Predicted: {class_names[pred_class]} ({pred_prob:.2%})")
    return pred_class, pred_prob
