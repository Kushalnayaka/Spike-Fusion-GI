"""
SpikeFusion-GI 2.0 — SpikeCAM (SNN Class Activation Mapping)
=============================================================
Generates attention maps from the SNN branch by backpropagating
class gradients through the spiking layers (using surrogate gradients).

Inspired by Grad-CAM but adapted for spiking neural networks:
- Uses spike-rate maps as spatial features
- Weights channels by their gradient contribution to the target class
- Produces a spatial heatmap showing where SNN spikes drive the decision
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2


class SpikeCAM:
    """
    SpikeCAM for the SNN branch.

    Generates a class-discriminative heatmap by:
        1. Forward pass to get spike-rate features and logits
        2. Backpropagate the target class score to the spike-rate tensor
        3. Global-average-pool the gradients to get channel weights
        4. Weighted combination of spike-rate channels = heatmap
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            # output is (spike_rate, retinal_map, spikes)
            self.activations = output[0]
            return None

        def backward_hook(module, grad_input, grad_output):
            # grad_output is tuple of grads w.r.t. outputs
            self.gradients = grad_output[0]
            return None

        self.hook_handles.append(
            self.model.snn_branch.register_forward_hook(forward_hook)
        )
        self.hook_handles.append(
            self.model.snn_branch.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

    def generate(self, input_image, target_class=None):
        """
        Args:
            input_image: [1, 3, H, W]
            target_class: int or None
        Returns:
            cam: [H, W] numpy array in [0, 1]
        """
        self.model.eval()
        input_image.requires_grad = True

        logits, _ = self.model(input_image)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Spike-rate features: [C, H, W]
        grads = self.gradients[0]      # [C, H, W]
        acts = self.activations[0]     # [C, H, W]

        # Channel weights: global average of gradients
        weights = grads.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]

        # Weighted sum of spike-rate channels
        cam = (weights * acts).sum(dim=0)  # [H, W]
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = cam.detach().cpu().numpy()
        h, w = input_image.shape[2:]
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        return cam, target_class

    def overlay(self, image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """Overlay CAM on original image."""
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        heatmap = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return overlay
