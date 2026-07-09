"""
SpikeFusion-GI 2.0 — XAI: Grad-CAM for CNN Branch
==================================================
Gradient-weighted Class Activation Mapping for the CNN branch.

Reference:
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization" (ICCV 2017)
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2


class GradCAM:
    """
    Grad-CAM for the CNN branch of SpikeFusion.
    """

    def __init__(self, model, target_layer_name="cnn_branch.stage6"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output[0] if isinstance(output, tuple) else output
            return None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0] if isinstance(grad_output, tuple) else grad_output
            return None

        # Navigate to target layer
        parts = self.target_layer_name.split(".")
        module = self.model
        for p in parts:
            module = getattr(module, p)

        self.hook_handles.append(module.register_forward_hook(forward_hook))
        self.hook_handles.append(module.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

    def generate(self, input_image, target_class=None):
        """
        Args:
            input_image: [1, 3, H, W] tensor
            target_class: int or None (uses predicted class)
        Returns:
            cam: [H, W] numpy array normalised to [0, 1]
        """
        self.model.eval()
        input_image.requires_grad = True

        logits, aux = self.model(input_image)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Grad-CAM weights
        grads = self.gradients[0]  # [C, H, W]
        acts = self.activations[0]  # [C, H, W]

        weights = grads.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        cam = (weights * acts).sum(dim=0)  # [H, W]
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize to input size
        cam = cam.detach().cpu().numpy()
        h, w = input_image.shape[2:]
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        return cam, target_class

    def overlay(self, image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay CAM on original image.

        Args:
            image: [H, W, 3] numpy array in [0, 255] or [0, 1]
            cam: [H, W] in [0, 1]
        Returns:
            overlay: [H, W, 3] uint8
        """
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        heatmap = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return overlay
