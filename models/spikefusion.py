"""
SpikeFusion-GI 2.0 — Main Model Assembly (v2.1)
================================================
Upgraded architecture:
    - Retinal Encoder (DoG, Gabor, color opponency) → SNN
    - CNN with ECA attention
    - Lightweight SE fusion
    - Bidirectional Vision Mamba with stochastic depth

Target: ~0.9M parameters, all task-relevant.
"""

import torch
import torch.nn as nn

from .snn_branch import SNNBranch
from .cnn_branch import CNNBranch
from .fusion import LightweightFusion
from .mamba_core import VisionMamba


class SpikeFusion(nn.Module):
    def __init__(
        self,
        num_classes=8,
        img_size=224,
        embed_dim=128,
        mamba_depth=3,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_drop_path=0.1,
        snn_timesteps=4,
        snn_tau=2.0,
        snn_vth=1.0,
        snn_surrogate="atan",
        cnn_base_width=24,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim

        # ---- SNN Branch (with retinal encoder) -----------------------
        self.snn_branch = SNNBranch(
            retinal_channels=10,
            out_channels=32,
            timesteps=snn_timesteps,
            tau=snn_tau,
            vth=snn_vth,
            surrogate=snn_surrogate,
        )

        # ---- CNN Branch (with ECA) -----------------------------------
        self.cnn_branch = CNNBranch(
            out_channels=48,
            base_width=cnn_base_width,
        )

        # ---- Fusion --------------------------------------------------
        self.fusion = LightweightFusion(
            snn_channels=32,
            cnn_channels=48,
            embed_dim=embed_dim,
            num_patches=49,
            img_size=img_size,
        )

        # ---- Bidirectional Vision Mamba ------------------------------
        self.mamba = VisionMamba(
            embed_dim=embed_dim,
            depth=mamba_depth,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            drop_path_rate=mamba_drop_path,
        )

        # ---- Classification head -------------------------------------
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224]
        Returns:
            logits: [B, num_classes]
            aux: dict for XAI
        """
        # 1. SNN branch
        snn_feat, retinal_map, snn_spikes = self.snn_branch(x)

        # 2. CNN branch
        cnn_feat, cnn_feature_maps = self.cnn_branch(x)

        # 3. Fusion
        tokens, fused_map = self.fusion(snn_feat, cnn_feat)

        # 4. Mamba
        mamba_out, mamba_states = self.mamba(tokens)

        # 5. Pool + classify
        pooled = mamba_out.mean(dim=1)
        pooled = self.norm(pooled)
        logits = self.head(pooled)

        aux = {
            "cnn_features": cnn_feat,
            "cnn_feature_maps": cnn_feature_maps,
            "snn_spikes": snn_spikes,
            "retinal_map": retinal_map,
            "mamba_states": mamba_states,
            "fused_tokens": tokens,
            "fused_map": fused_map,
            "pooled": pooled,
        }
        return logits, aux

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


if __name__ == "__main__":
    model = SpikeFusion(num_classes=8)
    total, trainable = model.count_parameters()
    print(f"Total params: {total:,}  ({total / 1e6:.2f}M)")
    print(f"Trainable:    {trainable:,}")

    x = torch.randn(2, 3, 224, 224)
    logits, aux = model(x)
    print(f"Logits shape: {logits.shape}")
    for k, v in aux.items():
        if isinstance(v, list):
            print(f"  {k}: list of {len(v)} tensors, first shape {v[0].shape}")
        else:
            print(f"  {k}: {v.shape}")
