"""
SpikeFusion-GI 2.0 — Inference + XAI (v2.1)
=============================================
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import config as cfg
from models import SpikeFusion
from utils.metrics import load_checkpoint
from xai.explain import explain_single_image


def preprocess_image(image_path, img_size=224):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img.resize((img_size, img_size)))
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
    ])
    tensor = tf(img).unsqueeze(0)
    return tensor, img_np


def main(args):
    device = torch.device(cfg.DEVICE)

    model = SpikeFusion(
        num_classes=cfg.NUM_CLASSES, img_size=cfg.IMG_SIZE,
        embed_dim=cfg.EMBED_DIM, mamba_depth=cfg.MAMBA_DEPTH,
        mamba_d_state=cfg.MAMBA_D_STATE, mamba_d_conv=cfg.MAMBA_D_CONV,
        mamba_expand=cfg.MAMBA_EXPAND, mamba_drop_path=0.0,
        snn_timesteps=cfg.SNN_TIMESTEPS, snn_tau=cfg.SNN_TAU,
        snn_vth=cfg.SNN_VTH, snn_surrogate=cfg.SNN_SURROGATE,
        cnn_base_width=cfg.CNN_BASE_WIDTH,
    ).to(device)

    load_checkpoint(model, None, None, args.checkpoint, device)
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    image_tensor, image_np = preprocess_image(args.image, cfg.IMG_SIZE)
    class_names = cfg.CLASS_NAMES

    pred_class, pred_prob = explain_single_image(
        model, image_tensor, image_np, class_names,
        output_dir=args.output_dir, device=device,
    )
    print(f"\nResult: {class_names[pred_class]} ({pred_prob:.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=cfg.XAI_OUTPUT_DIR)
    args = parser.parse_args()
    main(args)
