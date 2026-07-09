"""
SpikeFusion-GI 2.0 — Evaluation Script (v2.1)
===============================================
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import config as cfg
from models import SpikeFusion
from data.kvasir_dataset import get_dataloaders
from utils.metrics import compute_metrics, load_checkpoint


def main(args):
    device = torch.device(cfg.DEVICE)

    _, _, test_loader, class_names = get_dataloaders(
        data_root=args.data_root, img_size=cfg.IMG_SIZE,
        batch_size=cfg.BATCH_SIZE, train_split=cfg.TRAIN_SPLIT,
        val_split=cfg.VAL_SPLIT, seed=cfg.RANDOM_SEED,
        num_workers=args.num_workers,
    )

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
    print(f"[INFO] Loaded checkpoint from {args.checkpoint}")

    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            with autocast():
                logits, _ = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(test_loader.dataset)
    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds),
        np.array(all_probs), cfg.NUM_CLASSES, class_names,
    )
    metrics["test_loss"] = avg_loss

    print("\n========== Test Evaluation ==========")
    print(f"Loss:          {avg_loss:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"F1 (macro):    {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"Sensitivity:   {metrics['sensitivity_mean']:.4f}")
    print(f"Specificity:   {metrics['specificity_mean']:.4f}")
    print(f"AUC (macro):   {metrics['auc']:.4f}")
    print("\nPer-class Sensitivity:")
    for i, name in enumerate(class_names):
        print(f"  {name:30s}: {metrics['sensitivities'][i]:.4f}")
    print("\nPer-class Specificity:")
    for i, name in enumerate(class_names):
        print(f"  {name:30s}: {metrics['specificities'][i]:.4f}")
    print("=====================================")

    out_path = os.path.join(cfg.LOG_DIR, "eval_metrics.json")
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=cfg.DATA_ROOT)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
