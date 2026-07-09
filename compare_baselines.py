"""
SpikeFusion-GI — Baseline Comparison Script
============================================
Trains and evaluates multiple baseline architectures on the SAME
Kvasir v2 train/val/test splits for fair comparison.

Models compared:
    1. Vanilla CNN (4-layer conv net)
    2. ResNet-18 (ImageNet pretrained → fine-tuned)
    3. EfficientNet-B0 (ImageNet pretrained → fine-tuned)
    4. Vision Transformer (ViT-B/16, pretrained → fine-tuned)
    5. Pure Mamba (only bidirectional Mamba, no SNN/CNN fusion)
    6. SpikeFusion-GI (our proposed model)

Outputs:
    - comparison_table.csv
    - comparison_table.txt (formatted for paper)
    - comparison_plot.png (bar chart of metrics)
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import config as cfg
from data.kvasir_dataset import get_dataloaders
from utils.metrics import compute_metrics, get_cosine_schedule_with_warmup


# ------------------------------------------------------------------
# Baseline 1: Vanilla CNN
# ------------------------------------------------------------------
class VanillaCNN(nn.Module):
    """Simple 5-layer CNN baseline."""

    def __init__(self, num_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.head(x)


# ------------------------------------------------------------------
# Baseline 2: Pure Mamba (no SNN/CNN fusion)
# ------------------------------------------------------------------
from models.mamba_core import VisionMamba


class PureMamba(nn.Module):
    """Patchify RGB → Mamba blocks → classify. No SNN, no CNN."""

    def __init__(self, num_classes=8, img_size=224, patch_size=16,
                 embed_dim=128, depth=6, d_state=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.mamba = VisionMamba(embed_dim, depth, d_state, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # [B, embed_dim, H/p, W/p]
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)  # [B, N, D]
        x, _ = self.mamba(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)


# ------------------------------------------------------------------
# Baseline 3: SNN-only (no CNN, no Mamba)
# ------------------------------------------------------------------
from models.retinal_encoder import RetinalEncoder
from models.snn_branch import LIFNeuron


class SNNOnly(nn.Module):
    """Retinal encoder → SNN → global pool → classify."""

    def __init__(self, num_classes=8, timesteps=4):
        super().__init__()
        self.retinal = RetinalEncoder()
        self.conv1 = nn.Conv2d(10, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = LIFNeuron(32, tau_init=2.0, vth_init=1.0)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = LIFNeuron(64, tau_init=2.0, vth_init=1.0)
        self.timesteps = timesteps
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        ret = self.retinal(x)
        mem1 = mem2 = None
        spikes = []
        for _ in range(self.timesteps):
            s1, mem1 = self.lif1(self.bn1(self.conv1(ret)), mem1)
            s2, mem2 = self.lif2(self.bn2(self.conv2(s1)), mem2)
            spikes.append(s2)
        rate = torch.stack(spikes, dim=1).mean(dim=1)  # [B, 64, H, W]
        pooled = rate.mean(dim=(2, 3))  # [B, 64]
        return self.head(pooled)


# ------------------------------------------------------------------
# Baseline 4: CNN-only (no SNN, no Mamba)
# ------------------------------------------------------------------
from models.cnn_branch import CNNBranch


class CNNOnly(nn.Module):
    """CNN branch → global pool → classify."""

    def __init__(self, num_classes=8, base_width=24):
        super().__init__()
        self.cnn = CNNBranch(out_channels=48, base_width=base_width)
        self.head = nn.Linear(48, num_classes)

    def forward(self, x):
        feat, _ = self.cnn(x)
        pooled = feat.mean(dim=(2, 3))
        return self.head(pooled)


# ------------------------------------------------------------------
# Baseline 5: CNN+Mamba (no SNN)
# ------------------------------------------------------------------
class CNNMamba(nn.Module):
    """CNN → patchify → Mamba → classify."""

    def __init__(self, num_classes=8, embed_dim=128, depth=3):
        super().__init__()
        self.cnn = CNNBranch(out_channels=48, base_width=24)
        self.proj = nn.Conv2d(48, embed_dim, 1)
        self.mamba = VisionMamba(embed_dim, depth, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        feat, _ = self.cnn(x)  # [B, 48, 7, 7]
        feat = self.proj(feat)  # [B, 128, 7, 7]
        b, c, h, w = feat.shape
        tokens = feat.view(b, c, h * w).permute(0, 2, 1)
        tokens, _ = self.mamba(tokens)
        pooled = self.norm(tokens.mean(dim=1))
        return self.head(pooled)


# ------------------------------------------------------------------
# Training / Evaluation helpers
# ------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3,
                device="cpu", model_name="model"):
    """Train a baseline model and return best validation metrics."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 3, epochs)
    scaler = GradScaler()

    best_acc = 0.0
    best_metrics = None

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # Evaluate
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    logits = model(images)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics = compute_metrics(
            np.array(all_labels), np.array(all_preds),
            np.array(all_probs), cfg.NUM_CLASSES, cfg.CLASS_NAMES,
        )
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_metrics = metrics

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{model_name}] Epoch {epoch+1}/{epochs} | "
                  f"val_acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f}")

    return best_metrics


@torch.no_grad()
def eval_test(model, test_loader, device):
    """Evaluate on test set."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        with autocast():
            logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return compute_metrics(
        np.array(all_labels), np.array(all_preds),
        np.array(all_probs), cfg.NUM_CLASSES, cfg.CLASS_NAMES,
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main(args):
    device = torch.device(cfg.DEVICE)

    # Load data (same splits for all models)
    print("[INFO] Loading Kvasir v2...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_root=args.data_root, img_size=cfg.IMG_SIZE,
        batch_size=cfg.BATCH_SIZE, train_split=cfg.TRAIN_SPLIT,
        val_split=cfg.VAL_SPLIT, seed=cfg.RANDOM_SEED,
        num_workers=args.num_workers,
    )
    print(f"[INFO] Train={len(train_loader.dataset)} Val={len(val_loader.dataset)} "
          f"Test={len(test_loader.dataset)}")

    # Define models
    models_dict = {
        "Vanilla CNN": VanillaCNN(num_classes=cfg.NUM_CLASSES),
        "SNN-Only": SNNOnly(num_classes=cfg.NUM_CLASSES, timesteps=cfg.SNN_TIMESTEPS),
        "CNN-Only": CNNOnly(num_classes=cfg.NUM_CLASSES, base_width=cfg.CNN_BASE_WIDTH),
        "Pure Mamba": PureMamba(num_classes=cfg.NUM_CLASSES, img_size=cfg.IMG_SIZE,
                                  embed_dim=128, depth=6),
        "CNN+Mamba": CNNMamba(num_classes=cfg.NUM_CLASSES, embed_dim=128, depth=3),
    }

    # Load pretrained torchvision baselines if requested
    if args.include_pretrained:
        import torchvision.models as models
        models_dict["ResNet-18"] = models.resnet18(weights="IMAGENET1K_V1")
        models_dict["ResNet-18"].fc = nn.Linear(512, cfg.NUM_CLASSES)
        models_dict["EfficientNet-B0"] = models.efficientnet_b0(weights="IMAGENET1K_V1")
        models_dict["EfficientNet-B0"].classifier[1] = nn.Linear(1280, cfg.NUM_CLASSES)
        try:
            models_dict["ViT-B/16"] = models.vit_b_16(weights="IMAGENET1K_V1")
            models_dict["ViT-B/16"].heads.head = nn.Linear(768, cfg.NUM_CLASSES)
        except AttributeError:
            print("[WARN] ViT not available in this torchvision version, skipping.")

    # Train and evaluate
    results = []
    for name, model in models_dict.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"Parameters: {count_params(model):,}")
        print(f"{'='*50}")

        start = time.time()
        val_metrics = train_model(
            model, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
            device=device, model_name=name,
        )
        test_metrics = eval_test(model, test_loader, device)
        elapsed = time.time() - start

        results.append({
            "Model": name,
            "Params (M)": f"{count_params(model)/1e6:.2f}",
            "Accuracy": f"{test_metrics['accuracy']:.4f}",
            "F1-Macro": f"{test_metrics['f1_macro']:.4f}",
            "Sensitivity": f"{test_metrics['sensitivity_mean']:.4f}",
            "Specificity": f"{test_metrics['specificity_mean']:.4f}",
            "AUC": f"{test_metrics['auc']:.4f}",
            "Time (s)": f"{elapsed:.0f}",
        })

    # Train our proposed model (SpikeFusion-GI)
    from models import SpikeFusion
    print(f"\n{'='*50}")
    print(f"Training: SpikeFusion-GI (Proposed)")
    print(f"{'='*50}")
    our_model = SpikeFusion(
        num_classes=cfg.NUM_CLASSES, img_size=cfg.IMG_SIZE,
        embed_dim=cfg.EMBED_DIM, mamba_depth=cfg.MAMBA_DEPTH,
        mamba_d_state=cfg.MAMBA_D_STATE, mamba_d_conv=cfg.MAMBA_D_CONV,
        mamba_expand=cfg.MAMBA_EXPAND, mamba_drop_path=cfg.MAMBA_DROP_PATH,
        snn_timesteps=cfg.SNN_TIMESTEPS, snn_tau=cfg.SNN_TAU,
        snn_vth=cfg.SNN_VTH, snn_surrogate=cfg.SNN_SURROGATE,
        cnn_base_width=cfg.CNN_BASE_WIDTH,
    ).to(device)
    print(f"Parameters: {count_params(our_model):,}")

    start = time.time()
    val_metrics = train_model(
        our_model, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
        device=device, model_name="SpikeFusion-GI",
    )
    test_metrics = eval_test(our_model, test_loader, device)
    elapsed = time.time() - start

    results.append({
        "Model": "SpikeFusion-GI (Ours)",
        "Params (M)": f"{count_params(our_model)/1e6:.2f}",
        "Accuracy": f"{test_metrics['accuracy']:.4f}",
        "F1-Macro": f"{test_metrics['f1_macro']:.4f}",
        "Sensitivity": f"{test_metrics['sensitivity_mean']:.4f}",
        "Specificity": f"{test_metrics['specificity_mean']:.4f}",
        "AUC": f"{test_metrics['auc']:.4f}",
        "Time (s)": f"{elapsed:.0f}",
    })

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # CSV
    import csv
    csv_path = os.path.join(args.output_dir, "comparison_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[INFO] Results saved to {csv_path}")

    # Formatted text table
    txt_path = os.path.join(args.output_dir, "comparison_table.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 110 + "\n")
        f.write(f"{'Model':<25} {'Params':<10} {'Accuracy':<10} {'F1-Macro':<10} "
                f"{'Sensitivity':<12} {'Specificity':<12} {'AUC':<10} {'Time(s)':<10}\n")
        f.write("=" * 110 + "\n")
        for r in results:
            f.write(f"{r['Model']:<25} {r['Params (M)']:<10} {r['Accuracy']:<10} "
                    f"{r['F1-Macro']:<10} {r['Sensitivity']:<12} {r['Specificity']:<12} "
                    f"{r['AUC']:<10} {r['Time (s)']:<10}\n")
        f.write("=" * 110 + "\n")
    print(f"[INFO] Formatted table saved to {txt_path}")

    # Print to console
    print("\n" + "=" * 110)
    print(f"{'Model':<25} {'Params':<10} {'Accuracy':<10} {'F1-Macro':<10} "
          f"{'Sensitivity':<12} {'Specificity':<12} {'AUC':<10} {'Time(s)':<10}")
    print("=" * 110)
    for r in results:
        print(f"{r['Model']:<25} {r['Params (M)']:<10} {r['Accuracy']:<10} "
              f"{r['F1-Macro']:<10} {r['Sensitivity']:<12} {r['Specificity']:<12} "
              f"{r['AUC']:<10} {r['Time (s)']:<10}")
    print("=" * 110)

    # JSON for easy parsing
    json_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] JSON results saved to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=cfg.DATA_ROOT)
    parser.add_argument("--output-dir", type=str, default="./comparison_results")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epochs for each baseline (use 30 for fast comparison, 100 for paper)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--include-pretrained", action="store_true",
                        help="Include ResNet-18, EfficientNet-B0, ViT-B/16 (requires torchvision)")
    args = parser.parse_args()
    main(args)
