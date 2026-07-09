"""
SpikeFusion-GI 2.0 — Metrics & Utilities
========================================
Classification metrics, logging helpers, and training utilities.
"""

import os
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, roc_auc_score,
)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_prob, num_classes, class_names):
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: np.array of shape [N]
        y_pred: np.array of shape [N]
        y_prob: np.array of shape [N, num_classes]
    Returns:
        dict of metrics
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Sensitivity = recall per class
    # Specificity = TN / (TN + FP) per class
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    sensitivities = []
    specificities = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivities.append(tp / (tp + fn + 1e-8))
        specificities.append(tn / (tn + fp + 1e-8))

    # AUC (one-vs-rest)
    try:
        auc = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro"
        )
    except ValueError:
        auc = 0.0

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision": precision,
        "recall": recall,
        "sensitivity_mean": np.mean(sensitivities),
        "specificity_mean": np.mean(specificities),
        "sensitivities": sensitivities,
        "specificities": specificities,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
    }


# ------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------
class MetricLogger:
    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history = defaultdict(list)
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def log(self, epoch, train_loss, val_loss, metrics):
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": metrics["accuracy"],
            "val_f1_macro": metrics["f1_macro"],
            "val_auc": metrics["auc"],
        }
        for k, v in entry.items():
            self.history[k].append(v)

        if metrics["accuracy"] > self.best_val_acc:
            self.best_val_acc = metrics["accuracy"]
            self.best_epoch = epoch

        # Save history
        with open(os.path.join(self.log_dir, "history.json"), "w") as f:
            json.dump(dict(self.history), f, indent=2)

        return self.best_val_acc, self.best_epoch

    def print_epoch(self, epoch, train_loss, val_loss, metrics, lr):
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} "
            f"auc={metrics['auc']:.4f} | lr={lr:.2e}"
        )


# ------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0)


# ------------------------------------------------------------------
# LR Scheduler: cosine with warmup
# ------------------------------------------------------------------
def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs,
                                    min_lr=1e-6, base_lr=1e-3):
    """Simple cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr_ratio = min_lr / base_lr
        return lr_ratio + 0.5 * (1 - lr_ratio) * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------
# Mixup / CutMix (optional augmentation)
# ------------------------------------------------------------------
def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
