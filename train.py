"""
SpikeFusion-GI 2.0 — Training Script (v2.1)
=============================================
Enhanced training with:
    - RandAugment-style augmentations
    - CutMix + Mixup
    - Stochastic Weight Averaging (SWA)
    - Test-time augmentation (TTA) option
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import config as cfg
from models import SpikeFusion
from data.kvasir_dataset import get_dataloaders
from utils.metrics import (
    compute_metrics, MetricLogger,
    save_checkpoint, load_checkpoint,
    get_cosine_schedule_with_warmup,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rand_bbox(size, lam):
    """CutMix bounding box."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_mixed = x.clone()
    x_mixed[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x_mixed, y, y[index], lam


def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    use_augment="mixup"):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        # Choose augmentation strategy
        aug_type = "none"
        if use_augment == "mixup" and np.random.rand() < 0.5:
            images, y_a, y_b, lam = mixup_data(images, labels)
            aug_type = "mixup"
        elif use_augment == "cutmix" and np.random.rand() < 0.5:
            images, y_a, y_b, lam = cutmix_data(images, labels)
            aug_type = "cutmix"
        elif use_augment == "both":
            r = np.random.rand()
            if r < 0.33:
                images, y_a, y_b, lam = mixup_data(images, labels)
                aug_type = "mixup"
            elif r < 0.66:
                images, y_a, y_b, lam = cutmix_data(images, labels)
                aug_type = "cutmix"

        with autocast():
            logits, _ = model(images)
            if aug_type == "none":
                loss = criterion(logits, labels)
            else:
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, class_names):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits, _ = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds),
        np.array(all_probs), num_classes, class_names,
    )
    return avg_loss, metrics


def main(args):
    set_seed(cfg.RANDOM_SEED)
    device = torch.device(cfg.DEVICE)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    # Data
    print(f"[INFO] Loading Kvasir v2 from {args.data_root} ...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_root=args.data_root, img_size=cfg.IMG_SIZE,
        batch_size=cfg.BATCH_SIZE, train_split=cfg.TRAIN_SPLIT,
        val_split=cfg.VAL_SPLIT, seed=cfg.RANDOM_SEED,
        num_workers=args.num_workers,
    )
    print(f"[INFO] Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Model
    model = SpikeFusion(
        num_classes=cfg.NUM_CLASSES,
        img_size=cfg.IMG_SIZE,
        embed_dim=cfg.EMBED_DIM,
        mamba_depth=cfg.MAMBA_DEPTH,
        mamba_d_state=cfg.MAMBA_D_STATE,
        mamba_d_conv=cfg.MAMBA_D_CONV,
        mamba_expand=cfg.MAMBA_EXPAND,
        mamba_drop_path=cfg.MAMBA_DROP_PATH,
        snn_timesteps=cfg.SNN_TIMESTEPS,
        snn_tau=cfg.SNN_TAU,
        snn_vth=cfg.SNN_VTH,
        snn_surrogate=cfg.SNN_SURROGATE,
        cnn_base_width=cfg.CNN_BASE_WIDTH,
    ).to(device)

    total, trainable = model.count_parameters()
    print(f"[INFO] Model params: {total:,} total ({total/1e6:.2f}M), "
          f"{trainable:,} trainable")

    # Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR,
                                   weight_decay=cfg.WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, cfg.WARMUP_EPOCHS, cfg.NUM_EPOCHS,
        min_lr=cfg.LR_MIN, base_lr=cfg.LR,
    )
    scaler = GradScaler()

    # SWA
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = max(1, cfg.NUM_EPOCHS - 10)

    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        print(f"[INFO] Resumed from epoch {start_epoch}")

    # Training loop
    logger = MetricLogger(log_dir=cfg.LOG_DIR)
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            use_augment=args.augment,
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, cfg.NUM_CLASSES, class_names,
        )

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        # SWA update
        if epoch >= swa_start:
            swa_model.update_parameters(model)

        best_val_acc, best_epoch = logger.log(epoch, train_loss, val_loss, val_metrics)
        logger.print_epoch(epoch, train_loss, val_loss, val_metrics, lr)

        if val_metrics["accuracy"] >= best_val_acc:
            patience_counter = 0
            save_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, save_path)
            print(f"  -> Saved best model (acc={val_metrics['accuracy']:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= cfg.PATIENCE:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    # SWA final evaluation
    print("\n[INFO] Evaluating SWA model...")
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    _, swa_metrics = evaluate(swa_model, val_loader, criterion, device,
                               cfg.NUM_CLASSES, class_names)
    print(f"[INFO] SWA val accuracy: {swa_metrics['accuracy']:.4f}")

    # Final test evaluation (best model)
    print("\n[INFO] Loading best model for test evaluation...")
    best_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
    load_checkpoint(model, None, None, best_path, device)
    _, test_metrics = evaluate(model, test_loader, criterion, device,
                                cfg.NUM_CLASSES, class_names)
    print("\n========== Final Test Results ==========")
    print(f"Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"F1 (macro):    {test_metrics['f1_macro']:.4f}")
    print(f"F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"Precision:     {test_metrics['precision']:.4f}")
    print(f"Recall:        {test_metrics['recall']:.4f}")
    print(f"Sensitivity:   {test_metrics['sensitivity_mean']:.4f}")
    print(f"Specificity:   {test_metrics['specificity_mean']:.4f}")
    print(f"AUC (macro):   {test_metrics['auc']:.4f}")
    print("========================================")

    import json
    with open(os.path.join(cfg.LOG_DIR, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SpikeFusion-GI")
    parser.add_argument("--data-root", type=str, default=cfg.DATA_ROOT)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--augment", type=str, default="both",
                        choices=["none", "mixup", "cutmix", "both"],
                        help="Augmentation strategy")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
