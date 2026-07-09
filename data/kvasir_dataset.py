"""
SpikeFusion-GI 2.0 — Kvasir v2 Dataset Loader
==============================================
Handles downloading (if needed), splitting, and augmentations.
"""

import os
import random
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class KvasirDataset(Dataset):
    """
    Kvasir v2 dataset with 8 classes of GI endoscopy images.

    Expected directory structure:
        data_root/
            dyed-lifted-polyps/
            dyed-resection-margins/
            esophagitis/
            normal-cecum/
            normal-pylorus/
            normal-z-line/
            polyps/
            ulcerative-colitis/
    """

    def __init__(self, data_root, transform=None, img_size=224):
        self.data_root = data_root
        self.transform = transform
        self.img_size = img_size

        self.classes = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(data_root, cls)
            paths = glob(os.path.join(cls_dir, "*.jpg")) + \
                    glob(os.path.join(cls_dir, "*.png")) + \
                    glob(os.path.join(cls_dir, "*.jpeg"))
            for p in paths:
                self.samples.append((p, self.class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {data_root}. "
                "Please download Kvasir v2 from https://datasets.simula.no/kvasir/"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(img_size=224, augment=True):
    """Return train/val transforms."""
    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                    saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def get_dataloaders(data_root, img_size=224, batch_size=32,
                    train_split=0.7, val_split=0.15, seed=42,
                    num_workers=4):
    """
    Returns train, val, test DataLoaders.
    """
    train_tf, val_tf = get_transforms(img_size, augment=True)

    full_dataset = KvasirDataset(data_root, transform=train_tf, img_size=img_size)
    n_total = len(full_dataset)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Val/test use non-augmented transforms
    val_test_base = KvasirDataset(data_root, transform=val_tf, img_size=img_size)
    val_ds.dataset = val_test_base
    test_ds.dataset = val_test_base

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, full_dataset.classes
