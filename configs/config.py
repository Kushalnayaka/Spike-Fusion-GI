"""
SpikeFusion-GI 2.0 — Configuration (v2.1)
==========================================
"""

import torch

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
IMG_SIZE = 224
NUM_CLASSES = 8
CLASS_NAMES = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis",
]

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 120
LR = 8e-4
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1

WARMUP_EPOCHS = 5
LR_MIN = 1e-6
PATIENCE = 20

# ------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------
EMBED_DIM = 128
MAMBA_DEPTH = 3
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2
MAMBA_DROP_PATH = 0.1

SNN_TIMESTEPS = 4
SNN_TAU = 2.0
SNN_VTH = 1.0
SNN_SURROGATE = "atan"

CNN_BASE_WIDTH = 24

# ------------------------------------------------------------------
# Device & Paths
# ------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATA_ROOT = "./data/kvasir_v2"
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
XAI_OUTPUT_DIR = "./xai_outputs"
