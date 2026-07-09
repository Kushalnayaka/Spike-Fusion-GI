# SpikeFusion-GI 2.0 — Agent Coordination Spec

## User Goal
Build a complete, production-ready PyTorch project that classifies GI endoscopy images into 8 categories using the Kvasir v2 dataset (8,000 images). The model fuses three branches (SNN for edges, CNN for colour/texture, Vision Mamba for global context) and includes full XAI (explainable AI) visualization.

## Architecture (from user diagram)
```
Retinal Sobel edges → LIF-SNN (edge features)
                           ↘
RGB-CNN (colour + texture) → Fusion (edge + colour) → Tokens (49×128) → Mamba (global context) → Class (8 labels)
                           ↗
```

Target: ~0.81M parameters, 0% wasted params, all task-relevant.

## Dataset
- Kvasir v2: 8 classes, 8,000 images (~1,000 per class)
- Classes: dyed-lifted-polyps, dyed-resection-margins, esophagitis, normal-cecum, normal-pylorus, normal-z-line, polyps, ulcerative-colitis
- Download: https://datasets.simula.no/kvasir/
- Preprocessing: 224×224, ImageNet normalization, augmentation

## Shared Contract

### Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
mamba-ssm>=1.1.0  # Vision Mamba backbone
einops>=0.7.0
scikit-learn
matplotlib
seaborn
opencv-python
pillow
tqdm
tensorboard
pandas
numpy
```

### Directory Layout
```
spikefusion-gi/
├── configs/
│   └── config.py          # Hyperparameters and constants
├── models/
│   ├── __init__.py
│   ├── snn_branch.py      # LIF-SNN for edge features
│   ├── cnn_branch.py      # RGB-CNN for colour/texture
│   ├── fusion.py          # Cross-modal fusion module
│   ├── mamba_core.py      # Vision Mamba for global context
│   └── spikefusion.py     # Main model assembly
├── data/
│   ├── __init__.py
│   └── kvasir_dataset.py  # Dataset loader with transforms
├── utils/
│   ├── __init__.py
│   ├── metrics.py         # Accuracy, F1, sensitivity, specificity, AUC
│   ├── logger.py          # Training logger
│   └── helpers.py         # Misc utilities
├── xai/
│   ├── __init__.py
│   ├── gradcam.py         # Grad-CAM for CNN branch
│   ├── attention_viz.py   # Mamba attention/state visualization
│   ├── spike_viz.py       # SNN spike raster visualization
│   └── explain.py         # Unified XAI runner
├── train.py               # Training loop
├── evaluate.py            # Evaluation + metrics
├── infer.py               # Single-image inference + XAI
└── requirements.txt
```

### Data Interface
Dataset returns: `(image_tensor, label)` where image_tensor is `[3, 224, 224]` float32, label is int 0-7.

### Model Interface
```python
class SpikeFusion(nn.Module):
    def __init__(self, num_classes=8, img_size=224, embed_dim=128, mamba_depth=4):
        ...
    def forward(self, x):
        # x: [B, 3, 224, 224]
        # returns: logits [B, num_classes], aux_outputs dict for XAI
        ...
```

`aux_outputs` must contain:
- `cnn_features`: CNN branch output before fusion [B, C, H, W]
- `snn_spikes`: SNN spike tensor [B, T, C, H, W] or [B, C, H, W]
- `edge_map`: Sobel edge map [B, 1, H, W]
- `mamba_states`: Mamba hidden states [B, N, D]
- `fused_tokens`: Token sequence before Mamba [B, N, D]

### Hyperparameters (config.py)
```python
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 8
EMBED_DIM = 128
MAMBA_DEPTH = 4
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2
NUM_EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
SNN_TIMESTEPS = 4
SNN_TAU = 2.0
SNN_VTH = 1.0
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## Task Slices

### Worker A: Core Model (`models/`)
- Implement `snn_branch.py`: Sobel edge extraction + LIF spiking neuron layers
- Implement `cnn_branch.py`: Lightweight CNN (MobileNet-style or custom efficient CNN)
- Implement `fusion.py`: Concat + 1x1 conv fusion of edge and colour features
- Implement `mamba_core.py`: Vision Mamba block using selective scan (simplified from VMamba paper)
- Implement `spikefusion.py`: Main model assembling all branches
- Target: ~0.81M parameters

### Worker B: Data + Training (`data/`, `utils/`, `train.py`)
- Implement `kvasir_dataset.py`: Kvasir v2 loader with augmentations
- Implement `metrics.py`: Per-class accuracy, F1, sensitivity, specificity, AUC, confusion matrix
- Implement `train.py`: Full training loop with AMP, LR scheduler (cosine), early stopping, checkpointing
- Implement `evaluate.py`: Evaluation script

### Worker C: XAI + Inference (`xai/`, `infer.py`)
- Implement `gradcam.py`: Grad-CAM for CNN branch attention maps
- Implement `attention_viz.py`: Mamba SSM state evolution visualization
- Implement `spike_viz.py`: SNN spike raster + rate maps
- Implement `explain.py`: Unified XAI pipeline
- Implement `infer.py`: Single-image inference with all XAI outputs

## Key Design Decisions
1. **SNN Branch**: Use surrogate gradient (atan or fast_sigmoid) for backprop through LIF neurons. Sobel edges as input to SNN, not raw RGB.
2. **CNN Branch**: Use a lightweight efficient CNN (depthwise separable convs) to keep params low.
3. **Fusion**: Simple but effective — concat + 1x1 conv to project to embed_dim, then patchify to tokens.
4. **Mamba**: Use a simplified selective scan SSM. If `mamba-ssm` unavailable, implement a pure-Python fallback.
5. **XAI**: Grad-CAM on CNN branch, SNN spike visualization, Mamba state heatmaps, integrated gradients optional.

## References
- Kvasir dataset: https://datasets.simula.no/kvasir/
- Vision Mamba (VMamba): https://arxiv.org/abs/2401.09417
- Spiking Neural Networks with surrogate gradients: Neftci et al., "Surrogate Gradient Learning in Spiking Neural Networks"
- Mamba: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"

## Merge Order
1. Worker A commits models/ + configs/
2. Worker B commits data/ + utils/ + train.py + evaluate.py
3. Worker C commits xai/ + infer.py
4. Main agent integrates, writes requirements.txt, runs syntax check
