# 🌸 Oxford 102 Flowers Classification

> **Achieving 98.96% Accuracy Through Systematic Architecture Comparison, 2-Stage Training, and Test-Time Augmentation**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.96%25-brightgreen.svg)]()

---

## 🏆 Final Results

| Model | Training Strategy | Test Accuracy |
|-------|-------------------|---------------|
| ResNet-50 | Standard | 58.12% |
| EfficientNet-B2 | Standard | 86.24% |
| ViT-Small | Standard | 97.54% |
| ViT-Small | 2-Stage Fine-tune | 98.75% |
| **ViT-Small** | **2-Stage + TTA** | **98.96%** ⭐ |

### Best Model Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | 98.75% (98.96% with TTA) |
| Precision (Macro) | 98.52% |
| Recall (Macro) | 98.74% |
| F1-Score (Macro) | 98.60% |
| ROC-AUC | 0.9998 |
| Total Errors | 77 / 6,149 (64 with TTA) |

---

## 🎁 Bonus Features Implemented

| Bonus | Status | Impact |
|-------|--------|--------|
| ✅ Gradio Demo | Complete | Interactive web interface for live predictions |
| ✅ Test-Time Augmentation | Complete | **+0.23% accuracy** (98.73% → 98.96%) |
| ⬜ MC Dropout | Not attempted | - |
| ⬜ Knowledge Distillation | Not attempted | - |
| ⬜ Object Detection | Not attempted | - |

### 🎨 Gradio Demo (Notebook 06)
Interactive web interface where anyone can upload flower images and get instant predictions with confidence scores. Run the notebook to generate a public shareable URL.

**Features:**
- Drag-and-drop image upload
- Top-5 predictions with confidence bars
- Works directly in Google Colab
- Public URL valid for 72 hours

### 🔄 Test-Time Augmentation (Notebook 07)
Averages predictions across 10 augmented views to improve robustness:

| Metric | Value |
|--------|-------|
| Baseline Accuracy | 98.73% |
| **TTA Accuracy** | **98.96%** |
| Improvement | +0.23% |
| TTA Helped | 22 cases |
| TTA Hurt | 8 cases |
| **Net Gain** | **14 correct predictions** |

**Trade-off:** 10x slower inference (10 forward passes per image) - worth it for high-stakes predictions.

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Business Context](#business-context--problem-statement)
3. [Dataset Analysis](#dataset-analysis)
4. [Methodology](#methodology)
5. [Architecture Comparison](#architecture-comparison)
6. [Training Strategy](#training-strategy)
7. [Ensemble Experiment](#ensemble-experiment)
8. [Explainability Analysis](#explainability-analysis)
9. [Error Analysis](#error-analysis)
10. [Production Readiness](#production-readiness)
11. [Key Insights](#key-insights--conclusions)
12. [Project Structure](#project-structure)
13. [Quick Start](#quick-start)

---

## Executive Summary

This project presents a systematic approach to fine-grained image classification on the Oxford 102 Flowers dataset. Through rigorous experimentation comparing three architectures (ResNet-50, EfficientNet-B2, and ViT-Small) under identical conditions, we discovered that **Vision Transformers dramatically outperform CNNs** for this task.

### Key Findings

1. **ViT-Small achieves 98.96% accuracy** with TTA - exceptional for this challenging dataset
2. **Architecture matters more than training tricks** - ViT outperformed CNNs by 11-40%
3. **2-stage training provides +1.20% boost** - reducing errors by 48%
4. **Ensembles don't help** - single optimized ViT beats all combinations
5. **TTA adds +0.23%** - 14 additional correct predictions
6. **Global attention is crucial** - fine-grained classification needs holistic understanding

---

## Business Context & Problem Statement

### The Challenge

A botanical research company needs an automated system to identify flower species from field images for biodiversity studies. The system must be:

- **Accurate**: Correctly identify 102 different flower species
- **Robust**: Handle real-world image variations
- **Explainable**: Researchers need to trust and understand predictions
- **Deployable**: Run efficiently on standard hardware

### Why Fine-Grained Classification is Hard

| Challenge | General Classification | Fine-Grained (Flowers) |
|-----------|----------------------|------------------------|
| Inter-class difference | Dog vs Car (obvious) | Petunia vs Morning Glory (subtle) |
| Key features | Shape, size, context | Petal arrangement, color gradients, stamen patterns |
| Required understanding | Local features sufficient | Global structure essential |

---

## Dataset Analysis

### Oxford 102 Flowers Dataset

| Split | Images | Images/Class | Purpose |
|-------|--------|--------------|---------|
| Train | 1,020 | 10 | Model training |
| Validation | 1,020 | 10 | Hyperparameter tuning |
| Test | 6,149 | ~60 (varies) | Final evaluation |
| **Total** | **8,189** | **102 classes** | |

### Critical Observations

**⚠️ Extremely Small Training Set** - Only 10 images per class! This makes transfer learning essential.

**Visual Challenges Identified:**
- Inter-class similarity (Petunia vs Mexican Petunia)
- Intra-class variation (same species in sun vs shade)
- Background clutter (garden scenes, multiple flowers)
- Scale variation (close-up vs distant shots)

### Augmentation Strategy
```python
train_transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),  # Simulate camera distance
    A.HorizontalFlip(p=0.5),                          # Flowers are symmetric
    A.Affine(rotate=(-30, 30), p=0.5),               # Natural angle variation
    A.ColorJitter(brightness=0.2, contrast=0.2, 
                  saturation=0.3, hue=0.1),           # Color is key!
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
```

**Key decisions:**
- ✅ `HorizontalFlip` - Flowers are symmetric left-to-right
- ❌ `VerticalFlip` - NOT used - flowers don't grow upside down
- ✅ High `saturation` jitter - Color is primary differentiator

---

## Methodology

### Experimental Design
```
1. HYPOTHESIS: Architecture choice matters more than training tricks
2. EXPERIMENT: Compare 3 architectures with identical training setup
3. ANALYSIS: Select winner based on validation performance
4. OPTIMIZATION: Apply 2-stage fine-tuning to winner
5. ENHANCEMENT: Add TTA for final accuracy boost
6. VALIDATION: Confirm on held-out test set
```

### Fair Comparison Protocol

| Variable | Setting | Rationale |
|----------|---------|-----------|
| Optimizer | AdamW | State-of-the-art for vision |
| Learning Rate | 1e-4 | Standard for fine-tuning |
| Batch Size | 32 | Fits in Colab GPU memory |
| Epochs | 15 | Early stopping prevents overfitting |
| Augmentation | Identical | Isolate architecture effect |
| Seed | 42 | Reproducibility |

---

## Architecture Comparison

### Candidates

| Model | Type | Parameters | Pretrained On |
|-------|------|------------|---------------|
| ResNet-50 | CNN (residual) | 25.6M | ImageNet-1K |
| EfficientNet-B2 | CNN (compound scaling) | 9.2M | ImageNet-1K |
| ViT-Small | Transformer | 21.7M | ImageNet-21K |

### Results

| Model | Val Accuracy | Test Accuracy | Verdict |
|-------|-------------|---------------|---------|
| ResNet-50 | 59.51% | 58.12% | ❌ Underfitting |
| EfficientNet-B2 | 87.06% | 86.24% | ⚠️ Good, not great |
| **ViT-Small** | **97.75%** | **97.54%** | ✅ **Clear winner** |

### Why ViT Dominates

| Factor | CNN (EfficientNet) | Transformer (ViT) |
|--------|-------------------|-------------------|
| Receptive field | Local → grows with depth | Global from layer 1 |
| Part relationships | Implicit, indirect | Explicit via attention |
| Pretraining data | 1.2M images | 14M images (10x more) |

**Key insight:** Fine-grained classification requires understanding how petals relate to centers, how colors blend across the flower. ViT's global attention captures this; CNNs struggle.

---

## Training Strategy

### 2-Stage Fine-Tuning

#### Stage 1: Head Warm-up (5 epochs)
```python
# Freeze backbone, train only classifier head
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

optimizer = AdamW(model.head.parameters(), lr=1e-3)
```

#### Stage 2: Full Fine-Tuning (until convergence)
```python
# Unfreeze all, use differential learning rates
param_groups = [
    {'params': model.head.parameters(), 'lr': 1e-4},      # Head: higher LR
    {'params': model.blocks.parameters(), 'lr': 1e-5},    # Backbone: lower LR
]
optimizer = AdamW(param_groups, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=30)
```

### Results

| Stage | Val Accuracy | Test Accuracy |
|-------|--------------|---------------|
| After Stage 1 | 92.3% | - |
| After Stage 2 | 98.92% | 98.75% |
| **Improvement** | **+1.17%** | **+1.20%** |

The 2-stage approach reduced errors from 151 to 77 — a **48% error reduction**!

---

## Ensemble Experiment

### Research Question
> Does combining multiple models beat our optimized single ViT?

### Results

| Strategy | Test Accuracy | vs Single ViT |
|----------|---------------|---------------|
| Single ViT (2-stage) | **98.75%** | Baseline |
| Simple Average (all 3) | 87.42% | -11.33% ❌ |
| Weighted Average (all 3) | 93.51% | -5.24% ❌ |
| Top-2 Weighted | 98.21% | -0.54% ❌ |

### Conclusion
Ensembles **hurt** performance because ViT is simply better at almost everything. When models disagree, ViT is correct 23:1 over EfficientNet.

---

## Explainability Analysis

### ViT Attention Visualization

We visualized what image regions the model attends to when making predictions:

**Findings:**
- ✅ Model focuses on petals, centers, and stamens
- ✅ Different attention heads specialize in different features
- ✅ Backgrounds and leaves appropriately ignored
- ✅ **No evidence of shortcut learning**

### Multi-Layer Analysis

| Layer | Focus Pattern | Interpretation |
|-------|---------------|----------------|
| 1-3 | Edges, textures | Low-level features |
| 4-7 | Petal shapes, color regions | Mid-level parts |
| 8-12 | Whole flower structure | High-level semantics |

---

## Error Analysis

### Error Distribution (77 total errors)

| Error Type | Count | Percentage |
|------------|-------|------------|
| Inter-class similarity | 31 | 40% |
| Multiple flowers in image | 19 | 25% |
| Unusual viewpoint | 12 | 16% |
| Partial occlusion | 9 | 12% |
| Atypical specimen | 6 | 8% |

### Most Confused Pairs

| True Class | Predicted As | Botanical Similarity |
|------------|--------------|---------------------|
| Sword Lily | Hippeastrum | Both have elongated petals |
| Camellia | Mallow | Overlapping petal structure |
| Petunia | Morning Glory | Trumpet-shaped flowers |

**Key insight:** Errors are botanically understandable - confused pairs are genuinely similar species.

---

## Production Readiness

### Inference Pipeline
```python
from src.models.classifier import FlowerClassifier

classifier = FlowerClassifier(
    model_path="artifacts/models/vit_2stage_best.pt",
    device="cuda"
)

result = classifier.predict("path/to/flower.jpg")
# Returns: {
#     "class_name": "sunflower",
#     "confidence": 0.9847,
#     "top_k": [("sunflower", 0.98), ("black-eyed susan", 0.01), ...]
# }
```

### Deployment Recommendations

| Optimization | Impact |
|--------------|--------|
| ONNX Export | ~20% speedup |
| TensorRT | ~3x speedup on NVIDIA |
| INT8 Quantization | 4x smaller, 2x faster |

---

## Key Insights & Conclusions

1. **Architecture choice is the biggest lever** - ViT outperformed CNNs by 11-40%

2. **2-stage fine-tuning is essential** - 48% error reduction

3. **Ensembles aren't always better** - Single optimized model wins

4. **TTA provides free accuracy** - +0.23% with no retraining

5. **Explainability builds trust** - Attention maps confirm relevant features

6. **Errors are botanically understandable** - Model struggles where humans would too

---

## 📁 Project Structure
```
flowers-cv/
├── notebooks/
│   ├── 01_EDA.ipynb                    # Data exploration
│   ├── 02_Model_Comparison.ipynb       # Architecture comparison
│   ├── 03_Winner_Optimization.ipynb    # 2-stage training
│   ├── 04_Ensemble_Experiment.ipynb    # Ensemble strategies
│   ├── 05_Evaluation_and_Explainability.ipynb  # Metrics & attention
│   ├── 06_Gradio_Demo.ipynb            # 🎁 Bonus: Interactive demo
│   └── 07_TTA_Evaluation.ipynb         # 🎁 Bonus: Test-time augmentation
├── src/
│   ├── data/dataset.py                 # Dataset classes
│   ├── models/
│   │   ├── build_model.py              # Model factory
│   │   └── classifier.py               # Production inference
│   ├── training/
│   │   ├── engine.py                   # Training loops
│   │   └── callbacks.py                # EarlyStopping, Checkpointing
│   └── utils/                          # Seed, visualization
├── artifacts/
│   ├── models/                         # Trained weights
│   ├── figures/                        # Visualizations
│   └── reports/                        # JSON metrics
├── configs/config.yaml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Harik7-cmd/Flowers-cv.git
cd Flowers-cv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run in Colab

1. Upload to Google Drive
2. Open notebooks in order (01 → 07)
3. Runtime → Change runtime type → T4 GPU
4. Run all cells

### Quick Inference
```python
import torch
import timm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Load model
model = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", num_classes=102)
ckpt = torch.load("artifacts/models/vit_2stage_best.pt", map_location="cuda")
model.load_state_dict(ckpt)
model.eval().cuda()

# Transform
transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Predict
img = Image.open("flower.jpg").convert("RGB")
x = transform(image=np.array(img))["image"].unsqueeze(0).cuda()

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)
    pred = probs.argmax().item()
    conf = probs.max().item()
    
print(f"Predicted class: {pred}, Confidence: {conf:.1%}")
```

---

## Requirements
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
albumentations>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
gradio>=4.0.0
tqdm>=4.64.0
pyyaml>=6.0
Pillow>=9.0.0
numpy>=1.21.0
```

---

## Acknowledgments

- **Dataset**: [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- **Models**: [timm](https://github.com/huggingface/pytorch-image-models)
- **Augmentation**: [Albumentations](https://albumentations.ai/)

---

**Assessment**: AQREIGHT Computer Vision Engineer  
**Date**: January 2025

---

<p align="center">
  <i>Built with 🌸 and PyTorch</i>
</p>
