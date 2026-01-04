# 🌸 Oxford 102 Flowers Classification

> **A Deep Learning Journey: Achieving 98.75% Accuracy Through Systematic Architecture Comparison and Training Optimization**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.75%25-brightgreen.svg)]()

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Business Context & Problem Statement](#business-context--problem-statement)
3. [Dataset Analysis](#dataset-analysis)
4. [Methodology](#methodology)
5. [Architecture Comparison](#architecture-comparison)
6. [Training Strategy](#training-strategy)
7. [Ensemble Experiment](#ensemble-experiment)
8. [Final Results](#final-results)
9. [Explainability Analysis](#explainability-analysis)
10. [Error Analysis](#error-analysis)
11. [Production Readiness](#production-readiness)
12. [Key Insights & Conclusions](#key-insights--conclusions)
13. [Project Structure](#project-structure)
14. [Quick Start](#quick-start)

---

## Executive Summary

This project presents a systematic approach to fine-grained image classification on the Oxford 102 Flowers dataset. Through rigorous experimentation comparing three architectures (ResNet-50, EfficientNet-B2, and ViT-Small) under identical conditions, we discovered that **Vision Transformers dramatically outperform CNNs** for this task.

### 🏆 Final Results

| Model | Training Strategy | Test Accuracy | Improvement |
|-------|-------------------|---------------|-------------|
| ResNet-50 | Standard fine-tune | 58.12% | Baseline |
| EfficientNet-B2 | Standard fine-tune | 86.24% | +28.12% |
| ViT-Small | Standard fine-tune | 97.54% | +39.42% |
| **ViT-Small** | **2-Stage fine-tune** | **98.75%** ⭐ | **+40.63%** |

### Key Findings

1. **ViT-Small achieves 98.75% accuracy** - exceeding expectations for this challenging dataset
2. **Architecture matters more than training tricks** - ViT outperformed CNNs by 11-40%
3. **2-stage training provides +1.20% boost** - reducing errors by 48%
4. **Ensembles don't help** - single optimized ViT beats all combinations
5. **Global attention is crucial** - fine-grained classification needs holistic understanding

---

## Business Context & Problem Statement

### The Challenge

A botanical research company needs an automated system to identify flower species from field images for biodiversity studies. Field researchers capture images in varying conditions - different lighting, angles, backgrounds, and camera distances. The system must be:

- **Accurate**: Correctly identify 102 different flower species
- **Robust**: Handle real-world image variations
- **Explainable**: Researchers need to trust and understand predictions
- **Deployable**: Run efficiently on standard hardware

### Why This Problem is Hard

Fine-grained classification is fundamentally different from general image classification:

| Challenge | General Classification | Fine-Grained (Flowers) |
|-----------|----------------------|------------------------|
| Inter-class difference | Dog vs Car (obvious) | Petunia vs Morning Glory (subtle) |
| Key features | Shape, size, context | Petal arrangement, color gradients, stamen patterns |
| Required understanding | Local features sufficient | Global structure essential |
| Training data | Abundant | Often limited |

### Success Criteria

- Primary: Achieve high test accuracy (target: >90%)
- Secondary: Understand *why* the model works (explainability)
- Tertiary: Ensure reproducibility and production readiness

---

## Dataset Analysis

### Oxford 102 Flowers Dataset

The Oxford 102 Flowers dataset is a benchmark for fine-grained visual categorization, containing 102 flower species commonly found in the United Kingdom.

#### Dataset Statistics

| Split | Images | Images/Class | Purpose |
|-------|--------|--------------|---------|
| Train | 1,020 | 10 | Model training |
| Validation | 1,020 | 10 | Hyperparameter tuning |
| Test | 6,149 | ~60 (varies) | Final evaluation |
| **Total** | **8,189** | **102 classes** | |

#### Critical Observations

**1. Extremely Small Training Set**
```
Only 10 images per class for training!
```
This is the defining characteristic of this dataset. Most deep learning models expect thousands of images per class. With only 10, we must rely heavily on:
- Transfer learning from pretrained models
- Strong data augmentation
- Careful regularization

**2. Class Balance**
- Training/Validation: Perfectly balanced (10 per class)
- Test: Variable (20-80 per class)
- **Decision**: No class weighting needed for training

**3. Image Characteristics**
- Resolution: Highly variable (500x500 to 1024x768)
- Aspect ratios: Mixed (requires careful resizing)
- Quality: Generally good, some with watermarks
- **Decision**: Resize to 224x224 with center cropping

**4. Visual Challenges Identified**

| Challenge | Example | Impact |
|-----------|---------|--------|
| Inter-class similarity | Petunia vs Mexican Petunia | High confusion potential |
| Intra-class variation | Same species in sun vs shade | Model must generalize |
| Background clutter | Garden scenes, multiple flowers | Risk of shortcut learning |
| Scale variation | Close-up vs distant shots | Need scale-invariant features |
| Watermarks | Stock photo watermarks | Must ignore irrelevant features |

#### Augmentation Strategy

Based on our EDA, we designed flower-specific augmentations:
```python
train_transform = A.Compose([
    # Spatial transforms
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),  # Simulate camera distance
    A.HorizontalFlip(p=0.5),                          # Flowers are horizontally symmetric
    A.Affine(rotate=(-30, 30), p=0.5),               # Natural angle variation
    
    # Color transforms (critical for flowers!)
    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.3,  # Higher - color is key identifier
        hue=0.1          # Lower - preserve species color
    ),
    
    # Regularization
    A.GaussianBlur(blur_limit=3, p=0.1),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
    
    # Normalization (ImageNet statistics)
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
```

**Why these choices?**
- ✅ `HorizontalFlip`: Flowers are symmetric left-to-right
- ❌ `VerticalFlip`: NOT used - flowers don't grow upside down
- ✅ `ColorJitter` with high saturation: Color is primary differentiator
- ✅ `CoarseDropout`: Forces model to use multiple features, not just one

---

## Methodology

### Experimental Design

Our approach follows the scientific method:
```
1. HYPOTHESIS: Architecture choice matters more than training tricks
2. EXPERIMENT: Compare 3 architectures with identical training setup
3. ANALYSIS: Select winner based on validation performance
4. OPTIMIZATION: Apply advanced training techniques to winner
5. VALIDATION: Confirm on held-out test set
```

### Fair Comparison Protocol

To ensure valid comparisons, we controlled all variables except architecture:

| Variable | Setting | Rationale |
|----------|---------|-----------|
| Optimizer | AdamW | State-of-the-art for vision |
| Learning Rate | 1e-4 | Standard for fine-tuning |
| Batch Size | 32 | Fits in Colab GPU memory |
| Epochs | 15 | Early stopping prevents overfitting |
| Augmentation | Identical | Isolate architecture effect |
| Seed | 42 | Reproducibility |
| Weight Decay | 0.01 | Regularization |

### Reproducibility

Every experiment is fully reproducible:
```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## Architecture Comparison

### Candidates

We selected three architectures representing different design philosophies:

#### 1. ResNet-50 (Baseline CNN)

**Architecture**: Residual Network with skip connections
**Parameters**: 25.6M
**Pretrained on**: ImageNet-1K (1.2M images, 1000 classes)
```
Why ResNet-50?
├── Classic, well-understood baseline
├── Skip connections enable deep networks
├── Strong ImageNet performance
└── Fast inference
```

**Hypothesis**: Good baseline, but local receptive field may limit fine-grained discrimination.

#### 2. EfficientNet-B2 (Efficient CNN)

**Architecture**: Compound-scaled CNN (depth, width, resolution)
**Parameters**: 9.2M
**Pretrained on**: ImageNet-1K
```
Why EfficientNet-B2?
├── State-of-the-art efficiency (accuracy per FLOP)
├── Compound scaling optimizes all dimensions
├── B2 balances accuracy vs compute
└── Proven on fine-grained benchmarks
```

**Hypothesis**: Better than ResNet due to optimized architecture, but still limited by local operations.

#### 3. ViT-Small (Vision Transformer)

**Architecture**: Transformer with self-attention over image patches
**Parameters**: 21.7M
**Pretrained on**: ImageNet-21K (14M images, 21,000 classes)
```
Why ViT-Small?
├── Global attention from first layer
├── Explicit modeling of part relationships
├── Pretrained on 10x more data (21K classes)
└── Different inductive bias than CNNs
```

**Hypothesis**: Global attention will excel at fine-grained classification where understanding part relationships matters.

### Results: Architecture Comparison

| Model | Val Accuracy | Test Accuracy | Gap | Verdict |
|-------|-------------|---------------|-----|---------|
| ResNet-50 | 59.51% | 58.12% | -1.39% | ❌ Underfitting |
| EfficientNet-B2 | 87.06% | 86.24% | -0.82% | ⚠️ Good, not great |
| **ViT-Small** | **97.75%** | **97.54%** | **-0.21%** | ✅ **Clear winner** |

### Analysis: Why ViT Dominates

**The 11% Gap (ViT vs EfficientNet)**

| Factor | CNN (EfficientNet) | Transformer (ViT) |
|--------|-------------------|-------------------|
| Receptive field | Local → grows with depth | Global from layer 1 |
| Part relationships | Implicit, indirect | Explicit via attention |
| Pretraining data | 1.2M images | 14M images (10x more) |
| Inductive bias | Translation equivariance | Minimal (learns from data) |

**Why Global Attention Matters for Flowers:**
```
To distinguish Petunia from Morning Glory:
├── CNN approach: Build up local features → combine at end
│   └── Risk: May focus on textures, miss global structure
│
└── ViT approach: Compare all patches simultaneously
    └── Benefit: Sees how petals relate to center, color gradients across flower
```

**Attention Visualization Confirms:**
- ViT attention heads focus on botanically relevant regions
- Different heads specialize (some focus on petals, others on centers)
- No evidence of shortcut learning (backgrounds ignored)

---

## Training Strategy

### 2-Stage Fine-Tuning

After selecting ViT-Small as our winner, we applied a 2-stage fine-tuning strategy to maximize performance:

#### Why 2-Stage?
```
Problem with standard fine-tuning:
├── Pretrained backbone: High-quality features (trained on 14M images)
├── New classifier head: Random weights
└── Same learning rate for both → Destroys pretrained features!

Solution: 2-Stage approach
├── Stage 1: Train head only (backbone frozen)
│   └── Aligns random head with pretrained features
└── Stage 2: Unfreeze backbone with lower LR
    └── Gentle adaptation preserves knowledge
```

#### Stage 1: Head Warm-up (5 epochs)
```python
# Freeze backbone
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

# Higher LR for random head
optimizer = AdamW(model.head.parameters(), lr=1e-3)
```

**Purpose**: 
- Random classifier head needs aggressive training
- Frozen backbone provides stable features
- Prevents early gradient noise from corrupting backbone

**Results after Stage 1**: 92.3% validation accuracy

#### Stage 2: Full Fine-Tuning (until convergence)
```python
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Differential learning rates
param_groups = [
    {'params': model.head.parameters(), 'lr': 1e-4},      # Head: higher LR
    {'params': model.patch_embed.parameters(), 'lr': 1e-5}, # Backbone: lower LR
    {'params': model.blocks.parameters(), 'lr': 1e-5},
]
optimizer = AdamW(param_groups, weight_decay=0.01)

# Cosine annealing scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
```

**Why Differential Learning Rates?**
- Backbone already has good features → small adjustments only
- Head is still learning → needs larger updates
- Ratio of 10:1 (head:backbone) is empirically optimal

**Training Infrastructure**:
- Early stopping with patience=7 on validation loss
- Model checkpointing saves best weights
- Gradient clipping (max_norm=1.0) prevents instability

#### 2-Stage Results

| Stage | Epochs | Val Accuracy | Test Accuracy |
|-------|--------|--------------|---------------|
| After Stage 1 | 5 | 92.3% | - |
| After Stage 2 | 12 (early stopped) | 98.92% | 98.75% |
| **Improvement** | - | **+1.17%** | **+1.20%** |

The 2-stage approach reduced test errors from 151 to 77 — a **48% error reduction**!

---

## Ensemble Experiment

### Research Question

> Does combining multiple models (ensemble) beat our optimized single ViT?

### Hypothesis

Given the performance gap (ViT: 97.54% vs EfficientNet: 86.24% vs ResNet: 58.12%), we hypothesized that **ensembles would not help** because:
1. Weak models add noise, not signal
2. When one model dominates, averaging dilutes its predictions

### Ensemble Strategies Tested

| Strategy | Method | Models Used |
|----------|--------|-------------|
| Simple Average | Mean of softmax probabilities | All 3 |
| Weighted Average | Weight by validation accuracy | All 3 |
| Top-2 Average | Exclude weakest model | ViT + EfficientNet |
| Top-2 Weighted | Weighted top 2 | ViT + EfficientNet |

### Results

| Strategy | Test Accuracy | vs Single ViT |
|----------|---------------|---------------|
| Single ViT (2-stage) | **98.75%** | Baseline |
| Simple Average (all 3) | 87.42% | -11.33% ❌ |
| Weighted Average (all 3) | 93.51% | -5.24% ❌ |
| Top-2 Average | 97.98% | -0.77% ❌ |
| Top-2 Weighted | 98.21% | -0.54% ❌ |

### Analysis: Why Ensembles Failed

**Error Correlation Analysis:**

| Model Pair | Error Overlap | Unique Errors |
|------------|---------------|---------------|
| ViT vs EfficientNet | 42 shared | ViT: 35, Eff: 804 |
| ViT vs ResNet | 31 shared | ViT: 46, Res: 2543 |

**Key Insight**: ViT makes 77 errors. EfficientNet makes 846 errors. When they disagree:
- ViT correct, EfficientNet wrong: 804 cases
- EfficientNet correct, ViT wrong: 35 cases
- **Win ratio: 23:1 in favor of ViT**

**Conclusion**: Ensembling only helps when models make *complementary* errors. Here, ViT is simply better at almost everything.

---

## Final Results

### Classification Metrics
```
══════════════════════════════════════════════════════════════════════
              FINAL MODEL: ViT-Small with 2-Stage Fine-Tuning
══════════════════════════════════════════════════════════════════════

DATASET:
├── Test Images: 6,149
├── Classes: 102
└── Challenge: Fine-grained flower classification

PERFORMANCE:
├── Test Accuracy:     98.75%
├── Precision (Macro): 98.52%
├── Recall (Macro):    98.74%
├── F1-Score (Macro):  98.60%
└── ROC-AUC (Micro):   0.9998

ERROR ANALYSIS:
├── Total Errors: 77 / 6,149
├── Error Rate: 1.25%
└── Error Reduction vs Baseline: 48% (from 151 errors)

══════════════════════════════════════════════════════════════════════
```

### Model Specifications

| Specification | Value |
|--------------|-------|
| Architecture | ViT-Small (patch16, 224x224) |
| Parameters | 21.7M |
| Model Size | 83 MB |
| Inference Time | 1.54ms (GPU) |
| Throughput | 649 images/sec |
| Pretrained On | ImageNet-21K |

### Per-Class Performance

Most classes achieve 100% accuracy. Classes with <98% accuracy:

| Class | Accuracy | Common Confusion | Reason |
|-------|----------|------------------|--------|
| Sword Lily | 95.2% | Hippeastrum | Similar elongated petals |
| Camellia | 96.8% | Mallow | Overlapping petal structure |
| Petunia | 97.1% | Morning Glory | Trumpet-shaped flowers |
| Ball Moss | 97.4% | Bromelia | Spiky texture similarity |

---

## Explainability Analysis

### Why Explainability Matters

For a botanical research tool, researchers need to:
1. **Trust** the model's predictions
2. **Understand** why a prediction was made
3. **Identify** when the model might be wrong

### Method: ViT Self-Attention Visualization

Vision Transformers provide built-in explainability through attention weights. We visualize what image regions the model attends to when making predictions.
```python
# Extract attention from [CLS] token to all patches
attention = model.blocks[-1].attn.get_attention_map()  # [1, heads, 197, 197]
cls_attention = attention[0, :, 0, 1:]  # [heads, 196] - CLS to patches
```

### Findings

**Correctly Classified Samples:**
- Model consistently focuses on flower-relevant regions
- Petals, centers, and stamens receive highest attention
- Backgrounds and leaves are appropriately ignored
- Different attention heads specialize in different features

**Misclassified Samples:**
- Attention often split between multiple flowers in image
- Unusual angles cause attention to scatter
- Partially occluded flowers show fragmented attention

### Multi-Layer Attention Analysis

We analyzed attention across all 12 transformer layers:

| Layer | Focus Pattern | Interpretation |
|-------|---------------|----------------|
| 1-3 | Edges, textures | Low-level features |
| 4-7 | Petal shapes, color regions | Mid-level parts |
| 8-12 | Whole flower structure | High-level semantics |

**Key Finding**: The model builds hierarchical representations, with later layers capturing species-distinguishing features.

### Verification: No Shortcut Learning

We specifically checked for problematic patterns:
- ❌ Watermark focus: NOT observed
- ❌ Background reliance: NOT observed  
- ❌ Pot/container focus: NOT observed
- ✅ Flower-centric attention: CONFIRMED

---

## Error Analysis

### Error Distribution

Of 77 total errors:

| Error Type | Count | Percentage | Example |
|------------|-------|------------|---------|
| Inter-class similarity | 31 | 40% | Sword lily ↔ Canna lily |
| Multiple flowers in image | 19 | 25% | Bouquet images |
| Unusual viewpoint | 12 | 16% | Extreme close-up or angle |
| Partial occlusion | 9 | 12% | Flower behind leaves |
| Atypical specimen | 6 | 8% | Unusual coloration |

### Most Confused Class Pairs

| True Class | Predicted As | Count | Botanical Similarity |
|------------|--------------|-------|---------------------|
| Sword Lily | Hippeastrum | 5 | Both have elongated petals in star pattern |
| Camellia | Mallow | 4 | Both have layered petal arrangement |
| Petunia | Morning Glory | 3 | Both are trumpet-shaped |
| Sweet Pea | Globe-flower | 3 | Delicate, similar size |

### Insights from Error Analysis

1. **Errors are botanically understandable** - Confused pairs are genuinely similar species that even humans struggle with

2. **Multi-flower images are problematic** - Model wasn't trained on compositional scenes

3. **Dataset has labeling noise** - Some "errors" appear to be mislabeled ground truth

### Potential Improvements

Based on error analysis:

| Improvement | Expected Impact | Effort |
|-------------|-----------------|--------|
| Multi-crop TTA | Reduce scale/position errors | Low |
| Larger input resolution | Capture fine details | Medium |
| Object detection preprocessing | Handle multi-flower images | High |
| Active learning on hard cases | Target specific confusions | High |

---

## Production Readiness

### Inference Pipeline
```python
from src.models.classifier import FlowerClassifier

# Initialize classifier (loads trained model)
classifier = FlowerClassifier(
    model_path="artifacts/models/vit_2stage_best.pt",
    device="cuda"
)

# Single image prediction
result = classifier.predict("path/to/flower.jpg")
print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Top-3: {result['top_k']}")

# Batch prediction
results = classifier.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
```

### API Response Format
```json
{
    "predicted_class": "sunflower",
    "class_index": 54,
    "confidence": 0.9847,
    "top_k_predictions": [
        {"class": "sunflower", "confidence": 0.9847},
        {"class": "black-eyed susan", "confidence": 0.0089},
        {"class": "gazania", "confidence": 0.0031}
    ],
    "inference_time_ms": 1.54
}
```

### Production Deployment Recommendations

**Model Optimization:**
```
Current: PyTorch model (83MB, 1.54ms inference)
│
├── ONNX Export → ~20% speedup, framework-agnostic
├── TensorRT → ~3x speedup on NVIDIA GPUs
├── Quantization (INT8) → 4x smaller, ~2x faster
└── Distillation → Train smaller student model
```

**Infrastructure:**
```
Recommended Production Stack:
├── API: FastAPI (async, OpenAPI docs)
├── Serving: Triton Inference Server
├── Container: Docker with NVIDIA runtime
├── Orchestration: Kubernetes with GPU nodes
├── Monitoring: Prometheus + Grafana
└── Logging: ELK stack for predictions
```

**Reliability:**
- Implement request timeout (default: 5s)
- Add input validation (file type, size, dimensions)
- Return uncertainty scores for low-confidence predictions
- Monitor for data drift (input distribution changes)

---

## Key Insights & Conclusions

### What We Learned

1. **Architecture choice is the biggest lever**
   - ViT outperformed CNNs by 11-40% on this task
   - Global attention is crucial for fine-grained discrimination
   - Pretraining data quality/quantity matters (21K vs 1K classes)

2. **2-stage fine-tuning is essential**
   - 48% error reduction over standard fine-tuning
   - Preserves pretrained knowledge while adapting to task
   - Differential learning rates prevent catastrophic forgetting

3. **Ensembles aren't always better**
   - When one model dominates, ensembling hurts
   - Only combine models with complementary strengths
   - Simpler is often better

4. **Explainability builds trust**
   - ViT attention maps confirm model uses relevant features
   - No evidence of shortcut learning
   - Errors are botanically understandable

5. **Error analysis guides improvement**
   - Knowing *why* errors happen enables targeted fixes
   - Some "errors" are actually dataset labeling issues
   - Diminishing returns suggest near-optimal performance

### Future Work

With more time, we would explore:

- [ ] Test-Time Augmentation (TTA) for additional ~1% accuracy
- [ ] MC Dropout for uncertainty quantification
- [ ] Knowledge distillation to MobileNet for edge deployment
- [ ] Gradio/Streamlit demo for stakeholder presentation
- [ ] Multi-flower detection with YOLO preprocessing

---

## Project Structure
```
flowers-cv/
│
├── notebooks/                          # Jupyter notebooks (run in order)
│   ├── 01_EDA.ipynb                   # Exploratory Data Analysis
│   ├── 02_Model_Comparison.ipynb      # Architecture comparison (ResNet, EfficientNet, ViT)
│   ├── 03_Winner_Optimization.ipynb   # 2-stage training on ViT
│   ├── 04_Ensemble_Experiment.ipynb   # Ensemble strategies
│   └── 05_Evaluation_and_Explainability.ipynb  # Metrics, attention viz, error analysis
│
├── src/                               # Source code modules
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py                 # Dataset classes, transforms
│   ├── models/
│   │   ├── __init__.py
│   │   ├── build_model.py             # Model factory
│   │   └── classifier.py              # Production inference class
│   ├── training/
│   │   ├── __init__.py
│   │   ├── engine.py                  # Training loops
│   │   └── callbacks.py               # EarlyStopping, ModelCheckpoint
│   └── utils/
│       ├── __init__.py
│       ├── seed.py                    # Reproducibility
│       └── visualization.py           # Plotting utilities
│
├── artifacts/                         # Generated outputs
│   ├── models/                        # Trained model weights
│   │   ├── vit_2stage_best.pt        # Best model (98.75%)
│   │   ├── vit_comparison.pt         # ViT baseline (97.54%)
│   │   ├── efficientnet_comparison.pt # EfficientNet (86.24%)
│   │   └── resnet50_comparison.pt    # ResNet-50 (58.12%)
│   ├── figures/                       # All visualizations (17 PNGs)
│   └── reports/                       # JSON metrics and summaries
│
├── configs/
│   └── config.yaml                    # Training hyperparameters
│
├── requirements.txt                   # Python dependencies
├── .gitignore
└── README.md                          # This file
```

---

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/Harik7-cmd/Flowers-cv.git
cd Flowers-cv

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Notebooks in Colab

1. Upload to Google Drive
2. Open notebooks in order (01 → 05)
3. Change runtime to GPU (Runtime → Change runtime type → T4 GPU)
4. Run all cells (Runtime → Run all)

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
model.load_state_dict(ckpt["model_state_dict"])
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
    pred_class = probs.argmax().item()
    confidence = probs.max().item()
    
print(f"Predicted class: {pred_class}")
print(f"Confidence: {confidence:.1%}")
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
tqdm>=4.64.0
pyyaml>=6.0
Pillow>=9.0.0
numpy>=1.21.0
```

---

## Acknowledgments

- **Dataset**: [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) by Nilsback and Zisserman
- **Models**: [timm](https://github.com/huggingface/pytorch-image-models) by Ross Wightman
- **Augmentation**: [Albumentations](https://albumentations.ai/)

---

## License

MIT License - See LICENSE file for details

---

## Contact

**Assessment**: AQREIGHT Computer Vision Engineer  
**Date**: January 2025

---

<p align="center">
  <i>Built with 🌸 and PyTorch</i>
</p>
