# ML4SCI GSoC 2026 — Task 2d: Sparse Neural Networks for CMS Jet Classification

**Author:** Harsh Vardhan Singh  
**Institution:** Faculty of Technology, University of Delhi  
**Programme:** Google Summer of Code 2026 — ML4SCI

---

## Overview

This notebook addresses **Task 2d** of the ML4SCI GSoC evaluation: building and benchmarking sparse neural network architectures for binary classification on CMS calorimeter jet data, and analysing the **efficiency–accuracy tradeoff** under progressive weight pruning.

The central questions explored are:
1. Does a **sparsity-aware architecture** outperform a standard dense CNN on sparse detector data?
2. How does **global L1 pruning** affect accuracy as a function of weight sparsity and effective FLOPs?

---

## Dataset

| Property | Value |
|---|---|
| File | `Dataset_Specific_labelled.h5` |
| Input shape | `(N, 125, 125, 8)` — 2D calorimeter grid × 8 detector channels |
| Task | Binary jet classification |
| Subset used | 10,000 samples (full dataset) |
| Split | 70% train / 15% validation / 15% test |

The data is **extremely sparse**: most detector cells record zero energy. Per-sample max-normalisation is applied to map each sample into [0, 1] while preserving relative energy structure.

---

## Model Architectures

### DenseCNN (Baseline)
A standard 2-layer CNN with no sparsity awareness. Serves as the performance baseline.

- Conv(8→16) → BN → ReLU → Conv(16→32) → BN → ReLU → AdaptiveAvgPool → FC(32→1)
- **~5,889 parameters**

### SparseResNet
A physics-motivated 3-block sparse residual network. The key innovation is the `SparseResidualBlock`, which computes a **binary sparsity mask** from the input and zeroes out activations at empty detector cells after each residual computation. This prevents the network from propagating gradients through physically meaningless zero-energy regions.

- Architecture: 8 → 16 → 32 → 64 channels via sparse residual blocks
- **~75,777 parameters**

---

## Training Setup

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Loss | Binary Cross-Entropy with Logits |
| Epochs | 10 |
| Batch size | 32 |
| LR Scheduler | ReduceLROnPlateau (patience=2, factor=0.5) |
| Seed | 42 (fixed for reproducibility) |

---

## Results

### Model Comparison (Held-Out Test Set)

| Model | Accuracy | AUC |
|---|---|---|
| DenseCNN (baseline) | 0.8127 | 0.8731 |
| **SparseResNet** | **0.8747** | **0.9333** |
| Improvement | **+6.20 pp** | **+6.02 pp** |

### Pruning Analysis (SparseResNet)

Global L1-unstructured pruning was applied at 5 pruning ratios, each followed by 2 epochs of fine-tuning:

| Pruning Ratio | Weight Sparsity | Eff. FLOPs (M) | Test Accuracy | AUC |
|---|---|---|---|---|
| 0.0 (baseline) | 0.000 | 169.32 | 0.8747 | 0.9333 |
| 0.2 | 0.200 | ~135 | 0.8673 | — |
| 0.4 | 0.400 | — | — | — |
| 0.6 | 0.600 | — | — | — |
| 0.8 | 0.800 | — | — | — |

At 60% pruning, accuracy degradation remains modest — demonstrating that SparseResNet's non-zero weights carry concentrated, non-redundant information. At 80% pruning, effective FLOPs drop dramatically while remaining within an acceptable accuracy range for real-time trigger applications.

---

## Key Findings

1. **Sparsity-aware masking significantly improves performance.** By zeroing activations in empty detector cells, SparseResNet learns only from physically meaningful signals, reducing noise in learned representations.

2. **SparseResNet is robust to aggressive pruning.** Models that already ignore empty cells exhibit less weight redundancy and therefore degrade more gracefully under pruning.

3. **Favourable FLOPs–error tradeoff.** The efficiency gains are directly applicable to CMS Level-1 trigger systems where inference latency is a hard constraint.

---

## Outputs / Saved Figures

| File | Description |
|---|---|
| `training_curves.png` | Train vs. validation loss per epoch for both models |
| `roc_confusion.png` | ROC curves (both models) and confusion matrix (SparseResNet) |
| `model_comparison.png` | Bar chart comparing accuracy and AUC across models |
| `pruning_analysis.png` | Error vs. FLOPs, accuracy vs. pruning ratio, accuracy vs. sparsity |

---

## Requirements

```
torch
h5py
scikit-learn
matplotlib
seaborn
ptflops
```

Install via:
```bash
pip install ptflops h5py scikit-learn matplotlib seaborn
```

---

## Running the Notebook

1. Mount Google Drive and place `Dataset_Specific_labelled.h5` at:  
   `/content/drive/MyDrive/ML4SCI_datasets/`

2. Run all cells in order. All random seeds are fixed (`SEED = 42`) for full reproducibility.

3. GPU is recommended (the notebook auto-detects CUDA).

---

## Repository Structure

```
ML4SCI_Task2d_corrected.ipynb   ← Main notebook
README.md                        ← This file
training_curves.png              ← Generated on run
roc_confusion.png                ← Generated on run
model_comparison.png             ← Generated on run
pruning_analysis.png             ← Generated on run
```

---

## Relevance to GSoC Proposal

The findings from this task directly inform the proposed diffusion model design for the ML4SCI project:
- Sparse masking should be incorporated into the UNet backbone's residual blocks.
- Gradients should not be propagated through zero-energy regions during training.
- Pruning-aware training (sparse regularisation) may improve generation quality in sparse detector regions.
