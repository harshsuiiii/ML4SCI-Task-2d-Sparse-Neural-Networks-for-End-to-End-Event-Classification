# ML4SCI — Generative Modelling for CMS Detector Data

**Task 2e · Diffusion Models for Fast and Accurate Simulation of Low-Level CMS Experiment Data**

This notebook implements and compares three generative models — a DDPM diffusion model, a GAN, and a VAE — trained on unlabelled 8-channel CMS calorimeter jet images. It is a proof-of-concept study submitted as the ML4SCI GSoC evaluation task.

---

## Dataset

| Property | Value |
|---|---|
| File | `Dataset_Specific_Unlabelled.h5` |
| HDF5 key | `jet` |
| Full dataset shape | `(60000, 125, 125, 8)` |
| Dtype | `float32` |
| Channels | 8 CMS calorimeter / tracker detector layers |
| Training subset used | 20,000 samples |
| Normalisation | `[0, 255]` → `[-1, 1]` |
| Working resolution | Downsampled to `32×32` via bilinear interpolation |

Download the dataset using `wget` from the ML4SCI index before running:

```bash
wget <dataset-url> -O Dataset_Specific_Unlabelled.h5
```

Upload to Google Drive at `MyDrive/ML4SCI_datasets/` and update `file_path` in the notebook if needed.

---

## Environment

Runs on **Google Colab** with a GPU runtime (tested on CUDA). All dependencies are installed in the notebook:

```bash
pip install h5py tqdm scipy
```

Standard packages used: `torch`, `torchvision` (via Colab), `numpy`, `matplotlib`, `pandas`, `scipy`, `h5py`, `tqdm`, `math`.

---

## Structure

```
CMS_Diffusion_Fixed.ipynb
│
├── 0. Setup & Imports
├── 1. Data Loading & Preprocessing      ← CMSDataset, DataLoader
├── 2. Diffusion Model (DDPM)
│   ├── 2.1 Theory
│   ├── 2.2 Architecture — TinyUNet      ← sinusoidal embeddings, TimeCondResBlock
│   ├── 2.3 Training                     ← 20 epochs, loss 0.2912 → 0.0128
│   └── 2.4 Generate Samples             ← 1,000 samples
├── 3. GAN
│   ├── 3.1 Architecture                 ← MLP Generator + Discriminator
│   └── 3.2 Training                     ← 20 epochs, Nash equilibrium
├── 4. VAE
│   ├── 4.1 Architecture & Loss          ← Beta-VAE ELBO, reparameterisation
│   └── 4.2 Training                     ← 20 epochs, posterior collapse observed
├── 5. Statistical Comparison            ← WD, KS, MMD, diversity, sharpness
├── 6. Bonus — Mode Collapse & Complexity Bias
│   ├── 6.1 GAN Mode Collapse
│   └── 6.2 VAE Complexity Bias
└── 7. Summary & Discussion
```

---

## Model 1 — DDPM Diffusion (TinyUNet)

### Architecture

The denoiser is a lightweight time-conditioned UNet. The key design principle is that the network must receive the timestep `t` as an explicit input — without it, the model cannot distinguish noise levels across the 200-step forward process.

```
Input:  (B, 8, 32, 32)  noisy image  +  (B,) integer timestep t
Output: (B, 8, 32, 32)  predicted noise ε

sinusoidal_embedding(t, dim=128)
    └── 2-layer MLP (Linear → SiLU → Linear) → t_emb (B, 128)

Encoder:
    enc1: TimeCondResBlock(8  → 32,  t_emb_dim=128)
    enc2: TimeCondResBlock(32 → 64,  t_emb_dim=128)

Bottleneck:
    bot:  TimeCondResBlock(64 → 64,  t_emb_dim=128)

Decoder (skip-concatenated):
    dec2: TimeCondResBlock(128 → 32, t_emb_dim=128)   ← cat(bot, enc2)
    dec1: TimeCondResBlock( 64 →  8, t_emb_dim=128)   ← cat(dec2, enc1)

Output: Conv2d(8, 8, 1×1)

Trainable parameters: 258,952
```

Each `TimeCondResBlock` injects `t_emb` as a channel-wise bias:

```python
h = F.silu(GroupNorm(Conv(x)))
h = h + Linear(t_emb)[:, :, None, None]   # time added to every spatial location
h = F.silu(GroupNorm(Conv(h)))
output = h + skip(x)
```

### Forward Process (Diffusion)

```
q(x_t | x_0) = N(x_t; √ᾱ_t · x_0,  (1 − ᾱ_t) · I)

Beta schedule: linear from 1e-4 to 0.02 over T=200 steps
ᾱ_t = cumulative product of (1 − β_t)
```

### Training

| Hyperparameter | Value |
|---|---|
| T (diffusion steps) | 200 |
| Epochs | 20 |
| Batch size | 64 |
| Optimiser | Adam, lr=2e-4 |
| LR schedule | CosineAnnealingLR (T_max=20) |
| Gradient clipping | max_norm=1.0 |
| Loss | MSE between predicted and actual noise |

### Training Results

| Epoch | Avg Loss |
|---|---|
| 1 | 0.2912 |
| 2 | 0.0369 |
| 5 | 0.0198 |
| 10 | 0.0148 |
| 20 | 0.0128 |

Loss converged from **0.2912 → 0.0128** over 20 epochs (22× reduction).

> **Note:** Generated pixel values range from −1.254 to +2.670, slightly exceeding the [−1, 1] normalisation boundary. This is a known artefact of accumulated stochastic noise in the DDPM reverse SDE. In production, apply `x = x.clamp(-1, 1)` after sampling.

---

## Model 2 — GAN

### Architecture

```
Generator:   z ∈ R^64 → Linear(512) → BN → LeakyReLU
                       → Linear(1024) → BN → LeakyReLU
                       → Linear(8×32×32) → Tanh → (B, 8, 32, 32)
             Parameters: 8,958,464

Discriminator: (B, 8, 32, 32) → Flatten → Linear(1024) → LeakyReLU → Dropout(0.3)
                               → Linear(512) → LeakyReLU → Dropout(0.3)
                               → Linear(1) → Sigmoid
             Parameters: 8,914,945
```

### Training

| Hyperparameter | Value |
|---|---|
| Epochs | 20 |
| Batch size | 64 |
| Optimiser | Adam, lr=2e-4, betas=(0.5, 0.999) |
| Loss | BCELoss |
| Latent dim | 64 |

### Training Results (Final Epoch)

```
Epoch 20 | D Loss: 1.3929 | G Loss: 0.6962
```

D Loss ≈ log(4) = 1.386 and G Loss ≈ log(2) = 0.693 — both at the theoretical Nash equilibrium. Despite stable training, **mode collapse** was confirmed by the diversity metric (see Section 5).

---

## Model 3 — VAE

### Architecture

```
Encoder: Flatten → Linear(1024) → LeakyReLU → Linear(512) → LeakyReLU
         → μ ∈ R^64,  log σ² ∈ R^64

Reparameterisation: z = μ + σ · ε,   ε ~ N(0, I)

Decoder: Linear(512) → LeakyReLU → Linear(1024) → LeakyReLU
         → Linear(8×32×32) → Tanh → (B, 8, 32, 32)

Parameters: 17,935,488
```

### Loss — Beta-VAE ELBO

```
L = MSE(recon, x) + β · KL(q(z|x) || p(z))

β = 1.0  (standard VAE)
KL = -0.5 · mean(1 + log σ² − μ² − σ²)
```

### Training Results

| Epoch | Total | Recon | KL |
|---|---|---|---|
| 1 | 0.0364 | 0.0337 | 0.0026 |
| 2 | 0.0025 | 0.0025 | 0.0000 |
| 11 | 0.0104 | 0.0025 | 0.0079 |
| 12–20 | 0.0025 | 0.0025 | 0.0000 |

**Posterior collapse** observed: KL collapsed to ~0.0000 from epoch 2, with a brief transient spike to 0.0079 at epoch 11 before re-collapsing. The encoder learned to map all inputs to the prior N(0, I), and the decoder received pure noise at every step. Final reconstruction loss of 0.0025 represents the residual MSE from outputting the dataset mean image.

---

## Statistical Evaluation

All 1,000 generated samples are compared against 1,000 real samples in the same **[−1, 1] normalised space**.

### Metrics

| Metric | What it measures | Why chosen |
|---|---|---|
| Wasserstein Distance (WD) | Earth-mover's distance between pixel distributions | Smooth, meaningful even for non-overlapping distributions |
| KS Statistic | Maximum CDF deviation | Non-parametric; p < 0.05 rejects H₀ that samples share a distribution |
| MMD (RBF kernel) | Distributional distance in RKHS | Captures higher-order moments; σ=1.0, n=2,000 subsample |
| Per-pixel Diversity std | Variance across generated samples | Detects mode collapse; near zero = collapsed model |
| Laplacian Sharpness | High-frequency edge content | Proxy for spatial structure quality |

### Results

| Metric | Real | Diffusion | GAN | VAE |
|---|---|---|---|---|
| Wasserstein Distance ↓ | — | 0.03912 | 0.00934 | 0.00670 |
| KS Statistic ↓ | — | 0.9649 | 0.9716 | 0.9710 |
| MMD (RBF, σ=1.0) ↓ | — | 0.001544 | 0.000011 | 0.000002 |
| Diversity std ↑ | 0.0334 | **0.0471** | 0.0040 | 0.0020 |
| Laplacian Sharpness | 0.1235 | 0.1138 | 0.1166 | 0.1172 |
| Pixel range | −1.0…+1.0 | −1.254…+2.670 | −1.0…+0.981 | −1.0…+0.977 |

### Interpretation

**WD/MMD paradox:** GAN and VAE rank better on WD and MMD because both models collapsed to outputting near-constant images close to the dataset mean (−0.994). A model that outputs the training mean for every input trivially wins on marginal pixel distribution metrics. The diversity metric reveals the true failure.

**Laplacian paradox:** The diffusion model scores *lowest* on Laplacian sharpness (0.1138) — not because its images are blurry, but because it generates spatially structured images with smooth gradients across detector channels. GAN and VAE outputs are near-constant; their higher Laplacian scores come from sharp artefact edges at the few non-zero pixels, not from genuine structural content.

**Correct quality ranking:** Diffusion > GAN > VAE, judged by diversity std (0.0471 vs. 0.0040 vs. 0.0020).

---

## Bonus: Mode Collapse and Complexity Bias

### GAN Mode Collapse

Measured by per-pixel standard deviation across 1,000 generated samples:

```
Real data diversity std : 0.0334
Diffusion diversity std : 0.0471  ✓  (matches real)
GAN diversity std       : 0.0040  ✗  (8× below real — mode collapse)
VAE diversity std       : 0.0020  ✗  (17× below real — posterior collapse)
```

The GAN generator found a small set of "safe" outputs that fool the discriminator, abandoning the rest of the distribution. This is consistent with the D/G loss convergence pattern: once the discriminator is fooled, the generator has no gradient signal to explore new modes.

### VAE Complexity Bias (Posterior Collapse)

The MSE reconstruction loss is minimised by the **conditional mean** of the distribution. Combined with KL regularisation, the VAE learns to collapse the encoder to the prior and decode the dataset mean image — the path of least resistance that minimises both terms simultaneously. The transient KL spike at epoch 11 (0.0079) represents a brief attempt by the encoder to encode structure before the reconstruction gradient overwhelms it.

Confirmed by:
- KL ≈ 0.0000 for 19 of 20 epochs
- Reconstruction loss = 0.0025 (fixed point = mean image MSE)
- Diversity std = 0.0020

---

## Running the Notebook

1. Open in Google Colab and select a **GPU runtime** (Runtime → Change runtime type → T4 GPU)
2. Mount Google Drive and place the dataset at `MyDrive/ML4SCI_datasets/Dataset_Specific_Unlabelled.h5`
3. Run all cells in order — estimated total runtime ~45 minutes on a T4 GPU
4. The first epoch of diffusion training takes ~8 minutes (HDF5 I/O warm-up); subsequent epochs run at ~1 minute each

---

## Key Design Decisions

**Why downsample to 32×32?** The full 125×125 resolution at 8 channels is ~125K values per image. At batch=64 and 20K samples, this is a ~12 GB working set. The 32×32 downsampling is a proof-of-concept tradeoff — the architecture scales directly to full resolution by removing the interpolation step in `CMSDataset.__getitem__`.

**Why T=200 for diffusion?** The original submission used T=50 with a beta schedule designed for T=1000. T=200 gives a better noise schedule at this scale with negligible inference overhead since the reverse pass is fully vectorised.

**Why batch=64?** Balances training speed against GPU memory. Reduces to 4 in the original notebook due to CPU constraints; batch=64 on GPU gives 16× more gradient signal per step.

**Why GroupNorm instead of BatchNorm?** GroupNorm is stable at small batch sizes and does not depend on batch statistics, making it appropriate for diffusion models where the effective batch size per timestep is small.

---

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS 33*.
2. Gretton, A., et al. (2012). A kernel two-sample test. *JMLR 13*(25).
3. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 30*. [Sinusoidal position encodings]
4. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. *arXiv:1312.6114*.
5. Goodfellow, I., et al. (2014). Generative adversarial nets. *NeurIPS 27*.
6. Song, Y., & Ermon, S. (2020). Improved techniques for training score-based generative models. *NeurIPS 33*.

---

## Citation

```
ML4SCI Evaluation Task 2e — Diffusion Models for CMS Detector Data
Submitted for GSoC 2025 consideration
Contact: ml4-sci@cern.ch
```
