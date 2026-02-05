# Comprehensive Experiment Report

This document describes the full active learning work implemented in this repository, including:

1) **Baseline training (no active learning)**  
2) **Baseline active learning (no synthetic augmentation)**  
3) **Uncertainty‑conditioned CGAN augmentation + active learning**  
4) **VAE latent interpolation augmentation + active learning**

The focus is on **tabular datasets** (e.g., Iris, Wine, Two‑Moons) and **MNIST** (flattened 28×28), using a shared MLP backbone and a consistent “oracle labeling” setup.

---

## 1. Repository Structure (What Matters)

**Core modules**
- `data/datasets.py`: dataset loading + preprocessing (including MNIST via `torchvision`).
- `models/base_model.py`: `TabularMLP` backbone used for oracle + main model.
- `models/cgan.py`: conditional GAN implementation (categorical + continuous conditioning).
- `config.py`: shared dataclasses (model/training configs).
- `utils.py`: visualization helpers (mesh creation, PCA helper utilities).

**Main runnable experiment scripts**
- `run_baseline_no_al.py`: train-on-half baseline, no acquisitions.
- `run_baseline_al.py`: active learning baseline (no synthetic generation).
- `run_uncertainty_cgan_al.py`: uncertainty‑conditioned CGAN + oracle labeling + AL + visualization.
- `run_vae_interp_al.py`: VAE‑latent interpolation + oracle labeling + AL + visualization.

---

## 2. Datasets Supported

From `data/datasets.py`:
- Classification (tabular): `iris`, `wine`, `breast_cancer`, `two_moons`, `circles`, `synthetic_classification`
- Regression (tabular): `california_housing`, `synthetic_regression`, `boston` (falls back to California Housing if removed)
- Image (flattened): `mnist` (requires `torchvision`)

**MNIST note:** MNIST is loaded from `torchvision.datasets.MNIST`, normalized to `[0,1]`, then flattened to vectors of length `784`. The model is still an MLP (not a CNN).

---

## 3. Shared Experimental Protocol

All three baselines/methods use the same high‑level protocol to make comparisons fair:

### 3.0 Problem Setup (Pool‑based Active Learning)
We follow a standard pool‑based AL setup:
- You have a **training split** `D_train` and a separate **test split** `D_test`.
- From `D_train`, we create:
  - **Labeled set** `D_L` (initially small; grows over iterations),
  - **Pool set** `D_U` (unlabeled from the main model’s perspective; shrinks over iterations).

The goal is to reach strong performance on `D_test` while using as few oracle queries as possible.

### 3.1 Reproducible Half‑Split
We split the *training split* into two halves using a fixed seed:
- **Labeled half**: used to train the main model initially.
- **Pool half**: treated as unlabeled data available for acquisition.

This mimics “limited labeled data” settings.

### 3.2 Oracle Labeler (Hypernetwork as Human Oracle)
Each run trains an **oracle model** on the **full training split** (both halves).  
The oracle has access to ground truth and acts like a human annotator:
- When the active learning algorithm “acquires” a point, the oracle provides its label.
- For synthetic samples, the oracle also produces labels (so synthetic points can be used as labeled training points).

This is implemented as another `TabularMLP` instance trained on all training data.

### 3.3 Uncertainty Estimation (MC Dropout)
Uncertainty is computed with **MC Dropout**:
- Run multiple stochastic forward passes with dropout enabled.
- Classification: compute entropy of mean predictive distribution.
- Regression: use predictive variance proxy.

**Classification entropy (used as uncertainty):**
Let `p_t(y|x)` be the softmax output at MC sample `t` and `T` the number of samples.
We compute:
- `p̄(y|x) = (1/T) Σ_t p_t(y|x)`
- `H(p̄) = - Σ_c p̄_c log(p̄_c + ε)`  (higher = more uncertain)

**Regression variance (proxy):**
Let `μ_t(x)` be the regression output at MC sample `t`. Use:
- `Var(x) = Var_t[ μ_t(x) ]`

**Important:** In all AL scripts, **selection is always from the pool**, never from already‑labeled points.

### 3.4 Visualization Output (all AL scripts)
Each step saves:
- `step_XX.png`: a 2‑panel plot
  - Left: decision boundary / decision regions (2D via PCA if needed)
  - Right: uncertainty heatmap (bright = higher uncertainty)
  - Pool points are color‑coded by their uncertainty
  - Labeled points are shown separately
  - Acquired points are highlighted each step
- `active_learning.gif`: GIF of frames (optional via `--make-gif`)
- `results.json`: final metrics + frame list

For MNIST‑like inputs, extra image grids are saved to show what was generated.

### 3.5 Training Style: Retrain‑From‑Scratch per AL Step
In the AL scripts, after each acquisition step we **retrain the main model from scratch** on the expanded labeled set `D_L`.
This avoids subtle “state leakage” across steps and makes the effect of acquisitions easier to interpret (but costs more compute).

### 3.6 “What Counts as a Query?”
Whenever the algorithm selects a point from `D_U` (real or synthetic) for labeling, we treat that as an oracle query.
Synthetic samples are **not automatically labeled** unless they are (a) labeled by oracle immediately (as in these scripts) or
acquired for labeling (depending on method variant). The implemented runners label synthetic points via the oracle before
adding them to `D_L` or keeping them in `D_U`.

---

## 4. Baseline 1: No Active Learning (Train Once on Half)

**Script:** `run_baseline_no_al.py`

### What it does
1) Half‑split the train set into labeled half + pool half.
2) Train oracle on the full train split (for parity).
3) Train the main model **once** on the labeled half.
4) Produce a single visualization frame `step_00.png` showing:
   - decision boundary/regions
   - uncertainty heatmap
   - pool colored by uncertainty
5) Save `results.json` and an optional single‑frame GIF.

### Run command (example: MNIST)
```bash
python3 run_baseline_no_al.py \
  --dataset mnist \
  --seed 42 \
  --epochs 300 \
  --make-gif \
  --output runs/mnist_baseline_no_al
```

---

## 5. Baseline 2: Active Learning (No Synthetic Augmentation)

**Script:** `run_baseline_al.py`

### What it does
1) Half‑split the train set into labeled half + pool half.
2) Train oracle on full train split.
3) Train main model on labeled half.
4) Repeat for `acq_steps`:
   - Compute uncertainty on pool.
   - Acquire `acq_size` most‑uncertain pool samples.
   - Oracle labels them.
   - Add them to labeled set.
   - Remove them from pool.
   - Retrain main model from scratch on updated labeled set.
   - Save visualization frame (boundary + uncertainty heatmap).

### Run command (example: MNIST)
```bash
python3 run_baseline_al.py \
  --dataset mnist \
  --seed 42 \
  --epochs 300 \
  --acq-steps 10 \
  --acq-size 10 \
  --make-gif \
  --output runs/mnist_baseline_al
```

---

## 6. Method 1: Uncertainty‑Conditioned CGAN + Active Learning

**Script:** `run_uncertainty_cgan_al.py`

This implements the proposal-style loop:

### 6.1 Key idea
Train a generator to create samples that the current main model is uncertain about, then:
- add them to the pool,
- run active learning on the combined pool (real + synthetic),
- label acquired samples using the oracle,
- retrain the main model,
- repeat; re‑train/adjust the generator as the decision boundary evolves.

### 6.2 Conditioning signal
The CGAN is conditioned on the **main model’s uncertainty** (a continuous scalar):
- We compute uncertainty values for points in the combined labeled+pool set.
- We train the conditional generator to model `p(x | uncertainty)` via continuous conditioning.

### 6.2.1 What “CGAN” Means Here (Continuous Conditioning)
This implementation uses a *conditional GAN with continuous condition*:
- Generator: `G(z, u) → x̂` where `z ~ N(0, I)` and `u` is a scalar uncertainty value.
- Discriminator: `D(x, u) → logit(real)` decides if `x` is real or fake **given the same uncertainty condition**.

The condition `u` is **not** a class label in this method. It is the current main‑model uncertainty estimate for the sample.

### 6.2.2 How the Training Pairs are Built
At a given AL step, we build a CGAN training dataset:
- For each training point `x` in the current (labeled + pool) set, compute `u = uncertainty_main(x)`.
- Train the CGAN on pairs `(x, u)`.

This makes the generator learn to produce samples resembling real data *at a specified uncertainty level*.

### 6.3 Per‑iteration loop
For each active learning step:
1) **Recompute uncertainty** under the current main model.
2) **Re-fit the CGAN** using the latest uncertainty conditioning (fine-tuning each iteration).
3) **Generate synthetic candidates** with uncertainty conditions sampled in the **80th–100th percentile range** of current uncertainty values.
   - This prevents conditioning on only a single value and improves diversity.
4) **Filter synthetic candidates** by *actual uncertainty under the main model*:
   - oversample (generate more than needed),
   - compute `uncertainty_main(x̂)` on generated points,
   - keep the top `synthetic_count` most‑uncertain generated samples.
5) **Oracle labels synthetic samples** and they are added to the pool so the next acquisition can choose from (real + synthetic).
6) **Standard AL acquisition** from the combined pool (real + synthetic):
   - compute pool uncertainty,
   - select top‑`acq_size`,
   - oracle labels them,
   - add to `D_L`, remove from pool,
   - retrain main model.

### 6.3.1 Pseudocode (High‑Level)
```
Split D_train into D_L (half) and D_U (half)
Train oracle on D_train
Train main model on D_L

for step in 1..K:
    u_all = uncertainty_main(D_L ∪ D_U)
    train CGAN on pairs (x, u_all(x)) for x in (D_L ∪ D_U)

    # generate synthetic near decision boundary
    u_target ~ Uniform(Percentile80(u_all), Percentile100(u_all))
    X_syn_candidates = { G(z, u_target) }
    keep top uncertain X_syn by uncertainty_main(X_syn_candidates)
    y_syn = oracle(X_syn)
    D_U = D_U ∪ {(X_syn, y_syn)}   # available for acquisition

    # standard active learning acquisition
    acquire top uncertain from D_U
    oracle labels acquired points
    move acquired points from D_U → D_L
    retrain main model on D_L
```

### 6.4 Visualization details
Markers:
- Pool points: colored by uncertainty (magma)
- Labeled points: blue
- Acquired points (this step): red circles
- Synthetic points remaining in pool: black `×`
- Synthetic points acquired this step: green diamonds

### 6.5 MNIST generated sample dumps
If input dimension is `784`, the script saves grids per step under:
`<output>/synthetic_samples/`

It writes:
- generated uncertain samples
- sampled certain real images and uncertain real images for comparison

### Run command (example: MNIST)
```bash
python3 run_uncertainty_cgan_al.py \
  --dataset mnist \
  --seed 42 \
  --epochs 300 \
  --cgan-steps 10000 \
  --synthetic-count 20 \
  --acq-steps 10 \
  --acq-size 10 \
  --make-gif \
  --output runs/mnist_uncgan_al
```

---

## 7. Method 2: VAE Latent‑Interpolation + Active Learning

**Script:** `run_vae_interp_al.py`

### 7.1 Key idea
Instead of training a GAN, train a **VAE** on all training data.  
To generate “near boundary” samples:
1) identify uncertain real pool points,
2) encode them into latent space,
3) interpolate between latent points with many weights,
4) decode interpolations to produce synthetic samples,
5) filter and then treat them exactly like pool data for AL acquisition.

### 7.2 What “VAE Latent Interpolation” Means
A VAE learns a continuous latent representation `z` of data `x`:
- Encoder: `qϕ(z|x)` outputs parameters `(μ(x), logσ²(x))`.
- Decoder: `pθ(x|z)` maps latents back to data space.

Training minimizes a reconstruction + KL objective (ELBO):
- `L = E_q[ ||x - x̂||² ] + β * KL(qϕ(z|x) || N(0, I))`

Once trained, we can:
- encode real uncertain samples into latent space,
- create new latents by interpolation,
- decode those latents to synthetic samples expected to lie between the originals in data manifold terms.

### 7.2 VAE training
The VAE is trained on the **full train split** using reconstruction + KL loss (β‑VAE supported via `--vae-beta`).

### 7.3 How “uncertain sources” are chosen
At each step we compute uncertainty over the pool and take the **top 20%** as candidates for interpolation.
This is independent of the AL acquisition step (AL still selects top uncertain from the pool afterwards).

### 7.4 Diversity controls (important)
To avoid generating the same image repeatedly, the implementation:
- Samples latent `z` from the **posterior**, not just mean `mu`.
- Forces `idx1` to cover the candidate set uniformly (not always the same point).
- Uses **Beta(0.5, 0.5)** interpolation weights (not fixed 0.5), plus small latent noise.
- After generating many candidates, selects a **diverse** subset via farthest‑point sampling in latent space among the high‑uncertainty points.

### 7.4.1 Why Filtering by Main‑Model Uncertainty Still Matters
Interpolating between uncertain points does not guarantee every decoded sample is uncertain.
Therefore after decoding, we:
1) evaluate uncertainty under the current main model,
2) keep only the most uncertain decoded samples,
3) enforce diversity so the kept samples do not collapse to one mode.

### 7.5 MNIST visualization outputs
For MNIST, it saves:
- `synthetic_samples/step_XX_generated.png` (generated uncertain)
- `synthetic_samples/step_XX_real_uncertain.png` (real uncertain sampled)
- `synthetic_samples/step_XX_real_certain.png` (real certain sampled)

### Run command (example: MNIST)
```bash
python3 run_vae_interp_al.py \
  --dataset mnist \
  --seed 42 \
  --epochs 30 \
  --vae-epochs 10 \
  --vae-latent-dim 32 \
  --synthetic-count 200 \
  --acq-steps 5 \
  --acq-size 20 \
  --make-gif \
  --output runs/mnist_vae_interp_al
```

---

## 8. Dependencies / Installation

Install requirements:
```bash
pip install -r requirements.txt
```

Key packages:
- `torch`, `torchvision` (MNIST)
- `numpy`, `scikit-learn`
- `matplotlib`, `imageio` (plots + GIF)

---

## 9. Output Artifacts (What to Expect)

For each run directory (e.g., `runs/mnist_uncgan_al/`):
- `results.json`: includes dataset, seed, final metrics, and list of frames.
- `step_XX.png`: frames (boundary + uncertainty).
- `active_learning.gif`: created if `--make-gif`.
- `synthetic_samples/`: for MNIST-like inputs, saved image grids to inspect generation quality.

---

## 10. Notes / Limitations

- The backbone for MNIST is an MLP on flattened pixels, not a CNN. Generated samples may look blurrier and decision boundaries are visualized via PCA projection.
- PCA‑based visualization for high‑dimensional datasets is approximate; it is intended for qualitative debugging, not strict geometric truth.
- With MC Dropout, uncertainty heatmaps often align with decision boundaries (expected behavior for entropy-based uncertainty).
