# Active Learning With Uncertainty-Aware Generation

End-to-end experiments for uncertainty-driven active learning on tabular and vision datasets. The repo bundles three pipelines—baseline active learning, uncertainty-conditioned CGAN augmentation, and VAE latent interpolation—plus utilities for uncertainty estimation, visualization, and large ablation sweeps.

## Repository Map
- `run_baseline_no_al.py` — single-pass training on an initial labeled split; logs metrics and visualizes decision regions/uncertainty.
- `run_baseline_al.py` — pool-based active learning loop (no generation); acquires top-uncertainty points each step.
- `run_uncertainty_cgan_al.py` — active learning with a CGAN conditioned on model uncertainty; generates and labels synthetic pool points.
- `run_vae_interp_al.py` — active learning with VAE latent interpolation to propose diverse high-uncertainty samples.
- `run_ablations.py` — grid runner that shells out to the above scripts and writes a consolidated `RESULTS.md`.
- `data/datasets.py` — dataset loader for tabular baselines, synthetic shapes, and MNIST/Fashion-MNIST/CIFAR10.
- `models/` — tabular MLPs and conditional GAN components.
- `uncertainty_methods.py` — MC-dropout, deep-ensemble, and Laplace estimators shared by runners.
- `trainer.py`, `config.py`, `utils.py` — training helpers, dataclasses for configs, and plotting utilities.

## Setup
1) Create and activate a Python environment (Python 3.9+ recommended).
2) Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3) (Optional) Ensure CUDA PyTorch is installed if you want GPU acceleration.

## Supported Datasets
Pass any of these to `--dataset`:
- Tabular: `iris`, `wine`, `breast_cancer`, `boston` (falls back to California Housing if unavailable), `california_housing`
- Synthetic: `synthetic_classification`, `synthetic_regression`, `two_moons`, `circles`
- Vision (requires `torchvision`): `mnist`, `fashion_mnist`, `cifar10`

## Running Experiments
All runners write a `results.json` plus per-step frames to the folder given by `--output`. Add `--make-gif` to also produce `active_learning.gif`.

### Baseline (no active learning)
Train once on the initial labeled split and visualize uncertainty over the remaining pool:
```bash
python run_baseline_no_al.py --dataset iris --output runs/iris_no_al --initial-label-frac 0.5 \
  --epochs 30 --uncertainty dropout --mc-samples 20 --make-gif
```

### Baseline Active Learning
Pool-based acquisitions without generation:
```bash
python run_baseline_al.py --dataset iris --output runs/iris_al --acq-steps 5 --acq-size 20 \
  --epochs 30 --uncertainty laplace --laplace-prior 1.0 --laplace-samples 30 --make-gif
```

### Uncertainty-Conditioned CGAN + Active Learning
Generates samples conditioned on model uncertainty, labels them with an oracle, and acquires the most uncertain (real + synthetic):
```bash
python run_uncertainty_cgan_al.py --dataset iris --output runs/iris_uncgan \
  --cgan-steps 200 --synthetic-count 200 --acq-steps 5 --acq-size 20 \
  --epochs 30 --uncertainty ensemble --ensemble-size 5 --make-gif
```

### VAE Latent Interpolation + Active Learning
Trains a VAE on the full train split, interpolates uncertain latents to propose new samples, and runs AL:
```bash
python run_vae_interp_al.py --dataset mnist --output runs/mnist_vae_al \
  --vae-epochs 10 --vae-latent-dim 32 --synthetic-count 200 \
  --acq-steps 5 --acq-size 20 --epochs 30 --uncertainty dropout --make-gif
```

### Ablation Sweeps
Launch a grid of runs and collate metrics into `RESULTS.md`:
```bash
python run_ablations.py --datasets iris,wine,two_moons \
  --methods no_al,baseline_al,uncertainty_cgan,vae \
  --uncertainties dropout,ensemble,laplace \
  --label-fracs 0.8,0.5,0.3 --seeds 42 --out-root ablations_out --make-gif --resume
```

## Key Options (common across runners)
- `--initial-label-frac` — fraction of the train split treated as labeled at start.
- `--epochs` — training epochs per cycle for the main/oracle models.
- Uncertainty estimation: `--uncertainty` in `{dropout, ensemble, laplace}` with `--mc-samples`, `--ensemble-size`, and `--laplace-prior/--laplace-samples` controlling each.
- Acquisition: `--acq-steps`, `--acq-size` for AL loops.
- Generation-specific:
  - CGAN: `--cgan-steps`, `--synthetic-count`
  - VAE: `--vae-epochs`, `--vae-latent-dim`, `--vae-h1/--vae-h2`, `--vae-beta`, `--synthetic-count`

## Outputs
- `results.json` — dataset, uncertainty method, metrics (accuracy/F1/NLL or RMSE/MAE/MSE), and frame paths.
- `step_XX.png` — decision boundary + uncertainty heatmap per AL step; `active_learning.gif` if requested.
- `synthetic_samples/` — generated grids for vision datasets (e.g., MNIST) in CGAN/VAE runs.
- `RESULTS.md` — produced by `run_ablations.py` summarizing all collected runs.

## Tips
- GPU is optional; CPU works for tabular runs but vision datasets benefit from CUDA.
- For quick debugging, use `--max-train-samples` and `--max-test-samples` (or `--fast` in `run_ablations.py`) to cap dataset sizes.
- All scripts are deterministic given `--seed`.
