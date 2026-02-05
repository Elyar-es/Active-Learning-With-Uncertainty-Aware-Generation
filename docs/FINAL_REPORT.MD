# Final Report — Active Learning with CGAN and VAE Augmentation

Command that produced the ablation results (already run by you):
```
python3 run_ablations.py \
  --out-root ablations_out_2 \
  --methods no_al,baseline_al,uncertainty_cgan,vae \
  --uncertainties dropout,ensemble,laplace \
  --label-fracs 0.8,0.5,0.3,0.1 \
  --seeds 42 \
  --epochs 50 --acq-steps 2 --acq-size 20 \
  --cgan-steps 10 --synthetic-count 30 \
  --vae-epochs 50 --ensemble-size 3 --mc-samples 10 --laplace-samples 15
```
Results pulled from `ablations_out_2/**/results.json` (no re-runs).

## Methods Compared
- **NO-AL** (`run_baseline_no_al.py`): Train once on the initial labeled split; no acquisitions.
- **Baseline-AL** (`run_baseline_al.py`): Standard active learning loop; acquire most uncertain pool points.
- **Unc-CGAN-AL** (`run_uncertainty_cgan_al.py`): CGAN conditioned on model uncertainty, generates high-uncertainty samples, oracle-labels them, then active learning acquires uncertain points (real + synthetic).
- **VAE-AL** (`run_vae_interp_al.py`): VAE trained on full train; generates boundary samples by latent interpolation between uncertain pool points; oracle labels them; active learning acquires uncertain points (real + generated).

Uncertainty estimators swept: MC Dropout, Deep Ensemble, Laplace last-layer approximation. Initial labeled fractions swept: 0.8, 0.5, 0.3, 0.1. Best-for-each-cell values below are the best metric per dataset aggregated over those sweeps.

## Best Per-Dataset Winner (metric = accuracy for classification, RMSE for regression)
- **MNIST**: VAE-AL (ensemble, init=0.8), accuracy **0.9787**
- **Fashion-MNIST**: VAE-AL (ensemble, init=0.8), accuracy **0.9073**
- **CIFAR-10**: VAE-AL (ensemble, init=0.8), accuracy **0.5674**
- **Breast Cancer**: Tie (all methods), accuracy **0.9737**
- **Wine**: Unc-CGAN-AL (ensemble, init=0.8), accuracy **1.0000** (baseline/no-AL also hit 1.0)
- **Iris**: VAE-AL (ensemble, init=0.5), accuracy **0.9667** (Unc-CGAN-AL tie at 0.9667 with dropout, init=0.5)
- **Two Moons**: VAE-AL (laplace, init=0.3), accuracy **0.9950**
- **Circles**: Tie (all methods reach 1.0000). Example: Unc-CGAN-AL (ensemble, init=0.8) or VAE-AL (laplace, init=0.5)
- **Boston/California Housing (regression)**: VAE-AL (dropout, init=0.8), RMSE **0.5337** (lower is better)

## Average of Best-Per-Dataset Scores (classification datasets only; higher is better)
- NO-AL: **0.8914**
- Baseline-AL: **0.9037**
- Unc-CGAN-AL: **0.9122**
- VAE-AL: **0.9201**  ← highest average best-accuracy across classification datasets

Regression (only Boston/California Housing in this sweep): best RMSE
- NO-AL: 0.5511
- Baseline-AL: 0.5578
- Unc-CGAN-AL: 0.5579
- VAE-AL: **0.5337**  ← lowest RMSE

## Interpretation
- **VAE-AL leads overall**: It wins MNIST, Fashion-MNIST, CIFAR-10, Two Moons, and the regression task, and ties on Circles/Breast Cancer. Its average best accuracy across classification datasets is the highest.
- **Unc-CGAN-AL is competitive**: It matches the top iris score, wins wine (though several methods also hit 1.0), and ties on circles/breast_cancer. Averages sit between Baseline-AL and VAE-AL.
- **Active learning helps modestly**: Baseline-AL beats NO-AL on average, but synthetic augmentation (VAE/CGAN) yields the larger gains.
- **Uncertainty estimators**: Deep ensembles were the most frequent winner on high-dimensional image datasets; Laplace occasionally topped 2D toy datasets; Dropout worked best for the regression case.
- **Label fraction sensitivity**: Many best runs occurred at higher initial label fractions (0.8) for images, but Two Moons favored 0.3, indicating some benefit from iterative acquisition when the initial set is smaller.

## Notes and Caveats
- Circles, Wine, and Breast Cancer exhibit ceiling effects (multiple methods at 1.0 accuracy); “winner” there is effectively a tie.
- CIFAR-10 accuracies are low because models are simple MLPs on flattened inputs; results compare methods fairly within this constrained setup, not against modern CNN baselines.
- All numbers above are taken directly from `ablations_out_2/**/results.json`; no additional training was run.

## How to Regenerate the Summary Only (no reruns)
```
python3 run_ablations.py --out-root ablations_out_2 --resume --results-md RESULTS.md
```
This re-aggregates existing results and rewrites `RESULTS.md` without executing new trials.
