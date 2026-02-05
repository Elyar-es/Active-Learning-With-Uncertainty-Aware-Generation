## Uncertainty-Conditioned CGAN Active Learning & Baseline

This report summarizes the implemented pipelines, configuration, and how to run them. Two main scripts are provided:
- `run_uncertainty_cgan_al.py`: Active learning with an uncertainty-conditioned CGAN, oracle labeling, and visualization.
- `run_baseline_al.py`: Baseline active learning without CGAN augmentation, with the same visualization format.

### Core Components
- **Main model (TabularMLP)**: Trained on a labeled subset; retrained each AL step.
- **Oracle**: A model trained on the full training set; provides labels for acquired points and synthetic samples.
- **Uncertainty estimator**: MC Dropout-based entropy/variance from the main model.
- **CGAN** (synthetic pipeline only): Conditioned on the main model’s uncertainty; re-fit each AL step; generates high-uncertainty samples filtered by the current main-model uncertainty.
- **Active Learning Loop**: Selects top-uncertainty points (real + synthetic) from the pool, labels via oracle, adds to labeled set, retrains main model.
- **Visualization**: Per-step frames and optional GIF showing decision boundary and uncertainty heatmap. Pool points are color-coded by uncertainty; red circles = acquired that step; black × = synthetic still in pool; green diamonds = synthetic acquired; blue dots = labeled.

### Running the CGAN-augmented pipeline
Example (iris):
```
python3 run_uncertainty_cgan_al.py \
  --dataset iris \
  --seed 42 \
  --epochs 300 \
  --cgan-steps 10000 \
  --synthetic-count 20 \
  --acq-steps 10 \
  --acq-size 10 \
  --make-gif \
  --output runs/iris_uncgan_al
```
Key args:
- `--dataset`: iris, wine, breast_cancer, two_moons, circles, california_housing, etc.
- `--epochs`: epochs per training cycle for main/oracle.
- `--cgan-steps`: CGAN training steps per AL round.
- `--synthetic-count`: number of synthetic samples to keep per round (after filtering to most uncertain).
- `--acq-steps`, `--acq-size`: AL iterations and acquisitions per step.
- `--make-gif`: save GIF alongside frames.
- `--output`: destination folder (frames, GIF, results.json).

Behavior per step:
1) Compute main-model uncertainty over labeled+pool; fit CGAN conditioned on that uncertainty.
2) Generate synthetic samples across the 80–100th percentile uncertainty range; filter to the most uncertain according to the current main model; label synthetics via oracle; add to pool.
3) Select top-uncertainty points from the pool (real + synthetic), label via oracle, add to labeled set, retrain main model.
4) Visualize decision boundary + uncertainty (bright = high uncertainty).

### Running the baseline (no CGAN)
Example (iris):
```
python3 run_baseline_al.py \
  --dataset iris \
  --seed 42 \
  --epochs 300 \
  --acq-steps 10 \
  --acq-size 10 \
  --make-gif \
  --output runs/iris_baseline_al
```
Same flags except no CGAN-related args. Uses the same visualization layout for comparability.

### Outputs
- `results.json` in the output folder with final metrics (accuracy/NLL for classification; MSE/RMSE/MAE for regression) and frame paths.
- `active_learning.gif` (if `--make-gif`) plus per-step `step_XX.png`.

### Notes and Caveats
- PCA is used for visualization if features >2; warnings may appear from sklearn when inverting meshes; plots still save.
- Colors: bright in the uncertainty heatmap = high uncertainty; pool points are colored by their uncertainty at that step.
- Synthetic points are regenerated every AL step using the current main-model uncertainty; CGAN is re-fit each step to track the shifting boundary.

