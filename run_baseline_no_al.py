"""
Baseline without active learning:
- Split train set in half (labeled_main vs pool).
- Train oracle on full train (for consistency with other scripts).
- Train main model once on labeled_main only (no acquisitions).
- Visualize decision boundary + uncertainty heatmap (pool colored by uncertainty).
- Save frames/GIF and results.json similar to other runners.
"""
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from config import ModelConfig, TrainingConfig
from data import DatasetLoader
from models import TabularMLP
from uncertainty_methods import LaplaceState, fit_laplace_last_layer_diag, predict_mean_and_uncertainty
from utils import create_mesh, get_device


def log(msg: str):
    print(msg, flush=True)


def build_model(cfg: ModelConfig, device: str) -> nn.Module:
    return TabularMLP(
        input_dim=cfg.input_dim,
        hidden_dims=cfg.hidden_dims,
        output_dim=cfg.output_dim,
        task_type=cfg.task_type,
        num_classes=cfg.num_classes,
        dropout_rate=cfg.dropout_rate,
        activation=cfg.activation,
    ).to(device)


def train_model(model: nn.Module, loader: DataLoader, cfg: TrainingConfig, task_type: str) -> None:
    criterion = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    device = cfg.device
    model.train()
    for _ in range(cfg.num_epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if task_type == "regression" and y.ndim == 1:
                y = y.unsqueeze(1)
            opt.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            opt.step()


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, task_type: str, device: str) -> Dict[str, Any]:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, device=device)
    with torch.no_grad():
        out = model(X_t)
        if task_type == "classification":
            probs = torch.softmax(out, dim=1)
            preds = probs.argmax(dim=1)
            acc = (preds == y_t).float().mean()
            nll = -torch.log(probs[torch.arange(len(y_t)), y_t] + 1e-8).mean()
            f1 = f1_score(y_t.cpu().numpy(), preds.cpu().numpy(), average="macro")
            return {"accuracy": float(acc.item()), "f1_macro": float(f1), "nll": float(nll.item())}
        else:
            if out.ndim > 1 and out.shape[1] > 1:
                out = out.squeeze()
            mse = F.mse_loss(out, y_t.float())
            rmse = torch.sqrt(mse)
            mae = torch.mean(torch.abs(out - y_t.float()))
            return {"mse": float(mse.item()), "rmse": float(rmse.item()), "mae": float(mae.item())}

def evaluate_with_uncertainty_method(
    method: str,
    task_type: str,
    device: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    model: Optional[nn.Module] = None,
    ensemble: Optional[list[nn.Module]] = None,
    laplace_state: Optional[LaplaceState] = None,
) -> Dict[str, Any]:
    """Compute metrics using the method's mean predictive distribution where applicable."""
    method = method.lower()
    if method == "ensemble":
        mean_pred, _ = predict_mean_and_uncertainty(
            "ensemble",
            task_type=task_type,
            device=device,
            X=X_test,
            ensemble=ensemble,
        )
        if task_type == "classification":
            probs = torch.tensor(mean_pred, device=device)
            y_t = torch.tensor(y_test, device=device)
            preds = probs.argmax(dim=1)
            acc = (preds == y_t).float().mean()
            nll = -torch.log(probs[torch.arange(len(y_t)), y_t] + 1e-8).mean()
            f1 = f1_score(y_t.cpu().numpy(), preds.cpu().numpy(), average="macro")
            return {"accuracy": float(acc.item()), "f1_macro": float(f1), "nll": float(nll.item())}
        preds = torch.tensor(mean_pred, device=device).float()
        y_t = torch.tensor(y_test, device=device).float()
        mse = F.mse_loss(preds, y_t)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(preds - y_t))
        return {"mse": float(mse.item()), "rmse": float(rmse.item()), "mae": float(mae.item())}

    # For dropout/laplace, keep deterministic evaluation of the trained model.
    if model is None:
        raise ValueError("Non-ensemble evaluation requires 'model'.")
    return evaluate(model, X_test, y_test, task_type, device)


def compute_uncertainty(model: nn.Module, X: np.ndarray, task_type: str, device: str, mc_samples: int = 20):
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(mc_samples):
            out = model(X_t)
            if task_type == "classification":
                preds.append(torch.softmax(out, dim=1))
            else:
                preds.append(out.squeeze())
    stacked = torch.stack(preds)
    if task_type == "classification":
        mean_p = stacked.mean(dim=0)
        entropy = -torch.sum(mean_p * torch.log(mean_p + 1e-8), dim=1)
        return entropy.cpu().numpy()
    var = torch.var(stacked, dim=0)
    return var.cpu().numpy()


def tensor_loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool, device: str):
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, device=device)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=shuffle)


def plot_step(
    X_vis: np.ndarray,
    labeled_idx: np.ndarray,
    pool_idx: np.ndarray,
    pool_scores: np.ndarray,
    model: nn.Module,
    ensemble_models: Optional[list[nn.Module]],
    laplace_state: Optional[LaplaceState],
    uncertainty_method: str,
    task_type: str,
    mesh: Tuple[np.ndarray, np.ndarray, np.ndarray],
    pca: Optional[PCA],
    out_dir: Path,
    mc_samples: int = 20,
    laplace_samples: int = 30,
):
    device = next(model.parameters()).device
    xx, yy, mesh_points = mesh
    mesh_orig = pca.inverse_transform(mesh_points) if pca is not None else mesh_points
    mean_pred, unc_vals = predict_mean_and_uncertainty(
        uncertainty_method,
        task_type=task_type,
        device=str(device),
        X=mesh_orig,
        model=model,
        ensemble=ensemble_models,
        laplace_state=laplace_state,
        mc_samples=mc_samples,
        laplace_samples=laplace_samples,
    )
    if task_type == "classification":
        if mean_pred.ndim == 2 and mean_pred.shape[1] > 2:
            Z = mean_pred.argmax(axis=1).reshape(xx.shape)
        else:
            Z = mean_pred[:, 1].reshape(xx.shape) if mean_pred.ndim == 2 and mean_pred.shape[1] > 1 else mean_pred.reshape(xx.shape)
    else:
        Z = mean_pred.reshape(xx.shape)
    unc_grid = unc_vals.reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if pool_scores.size:
        ps = pool_scores
        norm = (ps - ps.min()) / (ps.max() - ps.min() + 1e-8)
        pool_colors = plt.cm.magma(norm)
    else:
        pool_colors = "lightgray"

    ax = axes[0]
    ax.contourf(xx, yy, np.nan_to_num(Z, nan=0.0), levels=20, alpha=0.35, cmap="coolwarm")
    ax.scatter(X_vis[pool_idx, 0], X_vis[pool_idx, 1], c=pool_colors, s=18, label="Pool (uncertainty color)", alpha=0.7, edgecolors="none")
    ax.scatter(X_vis[labeled_idx, 0], X_vis[labeled_idx, 1], c="tab:blue", s=22, label="Labeled", alpha=0.8, edgecolors="k", linewidth=0.2)
    ax.set_title("Decision Boundary")
    ax.legend(loc="upper right", fontsize=8)

    ax2 = axes[1]
    unc_plot = ax2.contourf(xx, yy, np.nan_to_num(unc_grid, nan=0.0), levels=20, alpha=0.85, cmap="magma")
    ax2.scatter(X_vis[pool_idx, 0], X_vis[pool_idx, 1], c=pool_colors, s=15, alpha=0.8, edgecolors="none")
    ax2.scatter(X_vis[labeled_idx, 0], X_vis[labeled_idx, 1], c="tab:blue", s=15, alpha=0.7)
    ax2.set_title("Uncertainty map (bright = high)")
    fig.colorbar(unc_plot, ax=ax2, shrink=0.9, label="Uncertainty")

    fig.tight_layout()
    out_path = out_dir / "step_00.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_pipeline(args):
    device = get_device()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    loader = DatasetLoader(
        dataset_name=args.dataset,
        test_size=0.2,
        random_state=args.seed,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    train_loader, test_loader, meta = loader.load()
    task_type = meta["task_type"]
    num_classes = meta.get("num_classes")
    input_dim = meta["input_dim"]

    X_train = train_loader.dataset.X.numpy()
    y_train = train_loader.dataset.y.numpy()
    X_test = test_loader.dataset.X.numpy()
    y_test = test_loader.dataset.y.numpy()

    idx = np.arange(len(X_train))
    rng.shuffle(idx)
    frac = float(args.initial_label_frac)
    frac = max(0.01, min(0.99, frac))
    labeled_n = int(round(len(idx) * frac))
    labeled_n = max(1, min(len(idx) - 1, labeled_n))
    labeled_idx = idx[:labeled_n]
    pool_idx = idx[labeled_n:]
    X_lab, y_lab = X_train[labeled_idx], y_train[labeled_idx]
    X_pool, y_pool = X_train[pool_idx], y_train[pool_idx]

    # Oracle (not used for labeling here but kept for parity)
    oracle = build_model(
        ModelConfig(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            output_dim=num_classes if task_type == "classification" else 1,
            task_type=task_type,
            num_classes=num_classes if task_type == "classification" else None,
            dropout_rate=args.dropout,
        ),
        device,
    )
    oracle_loader = tensor_loader(X_train, y_train, batch=64, shuffle=True, device=device)
    train_model(
        oracle,
        oracle_loader,
        TrainingConfig(num_epochs=args.epochs, batch_size=64, learning_rate=0.001, device=device),
        task_type=task_type,
    )

    uncertainty_method = args.uncertainty.lower()
    main_model: Optional[nn.Module] = None
    ensemble_models: Optional[list[nn.Module]] = None
    laplace_state: Optional[LaplaceState] = None

    if uncertainty_method == "ensemble":
        ensemble_models = []
        for i in range(args.ensemble_size):
            torch.manual_seed(args.seed + i)
            m = build_model(
                ModelConfig(
                    input_dim=input_dim,
                    hidden_dims=[64, 32],
                    output_dim=num_classes if task_type == "classification" else 1,
                    task_type=task_type,
                    num_classes=num_classes if task_type == "classification" else None,
                    dropout_rate=args.dropout,
                ),
                device,
            )
            train_model(
                m,
                tensor_loader(X_lab, y_lab, batch=64, shuffle=True, device=device),
                TrainingConfig(num_epochs=args.epochs, batch_size=64, learning_rate=0.001, device=device),
                task_type=task_type,
            )
            ensemble_models.append(m)
        main_model = ensemble_models[0]
    else:
        main_model = build_model(
            ModelConfig(
                input_dim=input_dim,
                hidden_dims=[64, 32],
                output_dim=num_classes if task_type == "classification" else 1,
                task_type=task_type,
                num_classes=num_classes if task_type == "classification" else None,
                dropout_rate=args.dropout,
            ),
            device,
        )
        train_model(
            main_model,
            tensor_loader(X_lab, y_lab, batch=64, shuffle=True, device=device),
            TrainingConfig(num_epochs=args.epochs, batch_size=64, learning_rate=0.001, device=device),
            task_type=task_type,
        )
        if uncertainty_method == "laplace":
            laplace_state = fit_laplace_last_layer_diag(
                main_model,
                X_lab,
                y_lab,
                task_type=task_type,
                device=device,
                prior_precision=args.laplace_prior,
            )

    # Visualization setup
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pca = None
    X_all = np.concatenate([X_lab, X_pool], axis=0)
    if X_all.shape[1] > 2:
        pca = PCA(n_components=2)
        X_vis_all = pca.fit_transform(np.nan_to_num(X_all.astype(np.float64)))
    else:
        X_vis_all = X_all
    vis_lab_idx = np.arange(len(X_lab))
    vis_pool_idx = np.arange(len(X_lab), len(X_lab) + len(X_pool))

    _, pool_scores = (
        predict_mean_and_uncertainty(
            uncertainty_method,
            task_type=task_type,
            device=device,
            X=X_pool,
            model=main_model,
            ensemble=ensemble_models,
            laplace_state=laplace_state,
            mc_samples=args.mc_samples,
            laplace_samples=args.laplace_samples,
        )
        if len(X_pool)
        else (np.array([]), np.array([]))
    )
    frame = plot_step(
        X_vis_all,
        vis_lab_idx,
        vis_pool_idx,
        pool_scores,
        main_model,
        ensemble_models,
        laplace_state,
        uncertainty_method,
        task_type,
        create_mesh(X_vis_all if X_vis_all.shape[1] == 2 else X_vis_all[:, :2], n_points=120, margin=0.2),
        pca,
        out_dir,
        mc_samples=args.mc_samples,
        laplace_samples=args.laplace_samples,
    )
    frames = [frame]

    metrics = evaluate_with_uncertainty_method(
        uncertainty_method,
        task_type=task_type,
        device=device,
        X_test=X_test,
        y_test=y_test,
        model=main_model,
        ensemble=ensemble_models,
        laplace_state=laplace_state,
    )
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "initial_label_frac": args.initial_label_frac,
        "uncertainty_method": uncertainty_method,
        "task_type": task_type,
        "num_classes": num_classes,
        "method": "no_al",
        "metrics": metrics,
        "frames": [str(f) for f in frames],
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)
    if args.make_gif and frames:
        imgs = [imageio.v2.imread(f) for f in frames]
        imageio.mimsave(out_dir / "active_learning.gif", imgs, duration=0.8)
        log(f"Saved GIF to {out_dir / 'active_learning.gif'}")
    log(f"Done. Metrics: {metrics}")


def parse_args():
    p = argparse.ArgumentParser(description="Baseline without active learning (train on half, visualize).")
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--initial-label-frac", type=float, default=0.5, help="Initial labeled fraction of the train split.")
    p.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for train samples after split.")
    p.add_argument("--max-test-samples", type=int, default=None, help="Optional cap for test samples after split.")
    p.add_argument("--uncertainty", choices=["dropout", "ensemble", "laplace"], default="dropout")
    p.add_argument("--mc-samples", type=int, default=20, help="MC samples for dropout uncertainty.")
    p.add_argument("--ensemble-size", type=int, default=5, help="Ensemble size for deep ensembles.")
    p.add_argument("--laplace-prior", type=float, default=1.0, help="Prior precision for diagonal Laplace.")
    p.add_argument("--laplace-samples", type=int, default=30, help="Samples for Laplace predictive approximation.")
    p.add_argument("--make-gif", action="store_true")
    p.add_argument("--output", required=True, help="Output folder for frames/GIF/results")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
