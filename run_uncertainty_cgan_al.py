"""
End-to-end run of the uncertainty-conditioned CGAN + oracle labeling + active learning loop
with visualization (frames + GIF).

Steps:
- Split train set into two halves (reproducible seed): labeled_main (for model/CGAN) and pool half.
- Train oracle on full train (sees ground truth).
- Train main model on labeled_main.
- Train CGAN conditioned on main-model uncertainty over labeled_main.
- Generate high-uncertainty synthetic samples; add them to the unlabeled pool.
- Active learning loop: pick most-uncertain points (real + synthetic), label via oracle, add to labeled set, retrain main model.
- Save per-step frames and a GIF with decision boundary and point annotations.

Run:
python3 run_uncertainty_cgan_al.py --dataset iris --output runs/iris_uncgan --seed 42 --epochs 30 --cgan-steps 200 --synthetic-count 200 --acq-steps 5 --acq-size 20
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

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
from models import CGAN, TabularMLP
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


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, task_type: str, device: str) -> Dict:
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
) -> Dict:
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


def save_image_grid(images: np.ndarray, path: Path, title: str = "", cols: int = 5):
    """Save a grid of 28x28 images for MNIST-like inputs."""
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i], cmap="gray")
            if title:
                ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_step(
    step: int,
    X_vis: np.ndarray,
    labeled_idx: np.ndarray,
    pool_idx: np.ndarray,
    selected_idx: np.ndarray,
    synth_pool_idx: np.ndarray,
    synth_acquired_idx: np.ndarray,
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
    # Decision boundary
    ax = axes[0]
    ax.contourf(xx, yy, np.nan_to_num(Z, nan=0.0), levels=20, alpha=0.35, cmap="coolwarm")

    # color pool by uncertainty scores
    pool_norm = None
    if pool_scores.size:
        ps = pool_scores
        pool_norm = (ps - ps.min()) / (ps.max() - ps.min() + 1e-8)
        pool_colors = plt.cm.magma(pool_norm)
    else:
        pool_colors = "lightgray"

    ax.scatter(X_vis[pool_idx, 0], X_vis[pool_idx, 1], c=pool_colors, s=18, label="Pool (uncertainty color)", alpha=0.7, edgecolors="none")
    ax.scatter(X_vis[labeled_idx, 0], X_vis[labeled_idx, 1], c="tab:blue", s=22, label="Labeled", alpha=0.8, edgecolors="k", linewidth=0.2)
    if len(selected_idx) > 0:
        ax.scatter(X_vis[selected_idx, 0], X_vis[selected_idx, 1], edgecolors="red", facecolors="none", s=80, linewidth=1.5, label="Acquired")
    if len(synth_pool_idx) > 0:
        ax.scatter(X_vis[synth_pool_idx, 0], X_vis[synth_pool_idx, 1], marker="x", c="k", s=40, label="Synthetic (pool)")
    if len(synth_acquired_idx) > 0:
        ax.scatter(X_vis[synth_acquired_idx, 0], X_vis[synth_acquired_idx, 1], marker="D", c="lime", s=55, label="Synthetic (acquired)", edgecolors="k", linewidth=0.5)
    ax.set_title(f"Step {step} - Decision Boundary")
    ax.legend(loc="upper right", fontsize=8)

    # Uncertainty heatmap
    ax2 = axes[1]
    unc_plot = ax2.contourf(xx, yy, np.nan_to_num(unc_grid, nan=0.0), levels=20, alpha=0.85, cmap="magma")
    ax2.scatter(X_vis[pool_idx, 0], X_vis[pool_idx, 1], c=pool_colors, s=15, alpha=0.8, edgecolors="none")
    ax2.scatter(X_vis[labeled_idx, 0], X_vis[labeled_idx, 1], c="tab:blue", s=15, alpha=0.7)
    if len(selected_idx) > 0:
        ax2.scatter(X_vis[selected_idx, 0], X_vis[selected_idx, 1], edgecolors="red", facecolors="none", s=60, linewidth=1.0)
    if len(synth_pool_idx) > 0:
        ax2.scatter(X_vis[synth_pool_idx, 0], X_vis[synth_pool_idx, 1], marker="x", c="k", s=30)
    if len(synth_acquired_idx) > 0:
        ax2.scatter(X_vis[synth_acquired_idx, 0], X_vis[synth_acquired_idx, 1], marker="D", c="lime", s=40, edgecolors="k", linewidth=0.5)
    ax2.set_title("Uncertainty map (bright = high)")
    fig.colorbar(unc_plot, ax=ax2, shrink=0.9, label="Uncertainty")

    fig.tight_layout()
    out_path = out_dir / f"step_{step:02d}.png"
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

    uncertainty_method = args.uncertainty.lower()

    # Oracle trained on full train
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

    def train_main_models(
        X_tr: np.ndarray, y_tr: np.ndarray
    ) -> tuple[nn.Module, Optional[list[nn.Module]], Optional[LaplaceState]]:
        if uncertainty_method == "ensemble":
            models = []
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
                    tensor_loader(X_tr, y_tr, batch=64, shuffle=True, device=device),
                    TrainingConfig(num_epochs=args.epochs, batch_size=64, learning_rate=0.001, device=device),
                    task_type=task_type,
                )
                models.append(m)
            return models[0], models, None

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
            tensor_loader(X_tr, y_tr, batch=64, shuffle=True, device=device),
            TrainingConfig(num_epochs=args.epochs, batch_size=64, learning_rate=0.001, device=device),
            task_type=task_type,
        )
        lap_state = (
            fit_laplace_last_layer_diag(
                m,
                X_tr,
                y_tr,
                task_type=task_type,
                device=device,
                prior_precision=args.laplace_prior,
            )
            if uncertainty_method == "laplace"
            else None
        )
        return m, None, lap_state

    main_model, ensemble_models, laplace_state = train_main_models(X_lab, y_lab)

    frames = []
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PCA for viz
    pca = None
    X_all = np.concatenate([X_lab, X_pool], axis=0)
    if X_all.shape[1] > 2:
        pca = PCA(n_components=2)
        X_vis_all = pca.fit_transform(X_all.astype(np.float64))
    else:
        X_vis_all = X_all
    vis_lab_idx = np.arange(len(X_lab))
    vis_pool_idx = np.arange(len(X_lab), len(X_lab) + len(X_pool))
    vis_synth_idx = np.array([], dtype=int)

    # Active learning loop
    synth_mask = np.zeros(len(X_pool), dtype=bool)
    for step in range(args.acq_steps):
        if len(X_pool) == 0:
            break

        # Update CGAN each step based on current main-model uncertainty over labeled+pool
        X_cgan_train = np.concatenate([X_lab, X_pool], axis=0)
        _, unc_scores = predict_mean_and_uncertainty(
            uncertainty_method,
            task_type=task_type,
            device=device,
            X=X_cgan_train,
            model=main_model,
            ensemble=ensemble_models,
            laplace_state=laplace_state,
            mc_samples=args.mc_samples,
            laplace_samples=args.laplace_samples,
        )
        unc_scores = np.nan_to_num(unc_scores, nan=0.0, posinf=1.0, neginf=0.0)
        cond_dim = 1
        cgan = CGAN(
            input_dim=input_dim,
            conditional_type="continuous",
            cond_dim=cond_dim,
            device=device,
        )
        cond_tensor = torch.tensor(unc_scores[:, None], dtype=torch.float32, device=device)
        cgan_loader = DataLoader(
            TensorDataset(torch.tensor(X_cgan_train, dtype=torch.float32, device=device), cond_tensor),
            batch_size=64,
            shuffle=True,
        )
        cgan.fit(cgan_loader, num_steps=args.cgan_steps)

        # Generate synthetic targeting high uncertainty range (80-100th percentiles)
        if len(unc_scores) > 0:
            low_p = np.percentile(unc_scores, 80)
            high_p = np.percentile(unc_scores, 100)
            sampled_unc = np.random.uniform(low_p, high_p, size=(args.synthetic_count * 3, 1))
            sampled_unc = np.nan_to_num(sampled_unc, nan=0.0, posinf=1.0, neginf=0.0)
        else:
            sampled_unc = np.full((args.synthetic_count * 3, 1), 0.5)
        target_cond = torch.tensor(sampled_unc, dtype=torch.float32, device=device)
        synth_X, _ = cgan.sample_from_continuous(target_cond)
        _, gen_unc = predict_mean_and_uncertainty(
            uncertainty_method,
            task_type=task_type,
            device=device,
            X=synth_X.cpu().numpy(),
            model=main_model,
            ensemble=ensemble_models,
            laplace_state=laplace_state,
            mc_samples=args.mc_samples,
            laplace_samples=args.laplace_samples,
        )
        top_gen_idx = np.argpartition(gen_unc, -args.synthetic_count)[-args.synthetic_count:]
        synth_X = synth_X[top_gen_idx]
        oracle.eval()
        with torch.no_grad():
            if task_type == "classification":
                logits = oracle(synth_X.to(device))
                synth_y = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
            else:
                preds = oracle(synth_X.to(device)).squeeze().cpu().numpy()
                synth_y = preds

        # Save a grid of generated samples for vision datasets (flattened 28x28)
        if synth_X.shape[1] == 784:
            synth_dir = out_dir / "synthetic_samples"
            synth_dir.mkdir(parents=True, exist_ok=True)
            imgs = synth_X[: min(25, synth_X.size(0))].cpu().numpy().reshape(-1, 28, 28)
            cols = 5
            rows = int(np.ceil(len(imgs) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            axes = axes.flatten()
            for i, ax in enumerate(axes):
                ax.axis("off")
                if i < len(imgs):
                    ax.imshow(imgs[i], cmap="gray")
            fig.tight_layout()
            fig.savefig(synth_dir / f"step_{step:02d}_synthetic_grid.png", dpi=150)
            plt.close(fig)
        pool_synth_start = len(X_pool)
        X_pool = np.concatenate([X_pool, synth_X.cpu().numpy()], axis=0)
        y_pool = np.concatenate([y_pool, synth_y], axis=0)
        new_synth_mask = np.zeros(len(X_pool), dtype=bool)
        new_synth_mask[: len(synth_mask)] = synth_mask
        new_synth_mask[pool_synth_start:] = True
        synth_mask = new_synth_mask

        # Optional visualization of certain vs uncertain samples (MNIST only)
        if synth_X.shape[1] == 784:
            synth_dir = out_dir / "synthetic_samples"
            synth_dir.mkdir(parents=True, exist_ok=True)
            # Uncertain samples (already top uncertain)
            imgs_uncertain = synth_X.cpu().numpy().reshape(-1, 28, 28)
            save_image_grid(imgs_uncertain[: min(25, len(imgs_uncertain))], synth_dir / f"step_{step:02d}_uncertain.png", title="Uncertain")
            # Certain samples: sample low uncertainty range from current pool
            _, pool_unc = (
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
            if len(pool_unc) > 0:
                certain_idx = np.argsort(pool_unc)[: min(25, len(pool_unc))]
                imgs_certain = X_pool[certain_idx].reshape(-1, 28, 28)
                save_image_grid(imgs_certain, synth_dir / f"step_{step:02d}_certain.png", title="Certain")

        # Recompute visualization coordinates after pool changes
        X_all = np.concatenate([X_lab, X_pool], axis=0)
        if pca is not None:
            X_vis_all = pca.transform(np.nan_to_num(X_all.astype(np.float64)))
        else:
            X_vis_all = X_all
        vis_lab_idx = np.arange(len(X_lab))
        vis_pool_idx = np.arange(len(X_lab), len(X_lab) + len(X_pool))
        vis_synth_idx = vis_pool_idx[np.where(synth_mask)[0]] if synth_mask.size else np.array([], dtype=int)

        _, scores = predict_mean_and_uncertainty(
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
        k = min(args.acq_size, len(X_pool))
        top_idx = np.argpartition(scores, -k)[-k:]
        # Map selected to visualization indices before pool shrink
        if len(vis_pool_idx) == len(X_pool):
            selected_vis = vis_pool_idx[top_idx]
        else:
            # rebuild vis_pool_idx
            vis_pool_idx = np.arange(len(X_lab), len(X_lab) + len(X_pool))
            selected_vis = vis_pool_idx[top_idx]
        synth_selected_vis = selected_vis[np.isin(top_idx, np.where(synth_mask)[0])] if synth_mask.size else np.array([], dtype=int)

        # Plot BEFORE pool shrink/retrain using current scores
        pool_scores_vis = scores
        frame = plot_step(
            step,
            X_vis_all,
            vis_lab_idx,
            vis_pool_idx,
            selected_vis if selected_vis.size else np.array([], dtype=int),
            vis_pool_idx[np.where(synth_mask)[0]] if synth_mask.size else np.array([], dtype=int),
            synth_selected_vis if synth_selected_vis.size else np.array([], dtype=int),
            pool_scores_vis,
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
        frames.append(frame)

        # Oracle labels (we already have y_pool as oracle labels)
        X_new, y_new = X_pool[top_idx], y_pool[top_idx]
        X_lab = np.concatenate([X_lab, X_new], axis=0)
        y_lab = np.concatenate([y_lab, y_new], axis=0)
        # Remove from pool
        mask = np.ones(len(X_pool), dtype=bool)
        mask[top_idx] = False
        X_pool, y_pool = X_pool[mask], y_pool[mask]
        if synth_mask.size:
            synth_mask = synth_mask[mask]
        # Retrain main model(s)
        main_model, ensemble_models, laplace_state = train_main_models(X_lab, y_lab)

        # Update viz indices
        X_all = np.concatenate([X_lab, X_pool], axis=0)
        if pca is not None:
            X_vis_all = pca.transform(X_all.astype(np.float64))
        else:
            X_vis_all = X_all
        vis_lab_idx = np.arange(len(X_lab))
        vis_pool_idx = np.arange(len(X_lab), len(X_lab) + len(X_pool))
        # remaining synthetic points in pool
        vis_synth_idx = vis_pool_idx[np.where(synth_mask)[0]] if synth_mask.size else np.array([], dtype=int)

    # Final metrics
    metrics = evaluate_with_uncertainty_method(
        uncertainty_method,
        task_type=task_type,
        device=device,
        X_test=X_test,
        y_test=y_test,
        model=main_model,
        ensemble=ensemble_models,
    )
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "initial_label_frac": args.initial_label_frac,
        "uncertainty_method": uncertainty_method,
        "task_type": task_type,
        "num_classes": num_classes,
        "method": "uncertainty_cgan",
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
    p = argparse.ArgumentParser(description="Uncertainty-conditioned CGAN + oracle AL with visualization")
    p.add_argument("--dataset", required=True, help="Dataset name (e.g., iris,wine,breast_cancer,two_moons,circles)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30, help="Epochs per training cycle")
    p.add_argument("--cgan-steps", type=int, default=200, help="CGAN training steps")
    p.add_argument("--synthetic-count", type=int, default=200, help="Synthetic samples to generate")
    p.add_argument("--acq-steps", type=int, default=5, help="Active learning acquisition steps")
    p.add_argument("--acq-size", type=int, default=20, help="Points to acquire per step")
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
