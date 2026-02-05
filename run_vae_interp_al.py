"""
VAE-latent interpolation active learning runner (new approach).

This file does NOT modify the existing CGAN pipeline. It implements a parallel approach:
- Train a VAE on the full training split (all available train data).
- Train an oracle on the full training split (has ground truth; acts as labeler).
- Split train data in half: labeled_half vs pool_half (main model only trains on labeled_half initially).
- Each AL step:
  1) Find uncertain real pool samples (top 20% by MC-dropout entropy/variance).
  2) Encode those samples into VAE latent space, interpolate latents, decode to generate new samples.
  3) Filter generated samples by main-model uncertainty and add to pool (oracle labels them).
  4) Standard active learning: acquire most-uncertain from pool (real + generated), oracle labels, add to labeled.
  5) Retrain main model from scratch on expanded labeled set.
  6) Save visualization frames (decision regions + uncertainty heatmap) and save sample grids for MNIST-like data.

Run example:
python3 run_vae_interp_al.py --dataset mnist --seed 42 --epochs 30 --vae-epochs 10 --vae-latent-dim 32 \\
  --synthetic-count 200 --acq-steps 5 --acq-size 20 --make-gif --output runs/mnist_vae_interp_al
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


class MLPVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Tuple[int, ...] = (512, 256)):
        super().__init__()
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.ReLU())
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.mu = nn.Linear(prev, latent_dim)
        self.logvar = nn.Linear(prev, latent_dim)

        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.ReLU())
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def tensor_loader(X: np.ndarray, y: Optional[np.ndarray], batch: int, shuffle: bool, device: str) -> DataLoader:
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    if y is None:
        ds = TensorDataset(Xt)
    else:
        yt = torch.tensor(y, device=device)
        ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


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


def train_main(model: nn.Module, loader: DataLoader, cfg: TrainingConfig, task_type: str) -> None:
    criterion = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    device = cfg.device
    model.train()
    for _ in range(cfg.num_epochs):
        for batch in loader:
            X, y = batch
            X, y = X.to(device), y.to(device)
            if task_type == "regression" and y.ndim == 1:
                y = y.unsqueeze(1)
            opt.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            opt.step()


def train_vae(vae: MLPVAE, loader: DataLoader, device: str, epochs: int, lr: float = 1e-3, beta: float = 1.0):
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    for _ in range(epochs):
        for (X,) in loader:
            X = X.to(device)
            opt.zero_grad()
            recon, mu, logvar = vae(X)
            recon_loss = F.mse_loss(recon, X, reduction="mean")
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl
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
) -> Dict[str, Any]:
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


def compute_uncertainty(model: nn.Module, X: np.ndarray, task_type: str, device: str, mc_samples: int) -> np.ndarray:
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    model.train()  # enable dropout
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


def select_diverse_by_latent(
    z: np.ndarray, scores: np.ndarray, k: int, top_factor: int = 10
) -> np.ndarray:
    """
    Select k diverse samples from candidates using farthest-point sampling in latent space,
    restricted to the top (k*top_factor) by uncertainty score.
    """
    if len(z) == 0:
        return np.array([], dtype=int)
    k = min(k, len(z))
    top_m = min(len(z), max(k, k * top_factor))
    top_idx = np.argpartition(scores, -top_m)[-top_m:]
    # start from most uncertain within top set
    start = top_idx[np.argmax(scores[top_idx])]
    chosen = [start]
    # distances to chosen set
    dist = np.full(len(z), np.inf, dtype=np.float32)
    for _ in range(1, k):
        last = chosen[-1]
        d = np.linalg.norm(z - z[last], axis=1)
        dist = np.minimum(dist, d)
        # restrict selection to top_idx for diversity among uncertain ones
        next_local = top_idx[np.argmax(dist[top_idx])]
        chosen.append(int(next_local))
    return np.array(chosen, dtype=int)


def save_image_grid(images: np.ndarray, path: Path, title: str, cols: int = 5):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i], cmap="gray")
            ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_frame(
    step: int,
    X_l_vis: np.ndarray,
    X_p_vis: np.ndarray,
    pool_scores: np.ndarray,
    y_l: np.ndarray,
    acquired_idx: np.ndarray,
    is_synth: np.ndarray,
    model: nn.Module,
    ensemble_models: Optional[list[nn.Module]],
    laplace_state: Optional[LaplaceState],
    uncertainty_method: str,
    task_type: str,
    pca: PCA,
    out_dir: Path,
    num_classes: Optional[int],
    mc_samples: int,
    laplace_samples: int,
) -> Path:
    X_all_vis = np.vstack([X_l_vis, X_p_vis])
    xx, yy, mesh_points = create_mesh(X_all_vis, n_points=120, margin=0.2)
    mesh_orig = pca.inverse_transform(mesh_points.astype(np.float64))
    mean_pred, unc_vals = predict_mean_and_uncertainty(
        uncertainty_method,
        task_type=task_type,
        device=str(next(model.parameters()).device),
        X=mesh_orig,
        model=model,
        ensemble=ensemble_models,
        laplace_state=laplace_state,
        mc_samples=mc_samples,
        laplace_samples=laplace_samples,
    )
    if task_type == "classification":
        pred = (
            mean_pred.argmax(axis=1).reshape(xx.shape)
            if mean_pred.ndim == 2 and mean_pred.shape[1] > 1
            else mean_pred.reshape(xx.shape)
        )
    else:
        pred = mean_pred.reshape(xx.shape)
    unc_grid = unc_vals.reshape(xx.shape)

    # pool colors by uncertainty
    if pool_scores.size:
        ps = pool_scores
        norm = (ps - ps.min()) / (ps.max() - ps.min() + 1e-8)
        pool_colors = plt.cm.magma(norm)
    else:
        pool_colors = "lightgray"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    if task_type == "classification" and num_classes and num_classes > 2:
        ax.contourf(xx, yy, pred, levels=num_classes, alpha=0.35, cmap="tab10")
    else:
        ax.contourf(xx, yy, pred, levels=20, alpha=0.35, cmap="coolwarm")
    ax.scatter(X_p_vis[:, 0], X_p_vis[:, 1], c=pool_colors, s=18, alpha=0.7, edgecolors="none", label="Pool (uncertainty color)")
    ax.scatter(X_l_vis[:, 0], X_l_vis[:, 1], c=y_l if (task_type == "classification" and num_classes and num_classes <= 10) else "tab:blue",
               cmap="tab10" if (task_type == "classification" and num_classes and num_classes <= 10) else None,
               s=18, alpha=0.8, edgecolors="k", linewidth=0.2, label="Labeled")
    if acquired_idx.size:
        ax.scatter(X_p_vis[acquired_idx, 0], X_p_vis[acquired_idx, 1], facecolors="none", edgecolors="red", s=80, linewidth=1.5, label="Acquired")
    synth_pool_idx = np.where(is_synth)[0]
    if synth_pool_idx.size:
        ax.scatter(X_p_vis[synth_pool_idx, 0], X_p_vis[synth_pool_idx, 1], marker="x", c="k", s=35, label="Synthetic (pool)")
    synth_acq = acquired_idx[np.where(is_synth[acquired_idx])[0]] if acquired_idx.size else np.array([], dtype=int)
    if synth_acq.size:
        ax.scatter(X_p_vis[synth_acq, 0], X_p_vis[synth_acq, 1], marker="D", c="lime", s=55, edgecolors="k", linewidth=0.5, label="Synthetic (acquired)")
    ax.set_title(f"Step {step} - Decision Regions")
    ax.legend(loc="upper right", fontsize=8)

    ax2 = axes[1]
    unc_plot = ax2.contourf(xx, yy, np.nan_to_num(unc_grid, nan=0.0), levels=20, alpha=0.85, cmap="magma")
    ax2.scatter(X_p_vis[:, 0], X_p_vis[:, 1], c=pool_colors, s=15, alpha=0.8, edgecolors="none")
    ax2.scatter(X_l_vis[:, 0], X_l_vis[:, 1], c="tab:blue", s=12, alpha=0.7)
    if acquired_idx.size:
        ax2.scatter(X_p_vis[acquired_idx, 0], X_p_vis[acquired_idx, 1], facecolors="none", edgecolors="red", s=60, linewidth=1.0)
    if synth_pool_idx.size:
        ax2.scatter(X_p_vis[synth_pool_idx, 0], X_p_vis[synth_pool_idx, 1], marker="x", c="k", s=25)
    if synth_acq.size:
        ax2.scatter(X_p_vis[synth_acq, 0], X_p_vis[synth_acq, 1], marker="D", c="lime", s=40, edgecolors="k", linewidth=0.5)
    ax2.set_title("Uncertainty map (bright = high)")
    fig.colorbar(unc_plot, ax=ax2, shrink=0.9, label="Uncertainty")

    fig.tight_layout()
    out_path = out_dir / f"step_{step:02d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run(args):
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
    scaler = getattr(loader, "scaler", None)

    task_type = meta["task_type"]
    num_classes = meta.get("num_classes")
    input_dim = meta["input_dim"]

    X_train = train_loader.dataset.X.numpy()
    y_train = train_loader.dataset.y.numpy()
    X_test = test_loader.dataset.X.numpy()
    y_test = test_loader.dataset.y.numpy()

    # Split train in half (main labeled vs pool)
    idx = np.arange(len(X_train))
    rng.shuffle(idx)
    frac = float(args.initial_label_frac)
    frac = max(0.01, min(0.99, frac))
    labeled_n = int(round(len(idx) * frac))
    labeled_n = max(1, min(len(idx) - 1, labeled_n))
    labeled_idx = idx[:labeled_n]
    pool_idx = idx[labeled_n:]
    X_l, y_l = X_train[labeled_idx], y_train[labeled_idx]
    X_p, y_p = X_train[pool_idx], y_train[pool_idx]  # oracle has access to labels
    is_synth = np.zeros(len(X_p), dtype=bool)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "synthetic_samples").mkdir(parents=True, exist_ok=True)

    # Fit PCA once for visualization consistency
    pca = PCA(n_components=2)
    pca.fit(np.nan_to_num(np.vstack([X_l, X_p]).astype(np.float64)))

    # Oracle on full train
    oracle = build_model(
        ModelConfig(
            input_dim=input_dim,
            hidden_dims=[512, 256],
            output_dim=num_classes if task_type == "classification" else 1,
            task_type=task_type,
            num_classes=num_classes if task_type == "classification" else None,
            dropout_rate=args.dropout,
        ),
        device,
    )
    train_main(
        oracle,
        tensor_loader(X_train, y_train, batch=args.batch_size, shuffle=True, device=device),
        TrainingConfig(num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=0.001, device=device),
        task_type,
    )

    # Main model initial on labeled half
    def fresh_main() -> nn.Module:
        return build_model(
            ModelConfig(
                input_dim=input_dim,
                hidden_dims=[512, 256],
                output_dim=num_classes if task_type == "classification" else 1,
                task_type=task_type,
                num_classes=num_classes if task_type == "classification" else None,
                dropout_rate=args.dropout,
            ),
            device,
        )

    uncertainty_method = args.uncertainty.lower()

    def train_main_models(
        X_tr: np.ndarray, y_tr: np.ndarray
    ) -> tuple[nn.Module, Optional[list[nn.Module]], Optional[LaplaceState]]:
        if uncertainty_method == "ensemble":
            models: list[nn.Module] = []
            for i in range(args.ensemble_size):
                torch.manual_seed(args.seed + i)
                m = fresh_main()
                train_main(
                    m,
                    tensor_loader(X_tr, y_tr, batch=args.batch_size, shuffle=True, device=device),
                    TrainingConfig(num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=0.001, device=device),
                    task_type,
                )
                models.append(m)
            return models[0], models, None

        m = fresh_main()
        train_main(
            m,
            tensor_loader(X_tr, y_tr, batch=args.batch_size, shuffle=True, device=device),
            TrainingConfig(num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=0.001, device=device),
            task_type,
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

    main, ensemble_models, laplace_state = train_main_models(X_l, y_l)

    # VAE on full train
    vae = MLPVAE(input_dim=input_dim, latent_dim=args.vae_latent_dim, hidden_dims=(args.vae_h1, args.vae_h2)).to(device)
    train_vae(
        vae,
        tensor_loader(X_train, None, batch=args.batch_size, shuffle=True, device=device),
        device=device,
        epochs=args.vae_epochs,
        lr=args.vae_lr,
        beta=args.vae_beta,
    )

    frames = []

    for step in range(args.acq_steps):
        if len(X_p) == 0:
            break

        # Pool uncertainty and uncertain candidate set (top 20%)
        _, pool_scores = predict_mean_and_uncertainty(
            uncertainty_method,
            task_type=task_type,
            device=device,
            X=X_p,
            model=main,
            ensemble=ensemble_models,
            laplace_state=laplace_state,
            mc_samples=args.mc_samples,
            laplace_samples=args.laplace_samples,
        )
        thresh = np.percentile(pool_scores, 80) if len(pool_scores) else 0.0
        candidates = np.where(pool_scores >= thresh)[0]
        if candidates.size < 2:
            candidates = np.arange(len(X_p))

        # Encode candidates and interpolate
        vae.eval()
        with torch.no_grad():
            X_cand_t = torch.tensor(X_p[candidates], dtype=torch.float32, device=device)
            mu_t, logvar_t = vae.encode(X_cand_t)
            std_t = torch.exp(0.5 * logvar_t)
            eps_t = torch.randn_like(std_t)
            z_cand_t = mu_t + eps_t * std_t
        z_cand = z_cand_t.cpu().numpy()

        gen_count = args.synthetic_count * 5
        if len(z_cand) < 2:
            # fall back to sampling from standard normal
            z_interp = rng.normal(size=(gen_count, args.vae_latent_dim)).astype(np.float32)
        else:
            # Ensure we don't keep reusing only a couple points: cover candidate set uniformly for idx1
            idx1 = np.tile(np.arange(len(z_cand)), int(np.ceil(gen_count / len(z_cand))))[:gen_count]
            rng.shuffle(idx1)
            idx2 = rng.integers(0, len(z_cand), size=gen_count)
            collisions = idx2 == idx1
            while np.any(collisions):
                idx2[collisions] = rng.integers(0, len(z_cand), size=int(np.sum(collisions)))
                collisions = idx2 == idx1

            # Interpolation weights: sample a range of weights (not fixed 0.5)
            # Beta(0.5,0.5) biases toward endpoints, increasing variety.
            alpha = rng.beta(0.5, 0.5, size=(gen_count, 1)).astype(np.float32)
            z_interp = alpha * z_cand[idx1] + (1.0 - alpha) * z_cand[idx2]
            # Add small latent noise for extra diversity
            z_interp = z_interp + rng.normal(scale=0.05, size=z_interp.shape).astype(np.float32)

        z_t = torch.tensor(z_interp, dtype=torch.float32, device=device)
        with torch.no_grad():
            X_gen = vae.decode(z_t).cpu().numpy()

        # Filter generated to most uncertain according to main model, then enforce diversity in latent space
        _, gen_scores = predict_mean_and_uncertainty(
            uncertainty_method,
            task_type=task_type,
            device=device,
            X=X_gen,
            model=main,
            ensemble=ensemble_models,
            laplace_state=laplace_state,
            mc_samples=args.mc_samples,
            laplace_samples=args.laplace_samples,
        )
        keep = select_diverse_by_latent(z_interp, gen_scores, k=args.synthetic_count, top_factor=10)
        X_gen = X_gen[keep]
        gen_scores = gen_scores[keep]

        # Oracle labels synthetic
        oracle.eval()
        with torch.no_grad():
            if task_type == "classification":
                logits = oracle(torch.tensor(X_gen, dtype=torch.float32, device=device))
                y_gen = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
            else:
                preds = oracle(torch.tensor(X_gen, dtype=torch.float32, device=device)).squeeze().cpu().numpy()
                y_gen = preds

        # Add synthetic to pool
        X_p = np.concatenate([X_p, X_gen], axis=0)
        y_p = np.concatenate([y_p, y_gen], axis=0)
        is_synth = np.concatenate([is_synth, np.ones(len(X_gen), dtype=bool)], axis=0)

        # Recompute pool uncertainty after augmentation
        _, pool_scores = predict_mean_and_uncertainty(
            uncertainty_method,
            task_type=task_type,
            device=device,
            X=X_p,
            model=main,
            ensemble=ensemble_models,
            laplace_state=laplace_state,
            mc_samples=args.mc_samples,
            laplace_samples=args.laplace_samples,
        )
        k = min(args.acq_size, len(X_p))
        acquired_idx = np.argpartition(pool_scores, -k)[-k:]

        # Save sample grids for MNIST-like vectors
        if input_dim == 784 and scaler is not None:
            synth_dir = out_dir / "synthetic_samples"
            # generated (uncertain)
            X_gen_img = scaler.inverse_transform(X_gen)
            X_gen_img = np.clip(X_gen_img, 0.0, 1.0).reshape(-1, 28, 28)
            save_image_grid(X_gen_img[: min(25, len(X_gen_img))], synth_dir / f"step_{step:02d}_generated.png", title="Generated (uncertain)")
            # sample certain and uncertain real pool images uniformly
            pool_scores_now = pool_scores[: len(X_p)]
            low_t = np.percentile(pool_scores_now, 20)
            high_t = np.percentile(pool_scores_now, 80)
            certain = np.where(pool_scores_now <= low_t)[0]
            uncertain = np.where(pool_scores_now >= high_t)[0]
            if certain.size:
                pick = rng.choice(certain, size=min(25, certain.size), replace=False)
                imgs = scaler.inverse_transform(X_p[pick])
                imgs = np.clip(imgs, 0.0, 1.0).reshape(-1, 28, 28)
                save_image_grid(imgs, synth_dir / f"step_{step:02d}_real_certain.png", title="Real (certain)")
            if uncertain.size:
                pick = rng.choice(uncertain, size=min(25, uncertain.size), replace=False)
                imgs = scaler.inverse_transform(X_p[pick])
                imgs = np.clip(imgs, 0.0, 1.0).reshape(-1, 28, 28)
                save_image_grid(imgs, synth_dir / f"step_{step:02d}_real_uncertain.png", title="Real (uncertain)")

        # Plot frame (before removing acquired)
        X_l_vis = pca.transform(np.nan_to_num(X_l.astype(np.float64)))
        X_p_vis = pca.transform(np.nan_to_num(X_p.astype(np.float64)))
        frame = plot_frame(
            step=step,
            X_l_vis=X_l_vis,
            X_p_vis=X_p_vis,
            pool_scores=pool_scores,
            y_l=y_l,
            acquired_idx=acquired_idx,
            is_synth=is_synth,
            model=main,
            ensemble_models=ensemble_models,
            laplace_state=laplace_state,
            uncertainty_method=uncertainty_method,
            task_type=task_type,
            pca=pca,
            out_dir=out_dir,
            num_classes=num_classes,
            mc_samples=args.mc_samples,
            laplace_samples=args.laplace_samples,
        )
        frames.append(frame)

        # Acquire and retrain
        X_new, y_new = X_p[acquired_idx], y_p[acquired_idx]
        X_l = np.concatenate([X_l, X_new], axis=0)
        y_l = np.concatenate([y_l, y_new], axis=0)
        mask = np.ones(len(X_p), dtype=bool)
        mask[acquired_idx] = False
        X_p, y_p, is_synth = X_p[mask], y_p[mask], is_synth[mask]

        main, ensemble_models, laplace_state = train_main_models(X_l, y_l)

    metrics = evaluate_with_uncertainty_method(
        uncertainty_method,
        task_type=task_type,
        device=device,
        X_test=X_test,
        y_test=y_test,
        model=main,
        ensemble=ensemble_models,
    )
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "initial_label_frac": args.initial_label_frac,
        "uncertainty_method": uncertainty_method,
        "task_type": task_type,
        "num_classes": num_classes,
        "metrics": metrics,
        "frames": [str(p) for p in frames],
        "method": "vae",
        "vae_variant": "latent_interpolation",
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)

    if args.make_gif and frames:
        imgs = [imageio.v2.imread(p) for p in frames]
        imageio.mimsave(out_dir / "active_learning.gif", imgs, duration=0.8)
        log(f"Saved GIF to {out_dir / 'active_learning.gif'}")

    log(f"Done. Metrics: {metrics}")


def parse_args():
    p = argparse.ArgumentParser(description="VAE-latent interpolation active learning runner.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30, help="Epochs for oracle and main model per AL step.")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--initial-label-frac", type=float, default=0.5, help="Initial labeled fraction of the train split.")
    p.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for train samples after split.")
    p.add_argument("--max-test-samples", type=int, default=None, help="Optional cap for test samples after split.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--mc-samples", type=int, default=20)
    p.add_argument("--uncertainty", choices=["dropout", "ensemble", "laplace"], default="dropout")
    p.add_argument("--ensemble-size", type=int, default=5, help="Ensemble size for deep ensembles.")
    p.add_argument("--laplace-prior", type=float, default=1.0, help="Prior precision for diagonal Laplace.")
    p.add_argument("--laplace-samples", type=int, default=30, help="Samples for Laplace predictive approximation.")
    p.add_argument("--acq-steps", type=int, default=5)
    p.add_argument("--acq-size", type=int, default=20)
    p.add_argument("--synthetic-count", type=int, default=200)
    p.add_argument("--vae-epochs", type=int, default=10)
    p.add_argument("--vae-latent-dim", type=int, default=32)
    p.add_argument("--vae-h1", type=int, default=512)
    p.add_argument("--vae-h2", type=int, default=256)
    p.add_argument("--vae-lr", type=float, default=1e-3)
    p.add_argument("--vae-beta", type=float, default=1.0)
    p.add_argument("--make-gif", action="store_true")
    p.add_argument("--output", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
