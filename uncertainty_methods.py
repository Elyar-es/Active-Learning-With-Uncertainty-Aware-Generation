"""
Shared uncertainty-estimation utilities for the experiment runner scripts.

Supported methods:
- dropout: MC dropout (entropy for classification, predictive variance for regression)
- ensemble: deep ensembles (entropy for classification, predictive variance for regression)
- laplace: diagonal Laplace approximation on the last layer (sampled predictive entropy/variance)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LaplaceState:
    var_w: torch.Tensor  # [out_dim, hidden_dim]
    var_b: torch.Tensor  # [out_dim]


def _to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)


def fit_laplace_last_layer_diag(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    device: str,
    prior_precision: float = 1.0,
    batch_size: int = 1024,
) -> LaplaceState:
    """
    Fit a diagonal Laplace approximation for TabularMLP's output layer parameters only.

    This uses a diagonal Hessian approximation of the negative log-likelihood w.r.t. the
    output-layer weights/biases.
    """
    if not hasattr(model, "hidden_layers") or not hasattr(model, "output_layer"):
        raise ValueError("Laplace estimator expects a TabularMLP-like model with hidden_layers/output_layer.")

    model.eval()
    out_dim = int(model.output_layer.out_features)
    hidden_dim = int(model.output_layer.in_features)

    precision_w = torch.full((out_dim, hidden_dim), float(prior_precision), device=device)
    precision_b = torch.full((out_dim,), float(prior_precision), device=device)

    Xt = _to_tensor(X, device=device)
    yt = torch.tensor(y, device=device)

    n = Xt.shape[0]
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        xb = Xt[start:end]
        yb = yt[start:end]
        with torch.no_grad():
            feats = model.hidden_layers(xb)  # [B, hidden_dim]
            logits = model.output_layer(feats)  # [B, out_dim]
            feats2 = feats.pow(2)  # [B, hidden_dim]

            if task_type == "classification":
                probs = torch.softmax(logits, dim=1)  # [B, C]
                w = probs * (1.0 - probs)  # [B, C]
                precision_w = precision_w + (w.T @ feats2)
                precision_b = precision_b + w.sum(dim=0)
            else:
                # MSE loss: diagonal Hessian wrt weights is proportional to sum(phi^2)
                # Using a constant scaling (2) is enough for ranking-based uncertainty.
                feats2_sum = feats2.sum(dim=0)  # [hidden_dim]
                precision_w[0] = precision_w[0] + 2.0 * feats2_sum
                precision_b[0] = precision_b[0] + 2.0 * float(end - start)

    var_w = 1.0 / (precision_w + 1e-8)
    var_b = 1.0 / (precision_b + 1e-8)
    return LaplaceState(var_w=var_w, var_b=var_b)


@torch.no_grad()
def predict_mean_and_uncertainty(
    method: str,
    task_type: str,
    device: str,
    X: np.ndarray,
    model: Optional[nn.Module] = None,
    *,
    ensemble: Optional[List[nn.Module]] = None,
    laplace_state: Optional[LaplaceState] = None,
    mc_samples: int = 20,
    laplace_samples: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      mean_pred:
        - classification: [N, C] probabilities
        - regression: [N] predictions
      unc:
        - classification: entropy(mean_pred)
        - regression: predictive variance (or std^2)
    """
    method = method.lower()
    if method not in {"dropout", "ensemble", "laplace"}:
        raise ValueError(f"Unknown uncertainty method: {method}")

    if method == "ensemble":
        if not ensemble:
            raise ValueError("Ensemble method requires a non-empty 'ensemble' list.")
        preds = []
        for m in ensemble:
            m.eval()
            out = m(_to_tensor(X, device=device))
            if task_type == "classification":
                preds.append(torch.softmax(out, dim=1))
            else:
                preds.append(out.squeeze())
        stacked = torch.stack(preds, dim=0)
        mean = stacked.mean(dim=0)
        if task_type == "classification":
            unc = _entropy_from_probs(mean)
            return mean.cpu().numpy(), unc.cpu().numpy()
        unc = torch.var(stacked, dim=0)
        return mean.cpu().numpy(), unc.cpu().numpy()

    if model is None:
        raise ValueError(f"Method '{method}' requires 'model' to be provided.")

    Xt = _to_tensor(X, device=device)

    if method == "dropout":
        model.train()  # enable dropout
        preds = []
        for _ in range(mc_samples):
            out = model(Xt)
            if task_type == "classification":
                preds.append(torch.softmax(out, dim=1))
            else:
                preds.append(out.squeeze())
        stacked = torch.stack(preds, dim=0)
        mean = stacked.mean(dim=0)
        if task_type == "classification":
            unc = _entropy_from_probs(mean)
            model.eval()
            return mean.cpu().numpy(), unc.cpu().numpy()
        unc = torch.var(stacked, dim=0)
        model.eval()
        return mean.cpu().numpy(), unc.cpu().numpy()

    # laplace
    if laplace_state is None:
        raise ValueError("Laplace method requires 'laplace_state' to be provided.")
    if not hasattr(model, "hidden_layers") or not hasattr(model, "output_layer"):
        raise ValueError("Laplace estimator expects a TabularMLP-like model with hidden_layers/output_layer.")

    model.eval()
    feats = model.hidden_layers(Xt)
    logits_mean = model.output_layer(feats)
    feats2 = feats.pow(2)
    var_logits = feats2 @ laplace_state.var_w.T + laplace_state.var_b
    var_logits = torch.clamp(var_logits, min=1e-10)

    if task_type == "classification":
        eps = torch.randn((laplace_samples, logits_mean.shape[0], logits_mean.shape[1]), device=device)
        logits_s = logits_mean.unsqueeze(0) + torch.sqrt(var_logits).unsqueeze(0) * eps
        probs_s = torch.softmax(logits_s, dim=2)
        mean_probs = probs_s.mean(dim=0)
        unc = _entropy_from_probs(mean_probs)
        return mean_probs.cpu().numpy(), unc.cpu().numpy()

    eps = torch.randn((laplace_samples, logits_mean.shape[0]), device=device)
    preds_s = logits_mean.squeeze().unsqueeze(0) + torch.sqrt(var_logits.squeeze()).unsqueeze(0) * eps
    mean_pred = preds_s.mean(dim=0)
    unc = torch.var(preds_s, dim=0)
    return mean_pred.cpu().numpy(), unc.cpu().numpy()

