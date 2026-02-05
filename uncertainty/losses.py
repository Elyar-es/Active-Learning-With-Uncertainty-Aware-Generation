"""
Loss functions for uncertainty estimation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def evidential_classification_loss(alpha: torch.Tensor, y: torch.Tensor, lambda_reg: float = 1.0):
    """
    Evidential loss for classification (Dirichlet)
    
    Args:
        alpha: Evidence parameters [batch_size, num_classes]
        y: True labels [batch_size] (class indices)
        lambda_reg: Regularization strength
    """
    alpha0 = alpha.sum(dim=1)
    
    # Likelihood term
    y_one_hot = F.one_hot(y, num_classes=alpha.shape[1]).float()
    log_likelihood = torch.sum(
        y_one_hot * (torch.digamma(alpha) - torch.digamma(alpha0.unsqueeze(1))),
        dim=1
    )
    
    # Regularization term (penalize high evidence)
    reg = lambda_reg * torch.sum((alpha - 1) * (1 - y_one_hot), dim=1)
    
    loss = -log_likelihood.mean() + reg.mean()
    return loss


def evidential_regression_loss(evidence: torch.Tensor, y: torch.Tensor, lambda_reg: float = 0.01):
    """
    Evidential loss for regression (Normal-Inverse-Gamma)
    
    Args:
        evidence: [mu, v, alpha, beta] concatenated [batch_size, 4 * output_dim]
        y: True targets [batch_size, output_dim]
        lambda_reg: Regularization strength
    """
    output_dim = y.shape[1]
    mu, v, alpha, beta = torch.split(evidence, output_dim, dim=1)
    
    # Likelihood term
    two_blambda = 2 * beta * (1 + v)
    pi_tensor = torch.tensor(3.141592653589793, device=evidence.device, dtype=evidence.dtype)
    nll = 0.5 * torch.log(pi_tensor / v) - alpha * torch.log(two_blambda) + \
          (alpha + 0.5) * torch.log((y - mu) ** 2 * v + two_blambda) + \
          torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    
    # Regularization term
    reg = lambda_reg * torch.abs(mu - y) * (2 * v + alpha)
    
    loss = nll.mean() + reg.mean()
    return loss


class EvidentialLoss(nn.Module):
    """Wrapper for evidential losses"""
    
    def __init__(self, task_type: str = "classification", lambda_reg: float = 1.0):
        super().__init__()
        self.task_type = task_type
        self.lambda_reg = lambda_reg
    
    def forward(self, evidence: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.task_type == "classification":
            return evidential_classification_loss(evidence, y, self.lambda_reg)
        else:
            return evidential_regression_loss(evidence, y, self.lambda_reg)

