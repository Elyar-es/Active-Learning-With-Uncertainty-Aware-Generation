"""
Uncertainty estimation methods for deep learning models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict
from abc import ABC, abstractmethod


class UncertaintyEstimator(ABC):
    """Base class for uncertainty estimators"""
    
    @abstractmethod
    def estimate(self, model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Estimate uncertainty for given inputs
        
        Returns:
            Dictionary with keys:
            - 'mean': Mean prediction
            - 'std' or 'var': Uncertainty measure
            - 'entropy': Entropy (for classification)
            - 'aleatoric': Aleatoric uncertainty
            - 'epistemic': Epistemic uncertainty
        """
        pass


class MCDropoutEstimator(UncertaintyEstimator):
    """
    Monte Carlo Dropout for uncertainty estimation
    """
    
    def __init__(self, num_samples: int = 100, dropout_rate: float = 0.5):
        """
        Args:
            num_samples: Number of Monte Carlo samples
            dropout_rate: Dropout rate to use during inference
        """
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
    
    def estimate(self, model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate uncertainty using MC Dropout"""
        model.train()  # Enable dropout
        
        predictions = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred = model(x)
                if hasattr(model, 'task_type') and model.task_type == "classification":
                    pred = F.softmax(pred, dim=1)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [num_samples, batch_size, output_dim]
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        var = predictions.var(dim=0)
        
        result = {
            'mean': mean,
            'std': std,
            'var': var
        }
        
        # For classification, compute entropy
        if len(mean.shape) > 1 and mean.shape[1] > 1:
            entropy = -torch.sum(mean * torch.log(mean + 1e-10), dim=1)
            result['entropy'] = entropy
        
        # Epistemic uncertainty (variance of predictions)
        if len(std.shape) > 1:
            epistemic = std.mean(dim=1) if len(std.shape) > 1 else std
            result['epistemic'] = epistemic
        
        model.eval()
        return result


class EnsembleEstimator(UncertaintyEstimator):
    """
    Deep Ensemble for uncertainty estimation
    """
    
    def __init__(self, models: List[nn.Module]):
        """
        Args:
            models: List of trained models
        """
        self.models = models
    
    def estimate(self, model: Optional[nn.Module], x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate uncertainty using ensemble"""
        predictions = []
        
        for m in self.models:
            m.eval()
            with torch.no_grad():
                pred = m(x)
                if hasattr(m, 'task_type') and m.task_type == "classification":
                    pred = F.softmax(pred, dim=1)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [num_models, batch_size, output_dim]
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        var = predictions.var(dim=0)
        
        result = {
            'mean': mean,
            'std': std,
            'var': var
        }
        
        # For classification, compute entropy
        if len(mean.shape) > 1 and mean.shape[1] > 1:
            entropy = -torch.sum(mean * torch.log(mean + 1e-10), dim=1)
            result['entropy'] = entropy
        
        # Epistemic uncertainty
        if len(std.shape) > 1:
            epistemic = std.mean(dim=1) if len(std.shape) > 1 else std
            result['epistemic'] = epistemic
        
        return result


class EvidentialEstimator(UncertaintyEstimator):
    """
    Evidential Deep Learning uncertainty estimation
    """
    
    def estimate(self, model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate uncertainty using evidential learning"""
        model.eval()
        with torch.no_grad():
            evidence = model(x)
        
        if hasattr(model, 'task_type') and model.task_type == "classification":
            # Classification: Dirichlet distribution
            alpha = evidence  # Evidence parameters
            alpha0 = alpha.sum(dim=1, keepdim=True)  # Total evidence
            
            # Mean prediction (expected probability)
            mean = alpha / alpha0
            
            # Uncertainty measures
            entropy = self._dirichlet_entropy(alpha)
            mutual_info = self._dirichlet_mutual_info(alpha, alpha0)
            
            # Aleatoric uncertainty (expected entropy)
            aleatoric = entropy
            # Epistemic uncertainty (mutual information)
            epistemic = mutual_info
            
            result = {
                'mean': mean,
                'entropy': entropy,
                'aleatoric': aleatoric,
                'epistemic': epistemic,
                'evidence': alpha0.squeeze()
            }
        else:
            # Regression: Normal-Inverse-Gamma distribution
            output_dim = model.output_dim
            mu, v, alpha, beta = torch.split(evidence, output_dim, dim=1)
            
            # Mean prediction
            mean = mu
            
            # Uncertainty measures
            # Aleatoric: expected variance
            aleatoric = beta / (alpha - 1)
            # Epistemic: variance of mean
            epistemic = beta / ((alpha - 1) * v)
            
            # Total uncertainty
            total_var = aleatoric + epistemic
            std = torch.sqrt(total_var)
            
            result = {
                'mean': mean,
                'std': std,
                'var': total_var,
                'aleatoric': aleatoric.mean(dim=1) if len(aleatoric.shape) > 1 else aleatoric,
                'epistemic': epistemic.mean(dim=1) if len(epistemic.shape) > 1 else epistemic
            }
        
        return result
    
    def _dirichlet_entropy(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute entropy of Dirichlet distribution"""
        alpha0 = alpha.sum(dim=1)
        entropy = torch.sum(
            torch.lgamma(alpha) - (alpha - 1) * torch.digamma(alpha),
            dim=1
        ) - torch.lgamma(alpha0) - (alpha0 - alpha.shape[1]) * torch.digamma(alpha0)
        return entropy
    
    def _dirichlet_mutual_info(self, alpha: torch.Tensor, alpha0: torch.Tensor) -> torch.Tensor:
        """Compute mutual information (epistemic uncertainty)"""
        num_classes = alpha.shape[1]
        mutual_info = torch.digamma(alpha + 1) - torch.digamma(alpha0 + 1)
        mutual_info = torch.sum(alpha / alpha0 * mutual_info, dim=1)
        return mutual_info


class TemperatureScalingEstimator(UncertaintyEstimator):
    """
    Temperature Scaling for calibration and uncertainty estimation
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Temperature parameter (learned or fixed)
        """
        self.temperature = temperature
    
    def estimate(self, model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate uncertainty using temperature scaling"""
        model.eval()
        with torch.no_grad():
            logits = model(x)
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=1)
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        
        result = {
            'mean': probs,
            'entropy': entropy,
            'confidence': probs.max(dim=1)[0]
        }
        
        return result

