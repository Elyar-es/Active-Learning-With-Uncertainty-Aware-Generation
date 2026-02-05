"""
Base model class for tabular data (regression and classification)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class TabularMLP(nn.Module):
    """
    Multi-layer perceptron for tabular data.
    Supports both regression and classification tasks.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        task_type: str = "classification",
        num_classes: Optional[int] = None,
        dropout_rate: float = 0.0,
        activation: str = "relu"
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output dimensions
            task_type: "classification" or "regression"
            num_classes: Number of classes (for classification)
            dropout_rate: Dropout probability
            activation: Activation function ("relu", "tanh", "gelu")
        """
        super(TabularMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.num_classes = num_classes if task_type == "classification" else None
        self.dropout_rate = dropout_rate
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        
        if self.task_type == "classification":
            return x  # Logits (softmax will be applied in loss)
        else:
            return x  # Direct regression output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            if self.task_type == "classification":
                return torch.softmax(output, dim=1)
            else:
                return output


class EvidentialMLP(nn.Module):
    """
    Evidential Deep Learning model for uncertainty estimation.
    Uses Dirichlet distribution for classification or Normal-Inverse-Gamma for regression.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        task_type: str = "classification",
        num_classes: Optional[int] = None,
        dropout_rate: float = 0.0,
        activation: str = "relu"
    ):
        super(EvidentialMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.num_classes = num_classes if task_type == "classification" else None
        self.dropout_rate = dropout_rate
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output layer for evidence parameters
        if task_type == "classification":
            # For classification: output evidence parameters (alpha) for Dirichlet
            self.evidence_layer = nn.Linear(hidden_dims[-1], num_classes)
            self.softplus = nn.Softplus()
        else:
            # For regression: output mu, v, alpha, beta for Normal-Inverse-Gamma
            self.evidence_layer = nn.Linear(hidden_dims[-1], output_dim * 4)
            self.softplus = nn.Softplus()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning evidence parameters"""
        x = self.hidden_layers(x)
        evidence = self.evidence_layer(x)
        
        if self.task_type == "classification":
            # Evidence should be positive
            evidence = self.softplus(evidence) + 1.0  # Add 1 for numerical stability
            return evidence
        else:
            # Split into mu, v, alpha, beta
            mu, v, alpha, beta = torch.split(evidence, self.output_dim, dim=1)
            v = self.softplus(v) + 1.0
            alpha = self.softplus(alpha) + 1.0
            beta = self.softplus(beta) + 1.0
            return torch.cat([mu, v, alpha, beta], dim=1)

