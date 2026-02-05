"""
Configuration system for experiments
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration"""
    input_dim: int
    hidden_dims: List[int] = None
    output_dim: int = 1
    task_type: str = "classification"
    num_classes: Optional[int] = None
    dropout_rate: float = 0.0
    activation: str = "relu"
    model_type: str = "standard"  # "standard" or "evidential"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]
        if self.task_type == "classification" and self.num_classes is None:
            raise ValueError("num_classes must be specified for classification")


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    device: str = "cpu"
    lambda_reg: float = 1.0  # For evidential models


@dataclass
class UncertaintyConfig:
    """Uncertainty estimation configuration"""
    method: str = "mc_dropout"  # "mc_dropout", "ensemble", "evidential", "temperature"
    num_samples: int = 100  # For MC Dropout
    num_models: int = 5  # For Ensemble
    temperature: float = 1.0  # For Temperature Scaling
    dropout_rate: float = 0.5  # For MC Dropout


@dataclass
class ActiveLearningConfig:
    """Active learning loop configuration"""
    initial_labeled_ratio: float = 0.1
    acquisition_size: int = 10
    acquisition_steps: int = 5
    use_cgan: bool = True
    cgan_steps: int = 200
    synthetic_per_class: int = 10
    random_state: int = 42


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    dataset_name: str = "iris"
    model_config: ModelConfig = None
    training_config: TrainingConfig = None
    uncertainty_config: UncertaintyConfig = None
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = ModelConfig(input_dim=4, num_classes=3)
        if self.training_config is None:
            self.training_config = TrainingConfig()
        if self.uncertainty_config is None:
            self.uncertainty_config = UncertaintyConfig()
