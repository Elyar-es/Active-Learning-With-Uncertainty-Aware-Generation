"""
Training utilities for models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, List
import numpy as np


class Trainer:
    """Trainer class for model training"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        task_type: str = "classification",
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            task_type: "classification" or "regression"
            device: Device to use
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_type = task_type
        self.device = device
        
        # Loss function
        if task_type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for X, y in tqdm(self.train_loader, desc="Training", leave=False):
            X, y = X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(X)
            
            if self.task_type == "classification":
                loss = self.criterion(output, y)
            else:
                if len(y.shape) == 1:
                    y = y.unsqueeze(1)
                loss = self.criterion(output, y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self) -> float:
        """Validate model"""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                output = self.model(X)
                
                if self.task_type == "classification":
                    loss = self.criterion(output, y)
                else:
                    if len(y.shape) == 1:
                        y = y.unsqueeze(1)
                    loss = self.criterion(output, y)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, num_epochs: int = 100, verbose: bool = True) -> Dict:
        """Train model for multiple epochs"""
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }


class EvidentialTrainer(Trainer):
    """Trainer for evidential models"""
    
    def __init__(self, *args, lambda_reg: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        from uncertainty.losses import EvidentialLoss
        self.criterion = EvidentialLoss(
            task_type=self.task_type,
            lambda_reg=lambda_reg
        )
    
    def train_epoch(self) -> float:
        """Train for one epoch with evidential loss"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for X, y in tqdm(self.train_loader, desc="Training", leave=False):
            X, y = X.to(self.device), y.to(self.device)
            
            # Handle regression targets (ensure 2D)
            if self.task_type == "regression" and len(y.shape) == 1:
                y = y.unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            evidence = self.model(X)
            loss = self.criterion(evidence, y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self) -> float:
        """Validate evidential model"""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                # Handle regression targets (ensure 2D)
                if self.task_type == "regression" and len(y.shape) == 1:
                    y = y.unsqueeze(1)
                
                evidence = self.model(X)
                loss = self.criterion(evidence, y)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss

