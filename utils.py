"""
Utility functions
"""
import torch
import numpy as np
from typing import Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_device() -> str:
    """Get available device"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def reduce_dimensions(
    X: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce dimensionality for visualization
    
    Args:
        X: Input data [n_samples, n_features]
        method: "pca" or "tsne"
        n_components: Number of components
        random_state: Random seed
    
    Returns:
        Reduced data [n_samples, n_components]
    """
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=30)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return reducer.fit_transform(X)


def create_mesh(
    X: np.ndarray,
    n_points: int = 100,
    margin: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create mesh for decision boundary visualization
    
    Args:
        X: Input data [n_samples, n_features] (should be 2D after PCA)
        n_points: Number of points per dimension
        margin: Margin around data
    
    Returns:
        xx, yy: Mesh grid
        mesh_points: Flattened mesh points [n_points^2, 2]
    """
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n_points),
        np.linspace(y_min, y_max, n_points)
    )
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    return xx, yy, mesh_points

