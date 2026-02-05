"""
Data loaders for well-known tabular datasets
"""
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer,
    fetch_california_housing,
    make_classification, make_regression
)

# load_boston has been removed from scikit-learn 1.2+
# We'll handle it separately
try:
    from sklearn.datasets import load_boston
    BOSTON_AVAILABLE = True
except ImportError:
    BOSTON_AVAILABLE = False
try:
    from torchvision import datasets as tv_datasets, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple, Optional, Dict


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y.dtype in [np.int32, np.int64] else torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DatasetLoader:
    """Loader for various tabular datasets"""
    
    def __init__(
        self,
        dataset_name: str,
        test_size: float = 0.2,
        random_state: int = 42,
        max_train_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
    ):
        """
        Args:
            dataset_name: Name of the dataset
            test_size: Proportion of test set
            random_state: Random seed
            max_train_samples: Optional cap for number of train samples (after split)
            max_test_samples: Optional cap for number of test samples (after split)
        """
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state
        self.max_train_samples = max_train_samples
        self.max_test_samples = max_test_samples
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.task_type = None
        self.num_classes = None
        self.feature_names = None
        self.target_names = None
    
    def load(self) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        Load dataset and return train and test dataloaders
        
        Returns:
            train_loader, test_loader, metadata
        """
        X, y, task_type, num_classes = self._load_raw_data()
        
        self.task_type = task_type
        self.num_classes = num_classes
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, 
            stratify=y if task_type == "classification" else None
        )

        # Optional subsampling for faster experiments (applies after split)
        if self.max_train_samples is not None and len(X_train) > self.max_train_samples:
            X_train, _, y_train, _ = train_test_split(
                X_train,
                y_train,
                train_size=self.max_train_samples,
                random_state=self.random_state,
                stratify=y_train if task_type == "classification" else None,
            )
        if self.max_test_samples is not None and len(X_test) > self.max_test_samples:
            X_test, _, y_test, _ = train_test_split(
                X_test,
                y_test,
                train_size=self.max_test_samples,
                random_state=self.random_state,
                stratify=y_test if task_type == "classification" else None,
            )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Create datasets
        train_dataset = TabularDataset(X_train, y_train)
        test_dataset = TabularDataset(X_test, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        metadata = {
            'input_dim': X_train.shape[1],
            'output_dim': num_classes if task_type == "classification" else 1,
            'num_classes': num_classes,
            'task_type': task_type,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        return train_loader, test_loader, metadata
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray, str, Optional[int]]:
        """Load raw data based on dataset name"""
        
        if self.dataset_name == "iris":
            data = load_iris()
            X, y = data.data, data.target
            self.feature_names = data.feature_names
            self.target_names = data.target_names
            return X, y, "classification", len(np.unique(y))
        
        elif self.dataset_name == "wine":
            data = load_wine()
            X, y = data.data, data.target
            self.feature_names = data.feature_names
            self.target_names = data.target_names
            return X, y, "classification", len(np.unique(y))
        
        elif self.dataset_name == "breast_cancer":
            data = load_breast_cancer()
            X, y = data.data, data.target
            self.feature_names = data.feature_names
            self.target_names = data.target_names
            return X, y, "classification", len(np.unique(y))
        
        elif self.dataset_name == "boston":
            # Boston dataset has been removed from scikit-learn 1.2+
            # Use California housing as alternative
            if not BOSTON_AVAILABLE:
                print("Warning: Boston dataset has been removed from scikit-learn. Using California Housing instead.")
            else:
                try:
                    data = load_boston()
                    X, y = data.data, data.target
                    self.feature_names = data.feature_names
                    return X, y, "regression", None
                except:
                    print("Warning: Boston dataset unavailable. Using California Housing instead.")
            
            # Fallback to California Housing
            data = fetch_california_housing()
            X, y = data.data, data.target
            self.feature_names = data.feature_names
            return X, y, "regression", None
        
        elif self.dataset_name == "california_housing":
            data = fetch_california_housing()
            X, y = data.data, data.target
            self.feature_names = data.feature_names
            return X, y, "regression", None
        
        elif self.dataset_name == "synthetic_classification":
            X, y = make_classification(
                n_samples=1000, n_features=20, n_informative=10,
                n_redundant=5, n_classes=3, random_state=self.random_state
            )
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            self.target_names = [f"class_{i}" for i in range(len(np.unique(y)))]
            return X, y, "classification", len(np.unique(y))
        
        elif self.dataset_name == "synthetic_regression":
            X, y = make_regression(
                n_samples=1000, n_features=20, n_informative=10,
                noise=10.0, random_state=self.random_state
            )
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            return X, y, "regression", None
        
        elif self.dataset_name == "two_moons":
            from sklearn.datasets import make_moons
            X, y = make_moons(n_samples=1000, noise=0.2, random_state=self.random_state)
            self.feature_names = ["feature_1", "feature_2"]
            self.target_names = ["class_0", "class_1"]
            return X, y, "classification", 2
        
        elif self.dataset_name == "circles":
            from sklearn.datasets import make_circles
            X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=self.random_state)
            self.feature_names = ["feature_1", "feature_2"]
            self.target_names = ["class_0", "class_1"]
            return X, y, "classification", 2

        elif self.dataset_name.lower() == "mnist":
            if not TORCHVISION_AVAILABLE:
                raise ImportError("torchvision is required for mnist dataset.")
            mnist_train = tv_datasets.MNIST(
                root="data_cache",
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
            X = mnist_train.data.numpy().astype(np.float32) / 255.0
            X = X.reshape(len(X), -1)
            y = mnist_train.targets.numpy()
            self.feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
            self.target_names = [str(i) for i in range(10)]
            return X, y, "classification", 10

        elif self.dataset_name.lower() in {"fashion_mnist", "fashion-mnist", "fmnist"}:
            if not TORCHVISION_AVAILABLE:
                raise ImportError("torchvision is required for fashion_mnist dataset.")
            ds = tv_datasets.FashionMNIST(
                root="data_cache",
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
            X = ds.data.numpy().astype(np.float32) / 255.0
            X = X.reshape(len(X), -1)
            y = ds.targets.numpy()
            self.feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
            self.target_names = [str(i) for i in range(10)]
            return X, y, "classification", 10

        elif self.dataset_name.lower() in {"cifar10", "cifar-10"}:
            if not TORCHVISION_AVAILABLE:
                raise ImportError("torchvision is required for cifar10 dataset.")
            ds = tv_datasets.CIFAR10(
                root="data_cache",
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
            X = ds.data.astype(np.float32) / 255.0  # [N,32,32,3]
            X = X.transpose(0, 3, 1, 2)  # [N,3,32,32]
            X = X.reshape(len(X), -1)
            y = np.array(ds.targets, dtype=np.int64)
            self.feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
            self.target_names = [str(i) for i in range(10)]
            return X, y, "classification", 10
        
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def get_full_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get full unscaled data for visualization"""
        X, y, _, _ = self._load_raw_data()
        return X, y
