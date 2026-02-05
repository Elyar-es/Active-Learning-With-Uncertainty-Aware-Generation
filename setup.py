"""
Setup script to verify installation and test imports
"""
import sys

def check_imports():
    """Check if all required modules can be imported"""
    print("Checking imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy not found: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ scikit-learn not found: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib not found: {e}")
        return False
    
    # Check project modules
    print("\nChecking project modules...")
    try:
        from data import DatasetLoader
        print("✓ data module")
    except ImportError as e:
        print(f"✗ data module: {e}")
        return False
    
    try:
        from models import TabularMLP, EvidentialMLP
        print("✓ models module")
    except ImportError as e:
        print(f"✗ models module: {e}")
        return False
    
    try:
        from uncertainty import MCDropoutEstimator, EnsembleEstimator, EvidentialEstimator
        print("✓ uncertainty module")
    except ImportError as e:
        print(f"✗ uncertainty module: {e}")
        return False
    
    try:
        from trainer import Trainer, EvidentialTrainer
        print("✓ trainer module")
    except ImportError as e:
        print(f"✗ trainer module: {e}")
        return False
    
    try:
        from utils import get_device
        print("✓ utils module")
    except ImportError as e:
        print(f"✗ utils module: {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from data import DatasetLoader
        from models import TabularMLP
        from utils import get_device
        
        # Test dataset loading
        loader = DatasetLoader(dataset_name="iris", test_size=0.2, random_state=42)
        train_loader, test_loader, metadata = loader.load()
        print(f"✓ Dataset loaded: {metadata['task_type']}, {metadata['input_dim']} features")
        
        # Test model creation
        model = TabularMLP(
            input_dim=metadata['input_dim'],
            hidden_dims=[32, 16],
            output_dim=metadata['num_classes'],
            task_type="classification",
            num_classes=metadata['num_classes']
        )
        print(f"✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test device
        device = get_device()
        print(f"✓ Device: {device}")
        
        print("\n✓ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Uncertainty Estimation Framework - Setup Check")
    print("=" * 50)
    
    if check_imports():
        if test_basic_functionality():
            print("\n" + "=" * 50)
            print("✓ Setup complete! Everything is working.")
            print("=" * 50)
            sys.exit(0)
        else:
            print("\n" + "=" * 50)
            print("✗ Setup incomplete. Please check errors above.")
            print("=" * 50)
            sys.exit(1)
    else:
        print("\n" + "=" * 50)
        print("✗ Setup incomplete. Please install missing dependencies.")
        print("=" * 50)
        sys.exit(1)

