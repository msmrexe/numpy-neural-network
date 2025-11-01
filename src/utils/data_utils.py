import numpy as np
import logging
from typing import Tuple, List, Optional
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logger
logger = logging.getLogger(__name__)

def load_mnist(validation_split: float = 0.1) -> Tuple[np.ndarray, ...]:
    """
    Fetches and prepares the original MNIST dataset (digits).
    
    Data is flattened to (N, 784) and normalized.

    Args:
        validation_split (float): Fraction of training data to use for validation.

    Returns:
        A tuple (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    logger.info("Loading MNIST (Digits) dataset...")
    try:
        # Using sklearn's fetch_openml which is more reliable
        mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False, parser='auto')
        X = mnist.data.astype(np.float32)
        y = mnist.target.astype(np.uint8)

        # Normalize pixel values to [0, 1]
        X = X / 255.0

        # Split into standard 60k train / 10k test
        X_train_full, X_test = X[:60000], X[60000:]
        y_train_full, y_test = y[:60000], y[60000:]

        # Split training into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=validation_split, 
            random_state=42, 
            stratify=y_train_full
        )
        
        logger.info(f"MNIST loaded. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test

    except Exception as e:
        logger.error(f"Error loading MNIST dataset: {e}")
        # Fallback to empty arrays
        return (np.array([]),) * 6

def load_fashion_mnist(filter_classes: Optional[List[int]] = None) -> Tuple[np.ndarray, ...]:
    """
    Fetches and prepares the Fashion-MNIST dataset.

    Data is flattened to (N, 784) and normalized to [-1, 1].

    Args:
        filter_classes (Optional[List[int]]): A list of class indices to keep.
                                              If None, all 10 classes are kept.

    Returns:
        A tuple (X_train, y_train, X_test, y_test).
    """
    logger.info("Loading Fashion-MNIST dataset...")
    try:
        fashion_mnist = fetch_openml("Fashion-MNIST", version=1, cache=True, as_frame=False, parser='auto')
        X = fashion_mnist.data.astype(np.float32)
        y = fashion_mnist.target.astype(np.uint8)
        
        # Normalize the pixels to be in [-1, +1] range
        X = ((X / 255.0) - 0.5) * 2.0

        if filter_classes is not None and len(filter_classes) > 0:
            logger.info(f"Filtering Fashion-MNIST for classes: {filter_classes}")
            mask = np.isin(y, filter_classes)
            X, y = X[mask], y[mask]
            
            # Remap labels to be contiguous (0, 1, 2, ...)
            label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(filter_classes))}
            y = np.array([label_map[old] for old in y], dtype=np.uint8)

        # Split into standard 60k train / 10k test
        X_train_full, X_test = X[:60000], X[60000:]
        y_train_full, y_test = y[:60000], y[60000:]
        
        # Split training into train and validation (using a fixed split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=0.1, 
            random_state=42, 
            stratify=y_train_full
        )

        logger.info(f"Fashion-MNIST loaded. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test

    except Exception as e:
        logger.error(f"Error loading Fashion-MNIST dataset: {e}")
        return (np.array([]),) * 6

def load_california_housing(validation_split: float = 0.1, test_split: float = 0.1) -> Tuple[np.ndarray, ...]:
    """
    Fetches, splits, and standard-scales the California Housing dataset.

    Args:
        validation_split (float): Fraction of data for validation.
        test_split (float): Fraction of data for testing.

    Returns:
        A tuple (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    logger.info("Loading California Housing dataset...")
    try:
        housing = fetch_california_housing()
        X, y = housing.data.astype(np.float32), housing.target.astype(np.float32)
        
        # Reshape y to be (N, 1) for consistency
        y = y.reshape(-1, 1)

        # First split: separate train from (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=(validation_split + test_split), 
            random_state=42
        )
        
        # Second split: separate val and test
        # Need to recalculate proportion
        val_test_proportion = test_split / (validation_split + test_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=val_test_proportion, 
            random_state=42
        )

        # Standardize features (Z-score normalization)
        # Fit scaler ONLY on training data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        # Transform val and test data with the same scaler
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        logger.info(f"California Housing loaded. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test

    except Exception as e:
        logger.error(f"Error loading California Housing dataset: {e}")
        return (np.array([]),) * 6
