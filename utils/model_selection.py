import mlx.core as mx
import numpy as np
import pandas as pd


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into random train and test subsets

    Args:
        X (mx.array): Features array
        y (mx.array): Target array
        test_size (float): Proportion of dataset to include in test split
        random_seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Handle pandas DataFrame/Series
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy().astype(np.float32)
    if isinstance(y, pd.Series):
        y = y.to_numpy().astype(np.float32)
        
    # Convert numpy arrays to MLX arrays
    if isinstance(X, np.ndarray):
        X = mx.array(X.astype(np.float32))
    if isinstance(y, np.ndarray):
        y = mx.array(y.astype(np.float32))
    
    # Convert MLX arrays to numpy for shuffling
    if isinstance(X, mx.array):
        X = np.array(X.tolist(), dtype=np.float32)
    if isinstance(y, mx.array):
        y = np.array(y.tolist(), dtype=np.float32)
    
    X = np.array(X)
    y = np.array(y)

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Get number of samples
    n_samples = X.shape[0]

    # Calculate number of test samples
    n_test = int(n_samples * test_size)

    # Generate random indices
    indices = np.random.permutation(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Split the data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return (mx.array(X_train), mx.array(X_test), mx.array(y_train), mx.array(y_test))