
import numpy as np


def generate_linear_data(n_samples=500, noise_std=1.0, seed=42):
   
    rng = np.random.default_rng(seed)

    X = rng.uniform(-5, 5, size=(n_samples, 3))
    true_weights = np.array([3.0, -2.0, 0.5])
    true_bias = 5.0

   
    noise = rng.normal(0, noise_std, size=n_samples)
    y = X @ true_weights + true_bias + noise

    return X, y, true_weights, true_bias


def train_test_split(X, y, test_ratio=0.2, seed=42):
   
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    indices = rng.permutation(n)
    test_size = int(n * test_ratio)

    test_idx, train_idx = indices[:test_size], indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
