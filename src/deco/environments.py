# src/deco/environments.py
import os

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_linnerud,
    load_svmlight_file,
    load_wine,
)


def normalize_data(X: np.ndarray) -> np.ndarray:
    """
    Normalize input vectors to have L2 norm equal to 1.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    Returns
    -------
    X_normalized : np.ndarray
        Normalized feature matrix of shape (n_samples, n_features).
    """
    norms = np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1  # Prevent division by zero
    X_normalized = X / norms
    return X_normalized


def standardize_features(X: np.ndarray) -> np.ndarray:
    """Center and scale each feature with only NumPy dependencies."""
    X = np.nan_to_num(np.asarray(X, dtype=float))
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (X - mean) / std


def standardize_targets(y: np.ndarray) -> np.ndarray:
    """Normalize targets to comparable scale across real-data experiments."""
    y = np.asarray(y, dtype=float)
    if y.ndim > 1:
        # Linnerud has multiple targets; use one target to keep the OLO task scalar.
        y = y[:, 0]
    y_std = np.std(y)
    if y_std > 1e-12:
        y = (y - np.mean(y)) / y_std
    return y


class SyntheticRegression:
    """Generates heterogeneous data for an online regression task."""

    def __init__(self, N, dim, u_star, noise_std=0.1, heterogeneity_scale=2.0):
        self.N = N
        self.dim = dim
        self.u_star = u_star
        self.noise_std = noise_std
        # Create a unique origin for each agent's feature distribution
        self.origins = np.random.randn(self.N, self.dim) * heterogeneity_scale

    def get_context(self, t, decisions):
        # All agents observe a common base feature vector
        base_features = np.random.randn(self.dim)

        agent_features = []
        for i in range(self.N):
            # Create a personalized feature vector by adding the agent's origin
            feat = base_features + self.origins[i]
            # Normalize to ensure gradient norm is bounded
            feat /= np.linalg.norm(feat)
            agent_features.append(feat)

        true_labels = [
            np.dot(feat, self.u_star) + np.random.normal(0, self.noise_std)
            for feat in agent_features
        ]

        losses = np.zeros(self.N)
        gradients = np.zeros((self.N, self.dim))

        for i in range(self.N):
            pred = np.dot(agent_features[i], decisions[i])
            losses[i] = np.abs(pred - true_labels[i])
            grad_sign = np.sign(pred - true_labels[i])
            gradients[i] = grad_sign * agent_features[i]

        return losses, gradients, agent_features, true_labels


class RealDataEnvironment:
    """Streams various regression datasets for online learning tasks."""

    DATASETS = {
        "YearPredictionMSD": {
            "loader": None,  # Will load from LIBSVM file
            "task_type": "regression",
            "description": "Year prediction from audio features dataset",
            "file_path": "data/YearPredictionMSD",
        },
        "cpusmall": {
            "loader": None,  # Will load from LIBSVM file
            "task_type": "regression",
            "description": "Computer activity dataset",
            "file_path": "data/cpusmall",
        },
        "cadata": {
            "loader": None,  # Will load from LIBSVM file
            "task_type": "regression",
            "description": "California housing dataset (LIBSVM format)",
            "file_path": "data/cadata",
        },
        "space_ga": {
            "loader": None,  # Will load from LIBSVM file
            "task_type": "regression",
            "description": "Space GA dataset",
            "file_path": "data/space_ga",
        },
        "abalone": {
            "loader": None,  # Will load from LIBSVM file
            "task_type": "regression",
            "description": "Predict the age of abalone from physical measurements",
            "file_path": "data/abalone",
        },
        "E2006.train": {
            "loader": None,  # Will load from LIBSVM file
            "task_type": "regression",
            "description": "Finance data.",
            "file_path": "data/E2006",
        },
        # Offline-friendly scikit-learn datasets used by the review-response tests.
        # They preserve the same absolute-loss OLO interface and make the Docker
        # sanity checks independent of external dataset mirrors.
        "diabetes": {
            "loader": load_diabetes,
            "task_type": "regression",
            "description": "Diabetes progression regression dataset bundled with scikit-learn",
            "file_path": None,
        },
        "linnerud": {
            "loader": load_linnerud,
            "task_type": "regression",
            "description": "Linnerud exercise/physiology dataset bundled with scikit-learn",
            "file_path": None,
        },
        "breast_cancer": {
            "loader": load_breast_cancer,
            "task_type": "binary-as-regression",
            "description": "Breast cancer diagnostic dataset bundled with scikit-learn",
            "file_path": None,
        },
        "wine": {
            "loader": load_wine,
            "task_type": "label-regression",
            "description": "Wine recognition dataset bundled with scikit-learn",
            "file_path": None,
        },
        "digits": {
            "loader": load_digits,
            "task_type": "label-regression",
            "description": "Handwritten digits dataset bundled with scikit-learn",
            "file_path": None,
        },
    }

    def __init__(self, N, dataset="cadata", seed=0):
        """
        Initialize environment with a specific regression dataset.

        Args:
            N: Number of agents
            dataset: Dataset name from DATASETS dict
            seed: Random seed for stream ordering
        """
        self.N = N
        self.dataset_name = dataset
        self.rng = np.random.default_rng(seed)

        if dataset not in self.DATASETS:
            available = list(self.DATASETS.keys())
            raise ValueError(f"Unknown dataset '{dataset}'. Available: {available}")

        dataset_info = self.DATASETS[dataset]
        self.task_type = dataset_info["task_type"]

        # Load the dataset
        if dataset_info["loader"] is not None:
            loader = dataset_info["loader"]
            data = loader()
            self.X = np.asarray(data.data, dtype=float)
            self.y = standardize_targets(data.target)
        else:
            # LIBSVM format datasets
            file_path = dataset_info["file_path"]
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Dataset file {file_path} not found. "
                    f"Please run download_datasets.py first."
                )
            X_sparse, self.y = load_svmlight_file(file_path)[:2]
            try:
                self.X = X_sparse.toarray()
            except AttributeError:
                self.X = np.asarray(X_sparse)
            self.y = standardize_targets(self.y)

        # Standardize and then normalize features to ensure bounded gradients.
        self.X = normalize_data(standardize_features(self.X))
        self.n_samples, self.dim = self.X.shape

        # Compute optimal parameters using least squares for regression.
        self.u_star = np.linalg.lstsq(self.X, self.y, rcond=None)[0]

        self.indices = np.arange(self.n_samples)
        self.rng.shuffle(self.indices)

    def get_context(self, t, decisions):
        """Get context for time step t."""
        # At each step, each agent gets one unique sample. We wrap and reshuffle
        # at epoch boundaries so the bundled datasets can support arbitrary T.
        start = (t * self.N) % self.n_samples
        if start + self.N <= self.n_samples:
            batch_indices = self.indices[start : start + self.N]
        else:
            self.rng.shuffle(self.indices)
            overflow = start + self.N - self.n_samples
            batch_indices = np.concatenate([self.indices[start:], self.indices[:overflow]])

        agent_features = self.X[batch_indices]
        agent_labels = self.y[batch_indices]

        losses = np.zeros(self.N)
        gradients = np.zeros((self.N, self.dim))

        for i in range(self.N):
            features = agent_features[i]
            label = agent_labels[i]
            pred = np.dot(features, decisions[i])

            # Absolute loss for regression
            losses[i] = np.abs(pred - label)
            grad_sign = np.sign(pred - label)
            grad = grad_sign * features

            grad_norm = np.linalg.norm(grad)
            assert grad_norm <= (
                1 + 1e-4
            ), f"Grad norm larger than one! It is {grad_norm}"
            gradients[i] = grad

        return losses, gradients, agent_features, agent_labels

    @classmethod
    def list_datasets(cls):
        """List all available datasets with descriptions."""
        for name, info in cls.DATASETS.items():
            print(f"'{name}': {info['description']} ({info['task_type']})")
