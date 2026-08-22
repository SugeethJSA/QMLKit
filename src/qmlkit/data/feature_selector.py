"""Quantum-Aware Feature Selection and Dimensionality Reduction."""

from __future__ import annotations

import pickle
from typing import Literal, Optional, Tuple
import numpy as np
from sklearn.decomposition import KernelPCA, PCA
from sklearn.feature_selection import mutual_info_classif
import torch
import torch.nn as nn


class LatentAutoencoder(nn.Module):
    """Deep autoencoder to compress sensor features to latent qubit register."""

    def __init__(self, in_features: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, latent_dim),
            nn.Tanh()  # Bounds output to [-1, 1]
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, in_features)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction


class QuantumFeatureSelector:
    """Reduces high-dimensional sensor arrays to match NISQ qubit register capacity."""

    def __init__(
        self,
        n_qubits: int = 6,
        method: Literal["pca", "kpca", "mutual_info", "autoencoder"] = "pca",
        angle_range: Tuple[float, float] = (-np.pi, np.pi)
    ):
        self.n_qubits = n_qubits
        self.method = method
        self.angle_range = angle_range
        self.is_fitted = False

        self.pca_model: Optional[PCA] = None
        self.kpca_model: Optional[KernelPCA] = None
        self.selected_indices: Optional[np.ndarray] = None
        self.autoencoder: Optional[LatentAutoencoder] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> QuantumFeatureSelector:
        """Fit reduction model strictly on training data."""
        n_samples, n_features = X_train.shape
        actual_qubits = min(self.n_qubits, n_features)

        if self.method == "pca":
            self.pca_model = PCA(n_components=actual_qubits, random_state=42)
            self.pca_model.fit(X_train)
        elif self.method == "kpca":
            self.kpca_model = KernelPCA(n_components=actual_qubits, kernel="rbf", random_state=42)
            self.kpca_model.fit(X_train)
        elif self.method == "mutual_info":
            if y_train is None:
                raise ValueError("y_train is required for supervised mutual information selection.")
            mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
            self.selected_indices = np.argsort(mi_scores)[::-1][:actual_qubits]
        elif self.method == "autoencoder":
            self._train_autoencoder(X_train, actual_qubits)

        self.is_fitted = True
        return self

    def _train_autoencoder(self, X_train: np.ndarray, latent_dim: int, epochs: int = 40) -> None:
        """Train compression autoencoder."""
        in_dim = X_train.shape[1]
        self.autoencoder = LatentAutoencoder(in_dim, latent_dim)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        tensor_x = torch.tensor(X_train, dtype=torch.float32)
        self.autoencoder.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            _, recon = self.autoencoder(tensor_x)
            loss = criterion(recon, tensor_x)
            loss.backward()
            optimizer.step()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features into angle-scaled quantum state parameters."""
        if not self.is_fitted:
            raise RuntimeError("Selector must be fitted before transforming.")

        if self.method == "pca":
            assert self.pca_model is not None
            z = self.pca_model.transform(X)
        elif self.method == "kpca":
            assert self.kpca_model is not None
            z = self.kpca_model.transform(X)
        elif self.method == "mutual_info":
            assert self.selected_indices is not None
            z = X[:, self.selected_indices]
        elif self.method == "autoencoder":
            assert self.autoencoder is not None
            self.autoencoder.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(X, dtype=torch.float32)
                z = self.autoencoder.encoder(tensor_x).numpy()

        # Scale into bounded quantum rotation angles [angle_min, angle_max]
        z_min, z_max = np.min(z), np.max(z)
        if z_max > z_min:
            z_scaled = (z - z_min) / (z_max - z_min)  # [0, 1]
            low, high = self.angle_range
            return low + z_scaled * (high - low)
        return z

    def fit_transform(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform training features."""
        return self.fit(X_train, y_train).transform(X_train)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """Map latent quantum space back to original chemical feature space for explainability."""
        if not self.is_fitted:
            raise RuntimeError("Selector must be fitted before inverse transforming.")

        if self.method == "pca" and self.pca_model is not None:
            return self.pca_model.inverse_transform(z)
        elif self.method == "autoencoder" and self.autoencoder is not None:
            self.autoencoder.eval()
            with torch.no_grad():
                tensor_z = torch.tensor(z, dtype=torch.float32)
                return self.autoencoder.decoder(tensor_z).numpy()
        else:
            raise NotImplementedError(f"Inverse transform not supported for method: {self.method}")

    def save(self, filepath: str) -> None:
        """Save selector to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> QuantumFeatureSelector:
        """Load selector from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
