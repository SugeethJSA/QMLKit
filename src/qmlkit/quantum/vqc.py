"""Variational Quantum Classifier (VQC) and Hybrid PyTorch Quantum Layers."""

from __future__ import annotations

import pickle
from typing import List, Literal, Optional, Tuple
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from qmlkit.quantum.feature_maps import BaseFeatureMap, BioZZFeatureMap, get_feature_map


class VariationalQuantumClassifier:
    """Variational Quantum Classifier (VQC) with Parameterized Quantum Circuits."""

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 2,
        feature_map_type: str = "BioZZ",
        ansatz_type: Literal["StronglyEntangling", "RealAmplitudes"] = "StronglyEntangling",
        learning_rate: float = 0.03,
        epochs: int = 35,
        covariance_matrix: Optional[np.ndarray] = None,
        device_name: str = "default.qubit"
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.feature_map_type = feature_map_type
        self.ansatz_type = ansatz_type
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.covariance_matrix = covariance_matrix

        self.feature_map = get_feature_map(
            feature_map_type,
            n_qubits=n_qubits,
            covariance_matrix=covariance_matrix
        )
        self.device = qml.device(device_name, wires=self.n_qubits)

        # Build QNode circuit
        self._build_circuit()
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: List[float] = []
        self.is_fitted = False

    def _build_circuit(self) -> None:
        """Construct PennyLane parameterized QNode."""
        if self.ansatz_type == "StronglyEntangling":
            weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
        else:
            weight_shape = (self.n_layers, self.n_qubits)

        self.weight_shape = weight_shape

        @qml.qnode(self.device, interface="autograd", diff_method="parameter-shift")
        def vqc_qnode(inputs: np.ndarray, weights: np.ndarray):
            # 1. Quantum State Encoding
            self.feature_map.apply(inputs, wires=range(self.n_qubits))

            # 2. Parameterized Variational Layers
            if self.ansatz_type == "StronglyEntangling":
                qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            else:
                for layer in range(self.n_layers):
                    for i in range(self.n_qubits):
                        qml.RY(weights[layer, i], wires=i)
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    if self.n_qubits > 2:
                        qml.CNOT(wires=[self.n_qubits - 1, 0])

            # 3. Measurement: Pauli-Z expectation value on qubit 0
            return qml.expval(qml.PauliZ(0))

        self._qnode = vqc_qnode

    def _init_weights(self, seed: int = 42) -> np.ndarray:
        """Initialize variational parameters."""
        rng = np.random.default_rng(seed)
        return rng.uniform(0, 2 * np.pi, size=self.weight_shape)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> VariationalQuantumClassifier:
        """Train variational quantum parameters using analytic gradient descent."""
        # Convert binary labels {0, 1} to {-1, +1} for Pauli-Z expectation matching
        y_signed = np.where(y_train == 0, -1.0, 1.0)
        self.weights = self._init_weights()
        self.bias = 0.0
        self.loss_history = []

        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        weights_var = self.weights
        bias_var = self.bias

        def cost_fn(w, b, x_batch, y_batch):
            preds = np.array([self._qnode(x, w) + b for x in x_batch])
            # Margin MSE loss
            return np.mean((preds - y_batch) ** 2)

        batch_size = min(32, len(X_train))
        n_batches = len(X_train) // batch_size

        for epoch in range(self.epochs):
            indices = np.random.permutation(len(X_train))
            epoch_losses = []

            for b in range(max(1, n_batches)):
                batch_idx = indices[b * batch_size : (b + 1) * batch_size]
                x_b, y_b = X_train[batch_idx], y_signed[batch_idx]

                (weights_var, bias_var), loss = opt.step_and_cost(
                    lambda w, b: cost_fn(w, b, x_b, y_b), weights_var, bias_var
                )
                epoch_losses.append(loss)

            mean_loss = float(np.mean(epoch_losses))
            self.loss_history.append(mean_loss)

        self.weights = weights_var
        self.bias = float(bias_var)
        self.is_fitted = True
        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Compute continuous quantum expectation values."""
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Model must be fitted before predicting.")
        return np.array([self._qnode(x, self.weights) + self.bias for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute sigmoid-calibrated diagnostic probabilities."""
        raw = self.predict_raw(X)
        # Sigmoid mapping from expectation [-1, 1] to [0, 1]
        prob_pos = 1.0 / (1.0 + np.exp(-2.0 * raw))
        prob_neg = 1.0 - prob_pos
        return np.vstack([prob_neg, prob_pos]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary classification labels {0, 1}."""
        raw = self.predict_raw(X)
        return np.where(raw >= 0.0, 1, 0)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Calculate classification accuracy."""
        preds = self.predict(X_test)
        return float(np.mean(preds == y_test))

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> VariationalQuantumClassifier:
        """Load model from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


class TorchVQC(nn.Module):
    """PyTorch Module encapsulating a PennyLane Variational Quantum Layer."""

    def __init__(self, n_qubits: int = 6, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.device, interface="torch")
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.head = nn.Linear(n_qubits, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = self.qlayer(x)
        return self.head(q_out)
