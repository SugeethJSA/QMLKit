"""Quantum Convolutional Neural Network (QCNN) for Hierarchical Sensor Feature Extraction."""

from __future__ import annotations

import pickle
from typing import List, Optional
import numpy as np
import pennylane as qml

from qmlkit.quantum.feature_maps import get_feature_map


class QuantumConvolutionalClassifier:
    """QCNN classifier applying quantum convolution and quantum pooling unitary blocks."""

    def __init__(
        self,
        n_qubits: int = 8,
        feature_map_type: str = "BioZZ",
        learning_rate: float = 0.03,
        epochs: int = 30,
        covariance_matrix: Optional[np.ndarray] = None,
        device_name: str = "default.qubit"
    ):
        if n_qubits < 2:
            raise ValueError("QCNN requires at least 2 qubits for convolution blocks.")
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.covariance_matrix = covariance_matrix

        self.feature_map = get_feature_map(
            feature_map_type,
            n_qubits=n_qubits,
            covariance_matrix=covariance_matrix
        )
        self.device = qml.device(device_name, wires=self.n_qubits)

        self._build_circuit()
        self.weights: Optional[np.ndarray] = None
        self.loss_history: List[float] = []
        self.is_fitted = False

    def _conv_block(self, params: np.ndarray, wires: List[int]) -> None:
        """2-qubit parameterized quantum convolution block."""
        qml.RY(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])
        qml.RZ(params[2], wires=wires[1])
        qml.RY(params[3], wires=wires[0])
        qml.RY(params[4], wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])

    def _pool_block(self, params: np.ndarray, source_wire: int, sink_wire: int) -> None:
        """Quantum pooling block reducing active qubit degrees of freedom."""
        qml.CRZ(params[0], wires=[source_wire, sink_wire])
        qml.PauliX(wires=source_wire)
        qml.CRX(params[1], wires=[source_wire, sink_wire])

    def _build_circuit(self) -> None:
        """Construct hierarchical QCNN circuit."""
        # Total parameters: 5 for Conv1 + 2 for Pool1 + 5 for Conv2 + 2 for Pool2
        self.n_params = 14

        @qml.qnode(self.device, interface="autograd", diff_method="parameter-shift")
        def qcnn_qnode(inputs: np.ndarray, weights: np.ndarray):
            # 1. State Encoding
            self.feature_map.apply(inputs, wires=range(self.n_qubits))

            # 2. Convolution Layer 1 (on adjacent qubit pairs)
            for i in range(0, self.n_qubits, 2):
                self._conv_block(weights[0:5], [i, (i + 1) % self.n_qubits])
            for i in range(1, self.n_qubits - 1, 2):
                self._conv_block(weights[0:5], [i, (i + 1) % self.n_qubits])

            # 3. Pooling Layer 1 (halves active qubits: reduces 8 -> 4, or 4 -> 2)
            active_qubits_layer1 = []
            for i in range(0, self.n_qubits, 2):
                self._pool_block(weights[5:7], source_wire=i + 1, sink_wire=i)
                active_qubits_layer1.append(i)

            # 4. Convolution Layer 2
            if len(active_qubits_layer1) >= 2:
                for idx in range(len(active_qubits_layer1) - 1):
                    w1, w2 = active_qubits_layer1[idx], active_qubits_layer1[idx + 1]
                    self._conv_block(weights[7:12], [w1, w2])

                # 5. Pooling Layer 2
                self._pool_block(weights[12:14], source_wire=active_qubits_layer1[1], sink_wire=active_qubits_layer1[0])
                final_sink = active_qubits_layer1[0]
            else:
                final_sink = 0

            # 6. Readout
            return qml.expval(qml.PauliZ(final_sink))

        self._qnode = qcnn_qnode

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> QuantumConvolutionalClassifier:
        """Train QCNN parameters."""
        y_signed = np.where(y_train == 0, -1.0, 1.0)
        rng = np.random.default_rng(42)
        self.weights = rng.uniform(0, 2 * np.pi, size=self.n_params)
        self.loss_history = []

        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        weights_var = self.weights

        def cost_fn(w, x_batch, y_batch):
            preds = np.array([self._qnode(x, w) for x in x_batch])
            return np.mean((preds - y_batch) ** 2)

        batch_size = min(32, len(X_train))
        n_batches = len(X_train) // batch_size

        for _ in range(self.epochs):
            indices = np.random.permutation(len(X_train))
            epoch_losses = []

            for b in range(max(1, n_batches)):
                batch_idx = indices[b * batch_size : (b + 1) * batch_size]
                x_b, y_b = X_train[batch_idx], y_signed[batch_idx]

                weights_var, loss = opt.step_and_cost(lambda w: cost_fn(w, x_b, y_b), weights_var)
                epoch_losses.append(loss)

            self.loss_history.append(float(np.mean(epoch_losses)))

        self.weights = weights_var
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute predicted cancer probabilities."""
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Model must be fitted before predicting.")
        raw = np.array([self._qnode(x, self.weights) for x in X])
        prob_pos = 1.0 / (1.0 + np.exp(-2.0 * raw))
        return np.vstack([1.0 - prob_pos, prob_pos]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels {0, 1}."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        return float(np.mean(self.predict(X_test) == y_test))

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> QuantumConvolutionalClassifier:
        with open(filepath, "rb") as f:
            return pickle.load(f)
