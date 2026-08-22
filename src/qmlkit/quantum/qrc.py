"""Quantum Reservoir Computing (QRC) for Temporal Olfactory Kinetics."""

from __future__ import annotations

import pickle

import numpy as np
import pennylane as qml
from sklearn.linear_model import RidgeClassifier


class QuantumReservoirClassifier:
    """Quantum Reservoir Computer utilizing transverse-field Ising dynamics to process time-series kinetics."""

    def __init__(
        self,
        n_qubits: int = 5,
        coupling_strength: float = 0.5,
        alpha_ridge: float = 1.0,
        random_state: int = 42
    ):
        self.n_qubits = n_qubits
        self.coupling_strength = coupling_strength
        self.alpha_ridge = alpha_ridge
        self.random_state = random_state

        self.rng = np.random.default_rng(random_state)
        # Random fixed Hamiltonian couplings J_ij and longitudinal fields h_i
        self.J = self.rng.uniform(-1.0, 1.0, size=(n_qubits, n_qubits))
        self.J = (self.J + self.J.T) * self.coupling_strength
        self.h = self.rng.uniform(-1.0, 1.0, size=n_qubits)

        self.device = qml.device("default.qubit", wires=self.n_qubits)
        self._build_reservoir_circuit()
        self.readout = RidgeClassifier(alpha=self.alpha_ridge)
        self.is_fitted = False

    def _build_reservoir_circuit(self) -> None:
        """Construct fixed dynamical reservoir circuit."""
        @qml.qnode(self.device)
        def reservoir_step(inputs: np.ndarray, prev_angles: np.ndarray):
            # 1. State recurrence
            for i in range(self.n_qubits):
                qml.RZ(prev_angles[i], wires=i)

            # 2. Input injection
            for i in range(self.n_qubits):
                val = inputs[i % len(inputs)]
                qml.RX(val, wires=i)

            # 3. Fixed Entangling Evolution (Ising interaction)
            for i in range(self.n_qubits):
                qml.RZ(self.h[i], wires=i)
                for j in range(i + 1, self.n_qubits):
                    qml.IsingXX(self.J[i, j], wires=[i, j])

            # 4. Measure multi-qubit Pauli-Z expectations
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._reservoir_step = reservoir_step

    def transform_series(self, time_series_tensor: np.ndarray) -> np.ndarray:
        """Process time-series tensor (N_samples, N_sensors, N_timesteps) through quantum reservoir.

        Returns condensed high-dimensional quantum dynamical state matrix (N_samples, N_features).
        """
        n_samples, n_sensors, timesteps = time_series_tensor.shape
        reservoir_features = np.zeros((n_samples, self.n_qubits * 3))

        for idx in range(n_samples):
            prev_angles = np.zeros(self.n_qubits)
            trajectory = []

            # Step through sub-sampled time steps for efficiency
            sample_step = max(1, timesteps // 15)
            for t in range(0, timesteps, sample_step):
                current_input = time_series_tensor[idx, :, t]
                expvals = np.array(self._reservoir_step(current_input, prev_angles))
                prev_angles = expvals * np.pi
                trajectory.append(expvals)

            traj_mat = np.array(trajectory)  # (T_steps, n_qubits)
            # Reservoir Summary Features: Mean, Max, and Final dynamical state
            mean_state = np.mean(traj_mat, axis=0)
            max_state = np.max(traj_mat, axis=0)
            final_state = traj_mat[-1]

            reservoir_features[idx] = np.concatenate([mean_state, max_state, final_state])

        return reservoir_features

    def fit(self, time_series_tensor: np.ndarray, y_train: np.ndarray) -> QuantumReservoirClassifier:
        """Fit classical readout on quantum reservoir representations."""
        X_res = self.transform_series(time_series_tensor)
        self.readout.fit(X_res, y_train)
        self.is_fitted = True
        return self

    def predict(self, time_series_tensor: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        X_res = self.transform_series(time_series_tensor)
        return self.readout.predict(X_res)

    def score(self, time_series_tensor: np.ndarray, y_test: np.ndarray) -> float:
        preds = self.predict(time_series_tensor)
        return float(np.mean(preds == y_test))

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> QuantumReservoirClassifier:
        with open(filepath, "rb") as f:
            return pickle.load(f)
