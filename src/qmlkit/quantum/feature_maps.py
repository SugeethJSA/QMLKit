"""Quantum Feature Maps and Hilbert-Space Embedding Circuits."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pennylane as qml


def angle_feature_map(x: np.ndarray, wires: list[int] | range) -> None:
    """First-order Angle Embedding using single-qubit Ry rotations."""
    for i, wire in enumerate(wires):
        qml.RY(x[i], wires=wire)


def pauli_zz_feature_map(x: np.ndarray, wires: list[int] | range, reps: int = 2) -> None:
    """Second-Order Pauli-Z (ZZ) Feature Map.

    Encodes linear phase shifts and non-linear pairwise products between features.
    """
    n_qubits = len(wires)
    for _ in range(reps):
        # Layer 1: Hadamard superposition
        for wire in wires:
            qml.Hadamard(wires=wire)

        # Layer 2: Single-qubit Z phase rotations
        for i, wire in enumerate(wires):
            qml.RZ(2.0 * x[i], wires=wire)

        # Layer 3: Two-qubit ZZ entangling interactions
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                phi_ij = 2.0 * (np.pi - x[i]) * (np.pi - x[j])
                qml.CNOT(wires=[wires[i], wires[j]])
                qml.RZ(phi_ij, wires=wires[j])
                qml.CNOT(wires=[wires[i], wires[j]])


def bio_zz_feature_map(
    x: np.ndarray,
    wires: list[int] | range,
    covariance_matrix: Optional[np.ndarray] = None,
    reps: int = 2
) -> None:
    """Biomimetic Covariance-Weighted Feature Map.

    Weights two-qubit entangling gates by empirical biochemical VOC correlation
    coefficients, directly mapping multi-analyte cross-reactivity into quantum entanglement.
    """
    n_qubits = len(wires)
    cov = covariance_matrix if covariance_matrix is not None else np.eye(n_qubits)

    for _ in range(reps):
        # Hadamard transformation
        for wire in wires:
            qml.Hadamard(wires=wire)

        # Linear encoding
        for i, wire in enumerate(wires):
            qml.RZ(2.0 * x[i], wires=wire)

        # Correlated biomimetic entangling phase
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                weight = float(cov[i, j]) if i < cov.shape[0] and j < cov.shape[1] else 0.5
                phi_ij = 2.0 * weight * (np.pi - x[i]) * (np.pi - x[j])
                qml.CNOT(wires=[wires[i], wires[j]])
                qml.RZ(phi_ij, wires=wires[j])
                qml.CNOT(wires=[wires[i], wires[j]])


class BaseFeatureMap:
    """Base wrapper for feature map circuits."""

    def __init__(self, n_qubits: int, name: str):
        self.n_qubits = n_qubits
        self.name = name

    def apply(self, x: np.ndarray, wires: list[int] | range) -> None:
        raise NotImplementedError


class AngleFeatureMap(BaseFeatureMap):
    def __init__(self, n_qubits: int):
        super().__init__(n_qubits, "Angle")

    def apply(self, x: np.ndarray, wires: list[int] | range) -> None:
        angle_feature_map(x, wires)


class PauliZZFeatureMap(BaseFeatureMap):
    def __init__(self, n_qubits: int, reps: int = 2):
        super().__init__(n_qubits, "ZZ")
        self.reps = reps

    def apply(self, x: np.ndarray, wires: list[int] | range) -> None:
        pauli_zz_feature_map(x, wires, reps=self.reps)


class BioZZFeatureMap(BaseFeatureMap):
    def __init__(self, n_qubits: int, covariance_matrix: Optional[np.ndarray] = None, reps: int = 2):
        super().__init__(n_qubits, "BioZZ")
        self.covariance_matrix = covariance_matrix
        self.reps = reps

    def set_covariance_matrix(self, cov: np.ndarray) -> None:
        self.covariance_matrix = cov

    def apply(self, x: np.ndarray, wires: list[int] | range) -> None:
        bio_zz_feature_map(x, wires, covariance_matrix=self.covariance_matrix, reps=self.reps)


class CovarianceFeatureMap(BioZZFeatureMap):
    """Alias for domain-adapted Covariance-Weighted Feature Map."""
    pass


def get_feature_map(
    map_type: str,
    n_qubits: int,
    covariance_matrix: Optional[np.ndarray] = None
) -> BaseFeatureMap:
    """Factory function to instantiate feature map objects."""
    map_type_lower = map_type.lower()
    if "bio" in map_type_lower or "cov" in map_type_lower:
        return BioZZFeatureMap(n_qubits=n_qubits, covariance_matrix=covariance_matrix)
    elif "zz" in map_type_lower or "pauli" in map_type_lower:
        return PauliZZFeatureMap(n_qubits=n_qubits)
    elif "angle" in map_type_lower:
        return AngleFeatureMap(n_qubits=n_qubits)
    else:
        return BioZZFeatureMap(n_qubits=n_qubits, covariance_matrix=covariance_matrix)
