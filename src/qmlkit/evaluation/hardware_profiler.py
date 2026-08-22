"""Quantum Hardware Profiler: Circuit Resource Estimation and Shot Noise Analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CircuitProfile:
    """Quantum Circuit Compilation & Hardware Profile."""
    n_qubits: int
    circuit_depth: int
    total_gates: int
    single_qubit_gates: int
    two_qubit_cnot_gates: int
    parameter_count: int
    nisq_compatibility_verdict: str


class QuantumHardwareProfiler:
    """Estimates quantum circuit complexity, gate decomposition, and shot noise sensitivity."""

    @staticmethod
    def profile_feature_map(
        feature_map_type: str = "BioZZ",
        n_qubits: int = 6,
        reps: int = 2
    ) -> CircuitProfile:
        """Profile quantum resource consumption of a feature map circuit."""
        # Calculate theoretical gate counts for ZZ feature map with reps
        single_qubit = (n_qubits + n_qubits) * reps  # Hadamards + RZ rotations
        two_qubit_cnots = (n_qubits * (n_qubits - 1)) * reps  # 2 CNOTs per pair per rep
        rz_two_qubit = (n_qubits * (n_qubits - 1) // 2) * reps
        total_gates = single_qubit + two_qubit_cnots + rz_two_qubit
        depth = 2 * reps + (n_qubits - 1) * 3 * reps

        verdict = "EXCELLENT - Fully Executable on Near-Term NISQ QPUs" if depth <= 60 else "MODERATE - Requires Error Mitigation"

        return CircuitProfile(
            n_qubits=n_qubits,
            circuit_depth=depth,
            total_gates=total_gates,
            single_qubit_gates=single_qubit + rz_two_qubit,
            two_qubit_cnot_gates=two_qubit_cnots,
            parameter_count=0,
            nisq_compatibility_verdict=verdict
        )

    @staticmethod
    def evaluate_shot_noise_robustness(
        kernel_engine,
        X_sample: np.ndarray,
        shot_levels: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """Evaluate quantum kernel stability under finite measurement shot sampling."""
        shots_list = shot_levels or [500, 1000, 4000, 8192]
        # Statevector baseline
        k_exact = kernel_engine.compute_kernel_matrix(X_sample[:5], X_sample[:5])

        robustness_scores = {}
        for shots in shots_list:
            # Simulate sampling noise with standard error sqrt(p(1-p)/shots)
            noise_sigma = np.sqrt(0.25 / shots)
            noise_matrix = np.random.default_rng(42).normal(0, noise_sigma, size=k_exact.shape)
            noise_matrix = (noise_matrix + noise_matrix.T) / 2.0
            np.fill_diagonal(noise_matrix, 0.0)

            k_noisy = np.clip(k_exact + noise_matrix, 0.0, 1.0)
            fidelity_loss = float(np.mean(np.abs(k_exact - k_noisy)))
            robustness_scores[f"shots_{shots}_mae"] = round(fidelity_loss, 4)

        return robustness_scores
