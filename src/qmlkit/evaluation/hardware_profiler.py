"""Quantum Hardware Profiler: Circuit Resource Estimation and Shot Noise Analysis.

Gate counts are derived analytically from the exact circuit structures used by
``qmlkit.quantum.feature_maps`` and the VQC ansaetze, so profiles match what is
actually executed (used for the manuscript's reproducibility reporting).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QuantumHardwareProfiler:
    """Estimates quantum circuit complexity, gate decomposition, and shot noise sensitivity."""

    @staticmethod
    def profile_feature_map(
        feature_map_type: str = "BioZZ",
        n_qubits: int = 6,
        reps: int = 2,
        trainable_params: int = 0,
    ) -> CircuitProfile:
        """Profile resource consumption of a feature-map circuit.

        ZZ-family maps (ZZ / BioZZ / CW-ZZ / Covariance), per repetition:
          - n Hadamards, n single-qubit RZ(2x) encodings
          - one CNOT-RZ-CNOT pair per unordered qubit pair
        Angle encoding: n RY rotations only.
        """
        map_lower = feature_map_type.lower()
        is_zz_family = any(t in map_lower for t in ("zz", "cov", "bio"))

        ry_rotations = n_qubits * reps
        if is_zz_family:
            pairs = n_qubits * (n_qubits - 1) // 2
            hadamards = n_qubits * reps
            rz_single = n_qubits * reps
            cnots = 2 * pairs * reps
            rz_pair = pairs * reps
            # Depth per rep: H layer + RZ layer + (CNOT,RZ,CNOT) serialised per pair.
            depth = reps * (2 + 3 * pairs)
        else:
            hadamards = 0
            rz_single = 0
            cnots = 0
            rz_pair = 0
            depth = ry_rotations

        single_qubit = hadamards + rz_single + rz_pair + ry_rotations
        total_gates = single_qubit + cnots

        verdict = (
            "EXCELLENT - Fully Executable on Near-Term NISQ QPUs"
            if depth <= 60
            else "MODERATE - Requires Error Mitigation"
        )

        return CircuitProfile(
            n_qubits=n_qubits,
            circuit_depth=depth,
            total_gates=total_gates,
            single_qubit_gates=single_qubit,
            two_qubit_cnot_gates=cnots,
            parameter_count=int(trainable_params),
            nisq_compatibility_verdict=verdict,
        )

    @staticmethod
    def profile_variational_ansatz(
        ansatz_type: str = "StronglyEntangling",
        n_qubits: int = 6,
        n_layers: int = 2,
    ) -> Dict[str, Any]:
        """Analytic gate/parameter counts for the VQC trainable block."""
        if ansatz_type == "StronglyEntangling":
            params = n_layers * n_qubits * 3
            rotations = n_layers * n_qubits
            # StronglyEntanglingLayers uses a ring CNOT ladder per layer.
            cnots = n_layers * n_qubits
            depth = n_layers * 3
        else:  # RealAmplitudes-style: RY ring + linear CNOTs
            params = n_layers * n_qubits
            rotations = n_layers * n_qubits
            cnots = n_layers * max(0, n_qubits - 1)
            depth = n_layers * 2
        return {
            "ansatz": ansatz_type,
            "n_layers": n_layers,
            "parameter_count": int(params),
            "rotations": int(rotations),
            "cnot_gates": int(cnots),
            "ansatz_depth": int(depth),
        }

    @staticmethod
    def describe_model(
        feature_map_type: str = "BioZZ",
        n_qubits: int = 6,
        reps: int = 2,
        variational: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Combined reproducibility record (feature map + optional ansatz)."""
        profile = QuantumHardwareProfiler.profile_feature_map(
            feature_map_type=feature_map_type, n_qubits=n_qubits, reps=reps
        )
        record: Dict[str, Any] = {"feature_map": profile.to_dict()}
        if variational:
            record["variational"] = QuantumHardwareProfiler.profile_variational_ansatz(**variational)
        return record

    @staticmethod
    def evaluate_shot_noise_robustness(
        kernel_engine,
        X_sample: np.ndarray,
        shot_levels: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """Evaluate quantum kernel stability under finite measurement shot sampling."""
        shots_list = shot_levels or [500, 1000, 4000, 8192]
        k_exact = kernel_engine.compute_kernel_matrix(X_sample[:5], X_sample[:5])

        robustness_scores = {}
        for shots in shots_list:
            noise_sigma = np.sqrt(0.25 / shots)
            noise_matrix = np.random.default_rng(42).normal(0, noise_sigma, size=k_exact.shape)
            noise_matrix = (noise_matrix + noise_matrix.T) / 2.0
            np.fill_diagonal(noise_matrix, 0.0)

            k_noisy = np.clip(k_exact + noise_matrix, 0.0, 1.0)
            fidelity_loss = float(np.mean(np.abs(k_exact - k_noisy)))
            robustness_scores[f"shots_{shots}_mae"] = round(fidelity_loss, 4)

        return robustness_scores
