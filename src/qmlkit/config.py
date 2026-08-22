"""Configuration models and global seed management for QMLKit."""

from __future__ import annotations

import os
import random
from typing import List, Literal, Optional
import numpy as np
from pydantic import BaseModel, Field
import torch


def set_seed(seed: int = 42) -> None:
    """Set seeds globally across Python, NumPy, PyTorch, and OS environment for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class SensorArrayConfig(BaseModel):
    """Configuration for Biomimetic Olfactory Sensor Array."""
    n_sensors: int = Field(default=16, ge=4, le=64, description="Number of cross-reactive sensor channels")
    sampling_freq_hz: float = Field(default=10.0, description="Sampling rate in Hz")
    transient_steps: int = Field(default=100, description="Time steps recorded per sensor during exposure")
    baseline_noise_sigma: float = Field(default=0.03, description="Sensor Gaussian white noise standard deviation")
    drift_amplitude: float = Field(default=0.02, description="Baseline low-frequency drift amplitude")
    adsorption_rate: float = Field(default=0.15, description="Mean Langmuir adsorption rate constant")
    desorption_rate: float = Field(default=0.08, description="Mean Langmuir desorption rate constant")


class VOCBiomarkerConfig(BaseModel):
    """Configuration for Chemical Volatile Organic Compounds (VOCs)."""
    compounds: List[str] = Field(
        default=[
            # Aldehydes (Lipid peroxidation)
            "Hexanal", "Heptanal", "Octanal", "Nonanal", "Benzaldehyde", "Decanal",
            # Ketones (Mitochondrial / fatty acid dysregulation)
            "Acetone", "2-Butanone", "2-Pentanone", "3-Octanone", "Acetophenone", "Cyclohexanone",
            # Aromatic Hydrocarbons (Cytochrome P450 alteration)
            "Ethylbenzene", "Styrene", "Toluene", "o-Xylene", "1,2,4-Trimethylbenzene", "Benzene",
            # Alkanes / Terpenes / Sulfur Metabolites
            "Isoprene", "Octane", "Decane", "D-Limonene", "Dimethyl_disulfide", "Dimethyl_sulfide"
        ]
    )
    cancer_types: List[str] = Field(
        default=["Healthy", "Lung_Cancer", "Breast_Cancer", "Colorectal_Cancer", "Prostate_Cancer", "Ovarian_Cancer"]
    )


class QuantumModelConfig(BaseModel):
    """Configuration for Quantum Machine Learning Circuits."""
    n_qubits: int = Field(default=6, ge=2, le=24, description="Number of active qubits in quantum register")
    feature_map_type: Literal["BioZZ", "ZZ", "Angle", "Covariance"] = Field(
        default="BioZZ", description="Quantum feature state embedding type"
    )
    ansatz_type: Literal["StronglyEntangling", "RealAmplitudes", "Basic"] = Field(
        default="StronglyEntangling", description="VQC variational ansatz circuit"
    )
    ansatz_layers: int = Field(default=2, ge=1, le=8, description="Number of parameterized repeating layers")
    device_name: str = Field(default="default.qubit", description="PennyLane device backend")
    shots: Optional[int] = Field(default=None, description="Number of measurement shots (None for statevector)")
    qsvm_c: float = Field(default=1.0, description="QSVM regularization C parameter")
    vqc_learning_rate: float = Field(default=0.02, description="VQC Adam optimizer learning rate")
    vqc_epochs: int = Field(default=35, description="VQC training epochs")


class BenchmarkConfig(BaseModel):
    """Configuration for leak-free cross-validation and benchmarking."""
    test_size: float = Field(default=0.20, ge=0.05, le=0.5, description="Held-out test set ratio")
    n_splits_cv: int = Field(default=5, ge=2, le=10, description="K-Fold cross-validation splits")
    random_state: int = Field(default=42, description="Master random seed for reproducibility")
    target_indication: str = Field(default="Lung_Cancer", description="Primary binary cancer target vs Healthy")
    run_all_indications: bool = Field(default=False, description="Run benchmarks across all cancer types")


class QMLKitAppConfig(BaseModel):
    """Unified master application configuration."""
    sensors: SensorArrayConfig = Field(default_factory=SensorArrayConfig)
    voc: VOCBiomarkerConfig = Field(default_factory=VOCBiomarkerConfig)
    quantum: QuantumModelConfig = Field(default_factory=QuantumModelConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    output_dir: str = Field(default="outputs", description="Directory to store benchmark reports, plots, and models")
