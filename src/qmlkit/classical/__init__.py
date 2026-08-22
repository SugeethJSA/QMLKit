"""Classical machine learning and deep learning baselines for comparative evaluation."""

from qmlkit.classical.baselines import (
    ClassicalBaselineSuite,
    Temporal1DCNN,
    get_all_classical_baselines,
)

__all__ = [
    "ClassicalBaselineSuite",
    "Temporal1DCNN",
    "get_all_classical_baselines",
]
