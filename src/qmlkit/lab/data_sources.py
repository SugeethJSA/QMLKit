"""Dataset loading for the hybrid training lab (shared by CLI and API)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from qmlkit.data.dataset_loader import (
    LUNG_VOC_MARKDOWN,
    balanced_subsample,
    load_lung_voc_dataset,
)


def load_lab_dataset(
    name: str,
    max_samples: int = 0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, List[str]]]:
    """Return (X, y, feature_groups) for a named lab dataset.

    Datasets:
      - ``voc_real``   : bundled Lung Cancer VOC dataset (cancer_vs_control).
        Compound columns are split into three groups so the modality-ablation
        runner can also probe chemical-subset effects.
      - ``kennel_synth``: simulated three-phase kennel trials with
        cancer-associated vs control micro-movement signatures.
    """
    if name == "voc_real":
        loaded = load_lung_voc_dataset(LUNG_VOC_MARKDOWN, task="cancer_vs_control")
        X, y = loaded.df_features, loaded.y
        cols = list(X.columns)
        third = max(1, len(cols) // 3)
        groups = {
            "compounds_early": cols[:third],
            "compounds_mid": cols[third : 2 * third],
            "compounds_late": cols[2 * third :],
        }
    elif name == "kennel_synth":
        from qmlkit.lab.kennel_synth import generate_synthetic_trials

        ds = generate_synthetic_trials(seed=seed)
        X, y, groups = ds.X, ds.y, ds.feature_groups
    else:
        raise ValueError(f"Unknown lab dataset: {name}")

    if max_samples and max_samples < len(y):
        X, y = balanced_subsample(X, y, max_samples=max_samples, seed=seed)
    return X.reset_index(drop=True), np.asarray(y), groups
