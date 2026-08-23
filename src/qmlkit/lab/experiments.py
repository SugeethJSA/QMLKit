"""Curated hybrid presets and experiment runners (search, ablations, robustness).

Runners implement the manuscript's evaluation protocol:
  - identical stratified partitions shared across configurations (§VII-B/F)
  - fresh pipeline + correlation matrix per fold (leak-free, §VI-C/VII-F)
  - feature-map ablation incl. permuted-correlation control (§VII-D)
  - sensor-modality ablation via feature groups (§VII-D)
  - noise-injection and dropout robustness (§VII-E)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from qmlkit.lab.cv import FoldResult, cross_validate_config, summarise_folds
from qmlkit.lab.pipeline import HybridPipeline, PipelineSpec
from qmlkit.lab.registry import RunRegistry


def make_pipeline(spec: PipelineSpec) -> HybridPipeline:
    """Factory used by CV/stacking so each fold gets an unfitted pipeline."""
    return HybridPipeline(spec)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def default_presets(vqc_epochs: int = 8, seed: int = 42) -> List[PipelineSpec]:
    """Curated 'best of both worlds' combination set (5-fold CV friendly)."""
    base = dict(n_components=6, vqc_epochs=vqc_epochs, seed=seed)
    return [
        PipelineSpec(name="CWZZ-QSVM (flagship)", reduction="pca", embedding="cwzz", head="qsvm", **base),
        PipelineSpec(name="ZZ-QSVM", reduction="pca", embedding="zz", head="qsvm", **base),
        PipelineSpec(name="Angle-QSVM", reduction="pca", embedding="angle", head="qsvm", **base),
        PipelineSpec(
            name="CWZZ-permuted-QSVM (control)", reduction="pca",
            embedding="cwzz_permuted", head="qsvm", **base,
        ),
        PipelineSpec(name="MI-CWZZ-QSVM", reduction="mutual_info", embedding="cwzz", head="qsvm", **base),
        PipelineSpec(name="AE-VQC", reduction="autoencoder", embedding="cwzz", head="vqc", **base),
        PipelineSpec(name="Quantum-Augmented-XGB", reduction="pca", embedding="cwzz",
                     head="quantum_augmented_xgb", **base),
        PipelineSpec(name="QCNN", reduction="pca", embedding="cwzz", head="qcnn", **base),
        PipelineSpec(name="PCA-XGBoost (classical control)", reduction="pca", embedding="none",
                     head="xgboost", **base),
        PipelineSpec(name="Raw-XGBoost (classical control)", reduction="none", embedding="none",
                     head="xgboost", **{**base, "n_components": 32}),
    ]


def ensemble_base_specs(vqc_epochs: int = 8, seed: int = 42) -> List[PipelineSpec]:
    base = dict(n_components=6, vqc_epochs=vqc_epochs, seed=seed)
    return [
        PipelineSpec(name="stack-rf", reduction="none", embedding="none", head="random_forest",
                     **{**base, "n_components": 32}),
        PipelineSpec(name="stack-xgb", reduction="none", embedding="none", head="xgboost",
                     **{**base, "n_components": 32}),
        PipelineSpec(name="stack-qsvm", reduction="pca", embedding="cwzz", head="qsvm", **base),
        PipelineSpec(name="stack-vqc", reduction="pca", embedding="cwzz", head="vqc", **base),
    ]


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _evaluate_config(spec, X, y, n_splits: int, seed: int) -> tuple:
    folds = cross_validate_config(spec, X, y, make_pipeline_fn=make_pipeline, n_splits=n_splits, seed=seed)
    summary = summarise_folds(spec.name, folds)
    details = {
        "spec": spec.to_dict() if hasattr(spec, "to_dict") else str(spec),
        "folds": [vars(f) for f in folds],
    }
    if isinstance(spec, PipelineSpec):
        try:
            details["circuit_profile"] = HybridPipeline(spec).describe()
        except Exception:  # pragma: no cover - profiling is best-effort
            pass
    return summary, details


def run_hybrid_search(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    presets: Optional[List[PipelineSpec]] = None,
    include_ensembles: bool = True,
    n_splits: int = 5,
    seed: int = 42,
    output_root: str = "outputs/lab",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Evaluate all curated combos under identical partitions; returns leaderboard."""
    from qmlkit.lab.stacking import SoftVotingEnsemble, StackingEnsemble

    specs = presets if presets is not None else default_presets(seed=seed)
    y_arr = np.asarray(y)
    registry = RunRegistry(output_root=output_root)

    def log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    for spec in specs:
        log(f"Evaluating {spec.name} ...")
        summary, details = _evaluate_config(spec, X, y_arr, n_splits, seed)
        registry.add_result(summary, details)

    if include_ensembles:
        # Soft voting over fitted pipelines sharing the same fold structure.
        log("Evaluating SoftVoting {SVM-RBF, RF, XGB, QSVM} ...")
        soft_specs = [
            PipelineSpec(name="vote-svm-rbf", reduction="none", embedding="none", head="svm_rbf",
                         n_components=8, vqc_epochs=specs[0].vqc_epochs, seed=seed),
            PipelineSpec(name="vote-rf", reduction="none", embedding="none", head="random_forest",
                         n_components=8, vqc_epochs=specs[0].vqc_epochs, seed=seed),
            PipelineSpec(name="vote-xgb", reduction="none", embedding="none", head="xgboost",
                         n_components=8, vqc_epochs=specs[0].vqc_epochs, seed=seed),
            PipelineSpec(name="vote-qsvm", reduction="pca", embedding="cwzz", head="qsvm",
                         n_components=6, vqc_epochs=specs[0].vqc_epochs, seed=seed),
        ]

        def soft_vote_factory(specs_list):
            def build(_spec):
                members = [make_pipeline(s) for s in specs_list]
                return SoftVotingEnsemble(members)
            return build

        try:
            folds = cross_validate_config(
                PipelineSpec(name="SoftVote-ensemble"), X, y_arr,
                make_pipeline_fn=soft_vote_factory(soft_specs), n_splits=n_splits, seed=seed,
            )
            registry.add_result(summarise_folds("SoftVote-ensemble", folds), {"kind": "soft_voting"})
        except Exception as exc:  # pragma: no cover - keep presets on ensemble failure
            log(f"SoftVote failed: {exc}")

        log("Evaluating Stacked ensemble {RF, XGB, QSVM, VQC} -> LogReg ...")
        stack_spec = PipelineSpec(name="Stacked-LR-ensemble")
        try:
            folds = cross_validate_config(
                stack_spec, X, y_arr,
                make_pipeline_fn=lambda _spec: StackingEnsemble(
                    ensemble_base_specs(vqc_epochs=specs[0].vqc_epochs, seed=seed),
                    make_pipeline_fn=make_pipeline, n_splits=n_splits, seed=seed,
                ),
                n_splits=n_splits, seed=seed,
            )
            registry.add_result(summarise_folds("Stacked-LR-ensemble", folds), {"kind": "stacking"})
        except Exception as exc:  # pragma: no cover - keep presets on ensemble failure
            log(f"Stacking failed: {exc}")

    run_dir = registry.finalise(extra_meta={
        "dataset_shape": list(np.asarray(X).shape),
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(y_arr, return_counts=True), strict=True)},
        "n_splits": n_splits,
        "seed": seed,
        "presets": [s.name for s in specs],
    })
    leaderboard_rows = sorted(
        (r["summary"] for r in registry.records),
        key=lambda s: s.get("roc_auc_mean", 0.0),
        reverse=True,
    )
    return {"run_dir": str(run_dir), "leaderboard": leaderboard_rows}


def run_feature_map_ablation(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    n_components: int = 6,
    n_splits: int = 5,
    vqc_epochs: int = 8,
    seed: int = 42,
    output_root: str = "outputs/lab",
) -> Dict[str, Any]:
    """RQ3: Angle vs ZZ vs CW-ZZ vs permuted-CW-ZZ under identical partitions."""
    variants = [
        ("Angle-QSVM", "angle"),
        ("ZZ-QSVM", "zz"),
        ("CWZZ-QSVM", "cwzz"),
        ("CWZZ-permuted-control", "cwzz_permuted"),
    ]
    registry = RunRegistry(output_root=output_root)
    for name, emb in variants:
        spec = PipelineSpec(
            name=name, reduction="pca", n_components=n_components,
            embedding=emb, head="qsvm", vqc_epochs=vqc_epochs, seed=seed,
        )
        summary, details = _evaluate_config(spec, X, np.asarray(y), n_splits, seed)
        registry.add_result(summary, details)
    run_dir = registry.finalise(extra_meta={"experiment": "feature_map_ablation"})
    return {"run_dir": str(run_dir), "leaderboard": sorted(
        (r["summary"] for r in registry.records), key=lambda s: s.get("roc_auc_mean", 0.0), reverse=True)}


def run_modality_ablation(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_groups: Dict[str, Sequence[str]],
    base_spec: Optional[PipelineSpec] = None,
    n_splits: int = 5,
    seed: int = 42,
    output_root: str = "outputs/lab",
) -> Dict[str, Any]:
    """RQ2: full vs single-modality-only vs remove-one-modality runs.

    ``feature_groups`` maps group name -> column names. The base spec defaults to
    the flagship CWZZ-QSVM with a qubit count matching the smallest group size.
    """
    groups = {g: list(cols) for g, cols in feature_groups.items()}
    all_cols = [c for cols in groups.values() for c in cols]
    base_spec = base_spec or PipelineSpec(
        name="CWZZ-QSVM-modality", reduction="pca", embedding="cwzz", head="qsvm", seed=seed,
    )

    configs: List[tuple[str, pd.DataFrame]] = [("full_multimodal", X[all_cols])]
    for g, cols in groups.items():
        configs.append((f"only_{g}", X[cols]))
    for g in groups:
        kept = [c for other_g, cols in groups.items() if other_g != g for c in cols]
        configs.append((f"remove_{g}", X[kept]))

    registry = RunRegistry(output_root=output_root)
    for name, X_sub in configs:
        spec = dataclasses_replace(base_spec, name=name)
        summary, details = _evaluate_config(spec, X_sub, np.asarray(y), n_splits, seed)
        registry.add_result(summary, details)
    run_dir = registry.finalise(extra_meta={"experiment": "modality_ablation", "groups": {
        g: len(c) for g, c in groups.items()}})
    return {"run_dir": str(run_dir), "leaderboard": sorted(
        (r["summary"] for r in registry.records), key=lambda s: s.get("roc_auc_mean", 0.0), reverse=True)}


def run_robustness(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    spec: Optional[PipelineSpec] = None,
    noise_levels: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3),
    dropout_fraction: float = 0.0,
    n_splits: int = 5,
    seed: int = 42,
    output_root: str = "outputs/lab",
) -> Dict[str, Any]:
    """§VII-E: Gaussian noise on normalised features + optional feature dropout at eval time."""
    spec = spec or PipelineSpec(
        name="CWZZ-QSVM-robust", reduction="pca", embedding="cwzz", head="qsvm", seed=seed,
    )
    y_arr = np.asarray(y)
    rng = np.random.default_rng(seed)
    registry = RunRegistry(output_root=output_root)

    rows: List[Dict[str, Any]] = []
    for level in noise_levels:
        # Train on clean data; evaluate under degradation (paper §VII-E).
        perturbed_rows: List[FoldResult] = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_arr)), y_arr)):
            X_tr, y_tr = _slice(X, train_idx), y_arr[train_idx]
            X_val = _slice(X, val_idx).copy()
            X_val = _add_noise_and_dropout(X_val, level, dropout_fraction, rng)
            pipeline = make_pipeline(spec)
            pipeline.fit(X_tr, y_tr)
            proba = pipeline.predict_proba(X_val)
            preds = np.argmax(proba, axis=1)
            from dataclasses import asdict

            from qmlkit.evaluation.benchmark_suite import compute_clinical_metrics

            metrics = compute_clinical_metrics(
                model_name=f"{spec.name}@noise{level}",
                paradigm="Hybrid", y_true=y_arr[val_idx], y_pred=preds, y_prob=proba,
            )
            perturbed_rows.append(FoldResult(fold_idx, {
                k: v for k, v in asdict(metrics).items() if isinstance(v, (int, float))
            }, 0.0, 0.0))

        summary = summarise_folds(f"{spec.name}@noise={level},dropout={dropout_fraction}", perturbed_rows)
        registry.add_result(summary, {"noise_level": level, "dropout_fraction": dropout_fraction})
        rows.append(summary)

    run_dir = registry.finalise(extra_meta={"experiment": "robustness"})
    return {"run_dir": str(run_dir), "levels": rows}


def _add_noise_and_dropout(X, noise_level: float, dropout_fraction: float, rng) -> pd.DataFrame:
    df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    values = df.values.astype(float)
    stds = values.std(axis=0)
    stds[stds == 0] = 1.0
    noisy = values + rng.normal(0.0, noise_level, size=values.shape) * stds
    if dropout_fraction > 0:
        mask = rng.random(values.shape) < dropout_fraction
        medians = np.nanmedian(values, axis=0)
        noisy[mask] = medians[mask]
    out = df.copy()
    out[:] = noisy
    return out


def dataclasses_replace(spec: PipelineSpec, **changes) -> PipelineSpec:
    import dataclasses

    return dataclasses.replace(spec, **changes)


def _slice(X, idx: np.ndarray):
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx]
    return np.asarray(X)[idx]
