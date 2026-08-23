"""Tests for the hybrid training lab (pipeline, stacking, CV, experiments)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qmlkit.lab import (
    HybridPipeline,
    PipelineSpec,
    SoftVotingEnsemble,
    StackingEnsemble,
    cross_validate_config,
    default_presets,
    make_pipeline,
    run_feature_map_ablation,
    run_hybrid_search,
    run_modality_ablation,
    run_robustness,
    summarise_folds,
)
from qmlkit.lab.registry import RunRegistry


@pytest.fixture(scope="module")
def toy_data():
    rng = np.random.default_rng(7)
    n = 90
    X = pd.DataFrame(rng.normal(size=(n, 10)), columns=[f"f{i}" for i in range(10)])
    signal = X["f0"] + 0.6 * X["f1"] - 0.4 * X["f2"]
    y = (signal + rng.normal(0, 0.4, n) > 0.2).astype(int).values
    return X, y


def _fast_spec(name="t", **kw):
    defaults = dict(
        name=name, reduction="pca", embedding="cwzz", head="qsvm",
        n_components=3, vqc_epochs=1, seed=42,
    )
    defaults.update(kw)
    return PipelineSpec(**defaults)


class TestHybridPipeline:
    def test_qsvm_pipeline_fit_predict(self, toy_data):
        X, y = toy_data
        pipe = HybridPipeline(_fast_spec()).fit(X.iloc[:60], y[:60])
        proba = pipe.predict_proba(X.iloc[60:])
        assert proba.shape == (30, 2)
        assert set(np.unique(np.argmax(proba, axis=1))) <= {0, 1}

    def test_permuted_covariance_differs_from_train(self, toy_data):
        X, y = toy_data
        p_train = HybridPipeline(_fast_spec(embedding="cwzz")).fit(X.iloc[:60], y[:60])
        p_perm = HybridPipeline(_fast_spec(embedding="cwzz_permuted")).fit(X.iloc[:60], y[:60])
        assert not np.allclose(p_train.cov_matrix, p_perm.cov_matrix)
        diag_ok = np.allclose(np.diag(p_perm.cov_matrix), 1.0)
        sym_ok = np.allclose(p_perm.cov_matrix, p_perm.cov_matrix.T)
        assert diag_ok and sym_ok

    def test_quantum_augmented_head(self, toy_data):
        X, y = toy_data
        spec = _fast_spec("aug", head="quantum_augmented_xgb")
        pipe = HybridPipeline(spec).fit(X.iloc[:50], y[:50])
        assert pipe.predict_proba(X.iloc[50:]).shape == (40, 2)

    def test_nan_imputation_train_median(self, toy_data):
        X, y = toy_data
        X_tr = X.iloc[:60].copy()
        X_te = X.iloc[60:].copy()
        X_tr.loc[3, "f0"] = np.nan
        X_te.loc[61, "f1"] = np.nan
        pipe = HybridPipeline(_fast_spec()).fit(X_tr, y[:60])
        assert np.isfinite(pipe.predict_proba(X_te)).all()

    def test_classical_head_and_describe(self, toy_data):
        X, y = toy_data
        spec = _fast_spec("xgbctl", embedding="none", head="random_forest")
        pipe = HybridPipeline(spec).fit(X.iloc[:60], y[:60])
        assert pipe.predict_proba(X.iloc[60:]).shape == (30, 2)
        record = pipe.describe()
        assert "feature_map" in record and "spec" in record


class TestCVHarness:
    def test_cross_validate_config_produces_folds(self, toy_data):
        X, y = toy_data
        folds = cross_validate_config(_fast_spec(), X, y, make_pipeline_fn=make_pipeline, n_splits=3, seed=42)
        assert len(folds) == 3
        summary = summarise_folds("t", folds)
        assert 0.0 <= summary["roc_auc_mean"] <= 1.0
        assert summary["n_folds"] == 3


class TestExperiments:
    def test_feature_map_ablation_smoke(self, tmp_path, toy_data):
        X, y = toy_data
        result = run_feature_map_ablation(
            X, y, n_components=3, n_splits=2, vqc_epochs=1,
            output_root=tmp_path / "lab",
        )
        names = [row["config"] for row in result["leaderboard"]]
        assert "CWZZ-permuted-control" in names and len(names) == 4

    def test_modality_ablation_groups(self, tmp_path, toy_data):
        X, y = toy_data
        groups = {"g1": ["f0", "f1", "f2"], "g2": [f"f{i}" for i in range(3, 10)]}
        result = run_modality_ablation(
            X, y, groups, base_spec=_fast_spec(), n_splits=2,
            output_root=tmp_path / "lab",
        )
        # full + 2 only + 2 remove-one = 5 configs
        assert result["run_dir"] and len(result["leaderboard"]) == 5

    def test_robustness_noise_levels(self, tmp_path, toy_data):
        X, y = toy_data
        result = run_robustness(
            X, y, spec=_fast_spec(), noise_levels=(0.0, 0.2),
            n_splits=2, output_root=tmp_path / "lab",
        )
        assert len(result["levels"]) == 2

    def test_run_hybrid_search_registry_files(self, tmp_path, toy_data):
        X, y = toy_data
        presets = [_fast_spec("a"), _fast_spec("b", embedding="none", head="random_forest")]
        result = run_hybrid_search(
            X, y, presets=presets, include_ensembles=False,
            n_splits=2, seed=42, output_root=tmp_path / "lab",
            progress_cb=lambda _m: None,
        )
        run_dir = Path(result["run_dir"])
        assert (run_dir / "leaderboard.csv").exists()
        assert (run_dir / "run.json").exists()
        assert len(result["leaderboard"]) == 2


class TestRegistry:
    def test_leaderboard_sorted_desc(self, tmp_path):
        reg = RunRegistry(tmp_path)
        reg.add_result({"config": "low", "roc_auc_mean": 0.6}, {})
        reg.add_result({"config": "high", "roc_auc_mean": 0.9}, {})
        run_dir = reg.finalise()
        lines = (run_dir / "leaderboard.csv").read_text(encoding="utf-8").strip().splitlines()
        assert "high" in lines[1] and "low" in lines[-1]


class TestKennelSynth:
    def test_generate_trials_shapes_and_labels(self):
        from qmlkit.lab.kennel_synth import generate_synthetic_trials

        ds = generate_synthetic_trials(trials_per_class=5, rate_hz=40, seed=3)
        assert ds.X.shape[0] == 10
        assert sorted(set(ds.y.tolist())) == [0, 1]
        assert list(ds.X.columns) == ds.feature_names
        assert all(len(v) > 0 for v in ds.feature_groups.values())

    @pytest.mark.parametrize("head", ["qsvm"])
    def test_lab_runs_on_synth_trials(self, head, tmp_path):
        from qmlkit.lab.kennel_synth import generate_synthetic_trials

        ds = generate_synthetic_trials(trials_per_class=8, rate_hz=40, seed=11)
        spec = _fast_spec("synth", reduction="pca", embedding="cwzz", head=head, n_components=4)
        folds = cross_validate_config(spec, ds.X, ds.y, make_pipeline_fn=make_pipeline, n_splits=2, seed=42)
        assert summarise_folds("s", folds)["n_folds"] == 2


def test_soft_voting_ensemble(toy_data):
    X, y = toy_data
    members = [
        make_pipeline(_fast_spec("m1")),
        make_pipeline(_fast_spec("m2", embedding="none", head="random_forest")),
    ]
    ens = SoftVotingEnsemble(members).fit(X.iloc[:60], y[:60])
    proba = ens.predict_proba(X.iloc[60:])
    assert proba.shape == (30, 2)


def test_stacking_ensemble_oof(toy_data):
    X, y = toy_data
    ens = StackingEnsemble(
        base_specs=[_fast_spec("b1"), _fast_spec("b2", embedding="none", head="random_forest")],
        make_pipeline_fn=make_pipeline,
        n_splits=2,
        seed=42,
    ).fit(X.iloc[:70], y[:70])
    proba = ens.predict_proba(X.iloc[70:])
    assert proba.shape == (20, 2)


def test_default_presets_complete():
    presets = default_presets(seed=1)
    names = {p.name for p in presets}
    assert any("flagship" in n for n in names)
    assert any("permuted" in n for n in names)
    assert all(isinstance(make_pipeline(p), HybridPipeline) for p in presets)

def test_angle_and_zz_use_different_feature_maps():
    angle_pipe = HybridPipeline(_fast_spec(embedding="angle"))
    zz_pipe = HybridPipeline(_fast_spec(embedding="zz"))

    from qmlkit.lab.pipeline import _make_head

    angle_head = _make_head(angle_pipe.spec, None)
    zz_head = _make_head(zz_pipe.spec, None)

    assert angle_head.feature_map_type == "Angle"
    assert zz_head.feature_map_type == "ZZ"
    assert angle_head.feature_map.name == "Angle"
    assert zz_head.feature_map.name == "ZZ"
