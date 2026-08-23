"""Hybrid training lab: composable quantum+classical pipeline experiments."""

from qmlkit.lab.cv import FoldResult, cross_validate_config, summarise_folds
from qmlkit.lab.experiments import (
    default_presets,
    ensemble_base_specs,
    make_pipeline,
    run_feature_map_ablation,
    run_hybrid_search,
    run_modality_ablation,
    run_robustness,
)
from qmlkit.lab.pipeline import HybridPipeline, PipelineSpec
from qmlkit.lab.registry import RunRegistry
from qmlkit.lab.stacking import SoftVotingEnsemble, StackingEnsemble

__all__ = [
    "HybridPipeline",
    "PipelineSpec",
    "FoldResult",
    "cross_validate_config",
    "summarise_folds",
    "default_presets",
    "ensemble_base_specs",
    "make_pipeline",
    "run_hybrid_search",
    "run_feature_map_ablation",
    "run_modality_ablation",
    "run_robustness",
    "RunRegistry",
    "SoftVotingEnsemble",
    "StackingEnsemble",
]
