"""CLI: run the hybrid quantum+classical training-lab experiments.

Examples:
  # curated preset search on the real VOC dataset (5-fold CV)
  python scripts/run_hybrid_search.py --dataset voc_real --max-samples 120

  # feature-map ablation (manuscript RQ3) incl. permuted-correlation control
  python scripts/run_hybrid_search.py --dataset voc_real --experiment map_ablation

  # modality ablation + robustness on synthetic kennel trials
  python scripts/run_hybrid_search.py --dataset kennel_synth --experiment modality_ablation
  python scripts/run_hybrid_search.py --dataset kennel_synth --experiment robustness
"""

import argparse
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd

from qmlkit.config import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid quantum+classical model search lab")
    parser.add_argument("--dataset", choices=["voc_real", "kennel_synth"], default="voc_real")
    parser.add_argument("--experiment", choices=["search", "map_ablation", "modality_ablation", "robustness"],
                        default="search")
    parser.add_argument("--max-samples", type=int, default=0, help="Balanced subsample cap (0=all)")
    parser.add_argument("--vqc-epochs", type=int, default=8)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-components", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default="outputs/lab")
    args = parser.parse_args()

    set_seed(args.seed)

    from qmlkit.lab.data_sources import load_lab_dataset

    X, y, groups = load_lab_dataset(args.dataset, max_samples=args.max_samples, seed=args.seed)
    print(f"=== QMLKit Hybrid Training Lab ===")
    print(f"Dataset: {args.dataset} | N={len(y)} | folds={args.n_splits} | experiment={args.experiment}\n")

    def progress(msg: str) -> None:
        print(f"  .. {msg}")

    from qmlkit.lab.experiments import (
        run_feature_map_ablation,
        run_hybrid_search,
        run_modality_ablation,
        run_robustness,
    )
    from qmlkit.lab.pipeline import PipelineSpec

    common = dict(n_splits=args.n_splits, seed=args.seed, output_root=args.output_root)

    if args.experiment == "search":
        from qmlkit.lab.experiments import default_presets

        presets = default_presets(vqc_epochs=args.vqc_epochs, seed=args.seed)
        for p in presets:
            p.n_components = args.n_components
        result = run_hybrid_search(X, y, presets=presets, **common, progress_cb=progress)
    elif args.experiment == "map_ablation":
        result = run_feature_map_ablation(
            X, y, n_components=args.n_components, vqc_epochs=args.vqc_epochs, **common
        )
    elif args.experiment == "modality_ablation":
        base = PipelineSpec(
            name="CWZZ-QSVM-modality", reduction="pca", embedding="cwzz", head="qsvm",
            n_components=args.n_components, vqc_epochs=args.vqc_epochs, seed=args.seed,
        )
        result = run_modality_ablation(X, y, groups, base_spec=base, **common)
    else:
        spec = PipelineSpec(
            name="CWZZ-QSVM-robust", reduction="pca", embedding="cwzz", head="qsvm",
            n_components=args.n_components, vqc_epochs=args.vqc_epochs, seed=args.seed,
        )
        result = run_robustness(X, y, spec=spec, **common)

    run_dir = Path(result["run_dir"])
    if "leaderboard" in result:
        leaderboard = pd.DataFrame(result["leaderboard"])
        print("\n=== LEADERBOARD (by mean ROC-AUC) ===")
        cols = [c for c in ("config", "roc_auc_mean", "roc_auc_std", "accuracy_mean",
                            "sensitivity_recall_mean", "specificity_mean", "train_time_s_mean")
                if c in leaderboard.columns]
        print(leaderboard[cols].to_markdown(index=False))
    else:  # robustness experiment returns per-level rows
        leaderboard = pd.DataFrame(result.get("levels", []))
        print("\n=== ROBUSTNESS LEVELS ===")
        cols = [c for c in ("config", "roc_auc_mean", "roc_auc_std",
                            "sensitivity_recall_mean", "specificity_mean")
                if c in leaderboard.columns]
        if not leaderboard.empty:
            print(leaderboard[cols].to_markdown(index=False))
        else:
            print("(no level summaries returned)")
    print(f"\n[OK] Artifacts in {run_dir}")
    summary_path = run_dir / "leaderboard.csv"
    print(f"[OK] Leaderboard: {summary_path}")
    with open(run_dir / "_cli_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"argv": sys.argv[1:], "dataset": args.dataset}, fh, indent=2)


if __name__ == "__main__":
    main()
