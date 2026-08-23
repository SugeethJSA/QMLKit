"""Run registry: persists hybrid-lab experiments to outputs/lab/<run_id>/."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

LEADERBOARD_FIELDS = [
    "config",
    "n_folds",
    "roc_auc_mean",
    "roc_auc_std",
    "accuracy_mean",
    "accuracy_std",
    "balanced_accuracy_mean",
    "sensitivity_recall_mean",
    "specificity_mean",
    "f1_macro_mean",
    "brier_score_mean",
    "train_time_s_mean",
]


class RunRegistry:
    def __init__(self, output_root: str | Path = "outputs/lab"):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_root) / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []
        self.meta: Dict[str, Any] = {"run_id": stamp, "started_at": time.time()}

    def add_result(self, summary: Dict[str, Any], details: Dict[str, Any]) -> None:
        self.records.append({"summary": summary, "details": details})

    def finalise(self, extra_meta: Dict[str, Any] | None = None) -> Path:
        if extra_meta:
            self.meta.update(extra_meta)
        self.meta["finished_at"] = time.time()

        run_json = self.run_dir / "run.json"
        run_json.write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")

        per_config = self.run_dir / "configs"
        per_config.mkdir(exist_ok=True)
        for record in self.records:
            name = record["summary"].get("config", f"config_{len(self.records)}")
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name))
            (per_config / f"{safe}.json").write_text(
                json.dumps(record, indent=2, default=str), encoding="utf-8"
            )

        leaderboard = self.run_dir / "leaderboard.csv"
        rows = sorted(
            [r["summary"] for r in self.records],
            key=lambda s: s.get("roc_auc_mean", 0.0),
            reverse=True,
        )
        with open(leaderboard, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=LEADERBOARD_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        return self.run_dir
