"""Train the kennel micro-movement classifier from recorded Data-Lab sessions.

Pipeline (mirrors GAIT methodology):
  data/kennel/<label>/<dog>_<label><trial>.csv
    -> windowed feature extraction via the SAME extractor used at serving time
    -> leave-one-dog-out cross-validation (honest generalization estimate)
    -> RandomForest artifact bundle at models/kennel_model.joblib

Usage:
  python scripts/train_kennel_model.py [--data-dir data/kennel] [--out models]
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from qmlkit.hardware.kennel_features import (
    KENNEL_FEATURE_NAMES,
    extract_window_features,
    frames_from_dicts,
)
from qmlkit.hardware.session_recording import RecordingManager

WINDOW = 400   # ~4 s @ 100 Hz
STEP = 200     # 50 % overlap


def encode_labels(raw_labels: list[str]) -> tuple[np.ndarray, dict]:
    levels = sorted(set(raw_labels))
    mapping = {level: i for i, level in enumerate(levels)}
    return np.array([mapping[label] for label in raw_labels], dtype=int), {
        str(i): level for level, i in mapping.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train kennel micro-movement classifier")
    parser.add_argument("--data-dir", default="data/kennel")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"No data directory: {data_dir}")

    manager = RecordingManager(data_dir)
    X_rows: list[np.ndarray] = []
    y_raw: list[str] = []
    dogs: list[str] = []

    pattern = re.compile(r"^(?P<dog>.+)_(?P<label>.+?)(?P<trial>\d+)\.csv$")
    csv_paths = sorted(data_dir.glob("*/*.csv"))
    if not csv_paths:
        raise SystemExit("No session recordings found (data/kennel/<label>/*.csv).")

    for path in csv_paths:
        match = pattern.match(path.name)
        dog = match.group("dog") if match else path.stem
        label = path.parent.name
        rows = manager.load_csv_rows(path)
        frames = frames_from_dicts(rows)
        for start in range(0, max(1, len(frames) - WINDOW + 1), STEP):
            chunk = frames[start : start + WINDOW]
            if len(chunk) < WINDOW:
                break
            feats = extract_window_features(chunk)
            if not np.isnan(feats).all():
                X_rows.append(feats)
                y_raw.append(label)
                dogs.append(dog)

    y, class_names = encode_labels(y_raw)
    X = np.vstack(X_rows)
    # Impute NaN features with column medians (robust to missing ultrasonic echo).
    col_median = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), col_median, X)

    print(f"Windows: {X.shape[0]} | Features: {X.shape[1]} | Classes: {class_names}")
    print(f"Dogs: {sorted(set(dogs))}")

    # Leave-one-dog-out validation.
    unique_dogs = sorted(set(dogs))
    fold_acc, fold_f1 = [], []
    for held_out in unique_dogs:
        test_idx = np.array([i for i, d in enumerate(dogs) if d == held_out])
        train_idx = np.array([i for i, d in enumerate(dogs) if d != held_out])
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        fold_acc.append(float(accuracy_score(y[test_idx], pred)))
        fold_f1.append(float(f1_score(y[test_idx], pred, average="macro", zero_division=0)))

    report = {
        "windows": int(X.shape[0]),
        "features": KENNEL_FEATURE_NAMES[:5] + ["..."] + KENNEL_FEATURE_NAMES[-3:],
        "n_features": len(KENNEL_FEATURE_NAMES),
        "class_names": class_names,
        "class_distribution": dict(Counter(y_raw)),
        "dogs": unique_dogs,
        "loso_accuracy_mean": float(np.mean(fold_acc)) if fold_acc else None,
        "loso_macro_f1_mean": float(np.mean(fold_f1)) if fold_f1 else None,
        "folds": len(fold_acc),
        "window": WINDOW,
        "step": STEP,
    }

    final_model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42, n_jobs=-1)
    final_model.fit(X, y)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "kennel_model.joblib"
    joblib.dump(
        {
            "model": final_model,
            "feature_names": KENNEL_FEATURE_NAMES,
            "classes": class_names,
            "metrics": report,
            "manifest": {"version": "0.1.0", "window": WINDOW, "step": STEP},
        },
        artifact_path,
    )
    report_path = out_dir / "kennel_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] Model artifact -> {artifact_path}")
    print(f"[OK] Training report -> {report_path}")
    if report["loso_accuracy_mean"] is not None:
        print(
            f"LOSO validation: accuracy={report['loso_accuracy_mean']:.3f} "
            f"macro-F1={report['loso_macro_f1_mean']:.3f} over {report['folds']} folds"
        )


if __name__ == "__main__":
    main()
