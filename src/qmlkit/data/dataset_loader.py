"""External dataset ingestion utilities.

Parses pipe-delimited markdown tables (as exported by pandas ``to_markdown``)
and plain CSV files into model-ready feature frames. This is the first
file-ingestion code path in QMLKit; previously the only data source was the
synthetic cohort generator.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Repo-root-relative default locations of bundled real datasets.
REPO_ROOT = Path(__file__).resolve().parents[3]
LUNG_VOC_MARKDOWN = REPO_ROOT / "docs" / "Lung_Cancer_VOC_Dataset_427.md"
DOG_ECG_MARKDOWN = REPO_ROOT / "docs" / "dataset_1.md"
DOG_INFO_MARKDOWN = REPO_ROOT / "docs" / "DogInfo.md"

LUNG_VOC_LABEL_NAMES: Dict[int, str] = {0: "Control", 1: "Cancer"}
# The bundled dataset is 3-class (Control/Benign/Cancer). Named tasks define
# how it is binarised; unmapped classes are dropped.
LUNG_VOC_TASKS: Dict[str, Dict[str, int]] = {
    "cancer_vs_control": {"control": 0, "cancer": 1},
    "disease_vs_control": {"control": 0, "benign": 1, "cancer": 1},
}


@dataclass
class LoadedDataset:
    """Container for an ingested tabular dataset."""

    df_features: pd.DataFrame
    y: np.ndarray
    ids: Optional[pd.Series]
    label_names: Dict[int, str]


def load_markdown_table(path: Union[str, Path]) -> pd.DataFrame:
    """Parse the first pipe-delimited markdown table found in ``path``.

    Handles pandas-style alignment separator rows (``|---:|---|``) and padded
    cells. Raises ``ValueError`` when no well-formed table exists.
    """
    text = Path(path).read_text(encoding="utf-8")
    rows: List[List[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [cell.strip() for cell in line[1:-1].split("|")]
        if all(cell and set(cell) <= set(":- ") for cell in cells):
            continue  # alignment separator row
        rows.append(cells)

    if len(rows) < 2:
        raise ValueError(f"No markdown table found in {path}")

    header = rows[0]
    data = [r for r in rows[1:] if len(r) == len(header)]
    if not data:
        raise ValueError(f"Markdown table in {path} has no data rows")

    return pd.DataFrame(data, columns=header)


def load_lung_voc_dataset(
    path: Union[str, Path] = LUNG_VOC_MARKDOWN,
    task: str = "cancer_vs_control",
    label_map: Optional[Dict[str, int]] = None,
    drop_unmapped: bool = True,
) -> LoadedDataset:
    """Load the real 427-patient Lung Cancer VOC dataset from its markdown dump.

    The source table is 3-class (Control/Benign/Cancer); ``task`` selects the
    binarisation: ``cancer_vs_control`` drops Benign rows, while
    ``disease_vs_control`` pools Benign+Cancer as positives. Returns 27 numeric
    compound concentration features plus binary labels. Rows containing
    unparseable feature values are dropped.
    """
    if label_map is None:
        if task not in LUNG_VOC_TASKS:
            raise ValueError(f"Unknown task '{task}'; expected one of {sorted(LUNG_VOC_TASKS)}")
        mapping = LUNG_VOC_TASKS[task]
    else:
        mapping = {k.lower(): v for k, v in label_map.items()}
    df = load_markdown_table(path)

    id_col = "PatientID"
    label_col = "Class"
    missing = [c for c in (id_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Lung VOC table missing expected column(s): {missing}")

    feature_cols = [c for c in df.columns if c not in (id_col, label_col)]
    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    labels_raw_all = df[label_col].str.strip().str.lower()
    valid = features.notna().all(axis=1)
    if drop_unmapped:
        valid &= labels_raw_all.isin(mapping)
    features = features.loc[valid].reset_index(drop=True)

    labels_raw = labels_raw_all.loc[valid]
    unknown = sorted(set(labels_raw) - set(mapping))
    if unknown:
        raise ValueError(
            f"Unknown class label(s) {unknown}; expected one of {sorted(mapping)}"
        )
    y = labels_raw.map(mapping).to_numpy(dtype=int)

    ids = df.loc[valid, id_col].reset_index(drop=True)
    return LoadedDataset(df_features=features, y=y, ids=ids, label_names=dict(LUNG_VOC_LABEL_NAMES))


def balanced_subsample(
    X: pd.DataFrame, y: np.ndarray, max_samples: int, seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray]:
    """Class-balanced subsample of at most ``max_samples`` rows (per-class cap).

    Guarantees both classes survive whenever present; useful to bound the O(N^2)
    quantum kernel cost on larger datasets.
    """
    if max_samples >= len(y):
        return X.reset_index(drop=True), y
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, max_samples // len(classes))
    chosen: List[np.ndarray] = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        take = min(per_class, len(idx))
        chosen.append(rng.choice(idx, size=take, replace=False))
    sel = np.sort(np.concatenate(chosen))
    return X.iloc[sel].reset_index(drop=True), y[sel]


def load_csv_dataset(
    path: Union[str, Path],
    label_column: str,
    id_column: Optional[str] = None,
    label_mapping: Optional[Dict[str, int]] = None,
    drop_nan_rows: bool = True,
) -> LoadedDataset:
    """Generic CSV ingestion with explicit label handling.

    When ``label_mapping`` is omitted, labels are auto-encoded from the sorted
    unique values (alphabetically first -> 0).
    """
    df = pd.read_csv(path)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found; columns={list(df.columns)}")

    ignore = {label_column} | ({id_column} if id_column else set())
    feature_cols = [c for c in df.columns if c not in ignore]
    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if drop_nan_rows:
        valid = features.notna().all(axis=1)
        features = features.loc[valid].reset_index(drop=True)
        df = df.loc[valid].reset_index(drop=True)

    labels_raw = df[label_column]
    if label_mapping is not None:
        normalized = labels_raw.astype(str).str.strip().str.lower()
        lowered = {k.lower(): v for k, v in label_mapping.items()}
        unknown = sorted(set(normalized) - set(lowered))
        if unknown:
            raise ValueError(f"Unknown label(s) {unknown} not in mapping {label_mapping}")
        y = normalized.map(lowered).to_numpy(dtype=int)
    else:
        levels = sorted(labels_raw.unique(), key=str)
        if len(levels) != 2:
            raise ValueError(f"Auto binary encoding needs exactly 2 labels, found {levels}")
        y = labels_raw.map({levels[0]: 0, levels[1]: 1}).to_numpy(dtype=int)

    ids = df[id_column] if id_column else None
    names = {int(v): k for k, v in (label_mapping or {}).items()}
    return LoadedDataset(df_features=features, y=y, ids=ids, label_names=names)


def _parse_literal(cell: str) -> object:
    """Safely parse list/tuple-looking cells ('nan'/'' -> None)."""
    text = cell.strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


LIST_COLUMNS = ["segments_br", "segments_hr", "ecg_pulses", "bad_ecg"]


def load_dog_ecg_metadata(path: Union[str, Path] = DOG_ECG_MARKDOWN) -> pd.DataFrame:
    """Parse the canine ECG metadata dump incl. nested list-valued cells."""
    df = load_markdown_table(path)
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(_parse_literal)
    numeric_cols = [
        c for c in ("duration", "pet_id", "weight", "age") if c in df.columns
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_dog_info(path: Union[str, Path] = DOG_INFO_MARKDOWN) -> pd.DataFrame:
    """Parse the dog metadata table (DogID, breed, weight, age months, ...)."""
    df = load_markdown_table(path)
    for col in ("DogID", "Weight", "Age months", "Gender", "NeuteringStatus"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def extract_ecg_features(ecg_df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-recording heart-rate/HRV features from R-peak annotations.

    Computed purely from the metadata table's ``ecg_pulses`` beat timestamps,
    ``segments_br`` bradycardia episodes and ``bad_ecg`` quality marks - no WAV
    audio is required (the audio files are not distributed with this repo).
    """
    records: List[Dict[str, float]] = []
    for _, row in ecg_df.iterrows():
        pulses = row.get("ecg_pulses")
        beats = np.asarray(sorted(pulses), dtype=float) if isinstance(pulses, (list, tuple)) else np.empty(0)
        rr_ms = np.diff(beats) * 1000.0 if len(beats) > 1 else np.empty(0)

        segments = row.get("segments_br")
        episodes = segments if isinstance(segments, (list, tuple)) else []
        br_durations = [
            float(seg.get("fin", 0.0)) - float(seg.get("deb", 0.0))
            for seg in episodes
            if isinstance(seg, dict) and "deb" in seg and "fin" in seg
        ]
        br_values = [
            float(seg["value"]) for seg in episodes if isinstance(seg, dict) and "value" in seg
        ]

        bad_marks = row.get("bad_ecg")
        duration = float(row.get("duration")) if pd.notna(row.get("duration")) else np.nan
        bad_seconds = sum(float(end) - float(start) for start, end in bad_marks) if isinstance(bad_marks, (list, tuple)) else 0.0

        records.append(
            {
                "_id": row.get("_id"),
                "pet_id": row.get("pet_id"),
                "breeds": row.get("breeds"),
                "duration_s": duration,
                "n_beats": len(beats),
                "hr_mean_bpm": (60_000.0 / rr_ms.mean()) if len(rr_ms) else np.nan,
                "hr_min_bpm": (60_000.0 / rr_ms.max()) if len(rr_ms) else np.nan,
                "hr_max_bpm": (60_000.0 / rr_ms.min()) if len(rr_ms) else np.nan,
                "sdnn_ms": float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else np.nan,
                "rmssd_ms": float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) > 2 else np.nan,
                "br_episode_count": len(br_durations),
                "br_total_s": float(np.sum(br_durations)) if br_durations else 0.0,
                "br_value_mean": float(np.mean(br_values)) if br_values else np.nan,
                "bad_ecg_fraction": (bad_seconds / duration) if duration and duration > 0 else np.nan,
                "has_bradycardia": int(len(br_durations) > 0),
            }
        )
    return pd.DataFrame.from_records(records)
