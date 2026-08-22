"""Leak-Free Biomedical Preprocessing Pipeline for Canine-Biomimetic Olfactory Signals."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass
class DatasetSplits:
    """Container for leak-free train/validation/test partitions."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    feature_names: Optional[list[str]] = None


class BiomedicalDataPipeline:
    """Preprocesses raw/extracted sensor features strictly preventing data leakage."""

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self.is_fitted = False
        self.feature_names: list[str] = []

    def fit(self, X_train: pd.DataFrame | np.ndarray) -> BiomedicalDataPipeline:
        """Fit scaler strictly on training samples only."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_mat = X_train.values
        else:
            X_mat = np.asarray(X_train)

        self.scaler.fit(X_mat)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform features using previously fitted training parameters."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on training data before transforming.")
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.scaler.transform(X_mat)

    def fit_transform(self, X_train: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit and transform training set in a single step."""
        return self.fit(X_train).transform(X_train)

    @staticmethod
    def create_leak_free_split(
        df_features: pd.DataFrame,
        y: np.ndarray | pd.Series,
        test_size: float = 0.20,
        val_size: float = 0.10,
        random_state: int = 42,
        scaler_type: str = "standard"
    ) -> Tuple[DatasetSplits, BiomedicalDataPipeline]:
        """Perform strict stratified train/val/test split and fit scalers solely on train.

        Guarantees zero data leakage (anti-defect rule L2).
        """
        y_arr = np.asarray(y)
        feat_names = list(df_features.columns)

        # 1. Stratified split for test holdout
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            df_features,
            y_arr,
            test_size=test_size,
            stratify=y_arr,
            random_state=random_state
        )

        # 2. Stratified split for validation if requested
        if val_size > 0:
            relative_val_size = val_size / (1.0 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val,
                y_train_val,
                test_size=relative_val_size,
                stratify=y_train_val,
                random_state=random_state
            )
        else:
            X_train, y_train = X_train_val, y_train_val
            X_val, y_val = None, None

        # 3. Fit scaler strictly on X_train
        pipeline = BiomedicalDataPipeline(scaler_type=scaler_type)
        X_train_scaled = pipeline.fit_transform(X_train)
        X_test_scaled = pipeline.transform(X_test)
        X_val_scaled = pipeline.transform(X_val) if X_val is not None else None

        splits = DatasetSplits(
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            X_val=X_val_scaled,
            y_val=y_val,
            feature_names=feat_names
        )
        return splits, pipeline

    def save(self, filepath: str) -> None:
        """Persist fitted pipeline to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> BiomedicalDataPipeline:
        """Load fitted pipeline from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
