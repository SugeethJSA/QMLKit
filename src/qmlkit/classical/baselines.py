"""Classical Machine Learning and Deep Learning Baselines for Early Cancer Detection."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class Temporal1DCNN(BaseEstimator, ClassifierMixin):
    """1D Convolutional Neural Network for processing multi-sensor time-series curves."""

    def __init__(
        self,
        n_sensors: int = 16,
        timesteps: int = 100,
        n_classes: int = 2,
        lr: float = 0.001,
        epochs: int = 25,
        batch_size: int = 32
    ):
        self.n_sensors = n_sensors
        self.timesteps = timesteps
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=n_sensors, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )
        self.is_fitted = False

    def fit(self, X_tensor: np.ndarray, y: np.ndarray) -> Temporal1DCNN:
        """Fit CNN on 3D time-series tensor (N, Sensors, Timesteps)."""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        tx = torch.tensor(X_tensor, dtype=torch.float32)
        ty = torch.tensor(y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(tx, ty)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.net(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def predict_proba(self, X_tensor: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            tx = torch.tensor(X_tensor, dtype=torch.float32)
            logits = self.net(tx)
            probs = torch.softmax(logits, dim=-1).numpy()
        return probs

    def predict(self, X_tensor: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_tensor)
        return np.argmax(probs, axis=1)


class ClassicalBaselineSuite:
    """Factory and manager for all classical comparison baselines."""

    @staticmethod
    def get_baselines(random_state: int = 42) -> Dict[str, Any]:
        """Return initialized dictionary of classical estimators."""
        baselines: Dict[str, Any] = {
            "SVM_RBF": SVC(
                kernel="rbf",
                C=1.5,
                gamma="scale",
                probability=True,
                random_state=random_state
            ),
            "SVM_Linear": SVC(
                kernel="linear",
                C=1.0,
                probability=True,
                random_state=random_state
            ),
            "Random_Forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=4,
                random_state=random_state
            ),
            "MLP_NeuralNet": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.001,
                max_iter=300,
                random_state=random_state
            )
        }

        if HAS_XGBOOST:
            baselines["XGBoost"] = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                eval_metric="logloss",
                random_state=random_state
            )
        else:
            baselines["Gradient_Boosting"] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                random_state=random_state
            )

        return baselines


def get_all_classical_baselines(random_state: int = 42) -> Dict[str, Any]:
    """Convenience accessor for baseline algorithms."""
    return ClassicalBaselineSuite.get_baselines(random_state=random_state)
