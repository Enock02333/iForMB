from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier

from .calibration import calibrate_isolation_scores


@dataclass
class IFMLPResult:
    probabilities: np.ndarray
    predictions: np.ndarray
    evidence: np.ndarray


class IFMLPDetector:
    """iForMB detector: training-only IF calibration plus early feature-level fusion."""

    def __init__(self, seed=13, n_estimators=300, max_samples=10_000):
        self.seed = seed
        self.iforest = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        )
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=5e-4,
            early_stopping=True,
            max_iter=300,
            random_state=seed,
        )
        self.normal_scores_: np.ndarray | None = None
        self.threshold_: float | None = None

    @staticmethod
    def _raw_score(model, x):
        return -model.score_samples(x)

    def fit(self, x_train, y_train):
        y = np.asarray(y_train, dtype=int)
        normal = x_train[y == 0]
        if normal.shape[0] == 0:
            raise ValueError("At least one normal training observation is required")
        self.iforest.fit(normal)
        self.normal_scores_ = self._raw_score(self.iforest, normal)
        evidence = calibrate_isolation_scores(self.normal_scores_, self._raw_score(self.iforest, x_train))
        fused = sparse.hstack([x_train, sparse.csr_matrix(evidence)], format="csr") if sparse.issparse(x_train) else np.column_stack([x_train, evidence])
        self.mlp.fit(fused, y)
        return self

    def score(self, x):
        if self.normal_scores_ is None:
            raise RuntimeError("Detector is not fitted")
        evidence = calibrate_isolation_scores(self.normal_scores_, self._raw_score(self.iforest, x))
        fused = sparse.hstack([x, sparse.csr_matrix(evidence)], format="csr") if sparse.issparse(x) else np.column_stack([x, evidence])
        probability = self.mlp.predict_proba(fused)[:, 1]
        return probability, evidence

    def predict(self, x, threshold=None):
        probability, evidence = self.score(x)
        selected = self.threshold_ if threshold is None else threshold
        if selected is None:
            raise RuntimeError("A validation-selected threshold is required")
        return IFMLPResult(probability, (probability >= selected).astype(int), evidence)
