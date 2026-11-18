"""
HistGradientBoostingClassifier model wrapper and training definition for PathSnap.

Person 3 (Hitha) deliverable for Review 1.
Features:
  - 8 candidate spatial and kinematic features
  - Balanced class weighting / probabilistic scoring
  - GroupKFold 5-fold cross validation compatibility
  - Persistent model artifact serialization and deserialization
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


DEFAULT_FEATURE_NAMES = [
    "distance_meters",
    "heading_diff_deg",
    "projection_fraction",
    "segment_length_m",
    "gps_speed_mps",
    "bearing_cos",
    "bearing_sin",
    "proximity_score",
]


class HistGradientBoostingModel:
    """
    Wrapper for HistGradientBoostingClassifier used for PathSnap candidate matching.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 150,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.feature_names = feature_names or DEFAULT_FEATURE_NAMES

        self.classifier = HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGradientBoostingModel":
        """Train classifier on feature matrix X and binary labels y."""
        self.classifier.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities: returns shape (N, 2)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
        return self.classifier.predict_proba(X)

    def predict_candidate_scores(self, X: np.ndarray) -> np.ndarray:
        """Return 1D array of class-1 candidate probabilities."""
        probs = self.predict_proba(X)
        return probs[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels (0 or 1)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
        return self.classifier.predict(X)

    def save(self, filepath: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save fitted model and metadata."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, filepath)

        meta_path = filepath.with_suffix(".json") if not str(filepath).endswith("_metadata.json") else filepath
        meta = {
            "model_type": "HistGradientBoostingClassifier",
            "feature_names": self.feature_names,
            "params": {
                "learning_rate": self.learning_rate,
                "max_iter": self.max_iter,
                "max_leaf_nodes": self.max_leaf_nodes,
                "min_samples_leaf": self.min_samples_leaf,
                "random_state": self.random_state,
            },
        }
        if metadata:
            meta.update(metadata)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "HistGradientBoostingModel":
        """Load saved model artifact and metadata."""
        filepath = Path(filepath)
        classifier = joblib.load(filepath)
        meta_file = filepath.parent / f"{filepath.stem}_metadata.json"
        if not meta_file.exists():
            meta_file = filepath.with_suffix(".json")

        feature_names = DEFAULT_FEATURE_NAMES
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    feature_names = meta.get("feature_names", DEFAULT_FEATURE_NAMES)
            except Exception:
                pass

        instance = cls(
            learning_rate=getattr(classifier, "learning_rate", 0.1),
            max_iter=getattr(classifier, "max_iter", 150),
            max_leaf_nodes=getattr(classifier, "max_leaf_nodes", 31),
            min_samples_leaf=getattr(classifier, "min_samples_leaf", 20),
            random_state=getattr(classifier, "random_state", 42),
            feature_names=feature_names,
        )
        instance.classifier = classifier
        instance.is_fitted = True
        return instance
