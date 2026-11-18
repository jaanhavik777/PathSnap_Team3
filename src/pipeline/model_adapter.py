"""
Unified model adapter for PathSnap candidate classifiers.

Supports:
  - HistGradientBoostingClassifier (Person 3)
  - Random Forest, MLP, Logistic Regression, KNN (Person 2 / Person 4)
  - Automatic feature schema resolution
  - Robust probability extraction and calibration clipping
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np


# 8-feature schema used by HistGradientBoosting baseline
HGB_FEATURE_NAMES = [
    "distance_meters",
    "heading_diff_deg",
    "projection_fraction",
    "segment_length_m",
    "gps_speed_mps",
    "bearing_cos",
    "bearing_sin",
    "proximity_score",
]

# 9-feature schema used by shared team pipeline
SHARED_FEATURE_NAMES = [
    "distance_m",
    "projection_fraction",
    "candidate_rank",
    "arc_length_m",
    "from_node_in_degree",
    "to_node_out_degree",
    "step_distance_m",
    "speed_mps",
    "heading_error_deg",
]


class ModelAdapter:
    """
    Uniform classifier adapter for candidate classification models.
    """

    def __init__(
        self,
        model_artifact_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
    ):
        self.model_path = Path(model_artifact_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")

        # Load fitted estimator or pipeline
        raw_obj = joblib.load(self.model_path)
        if hasattr(raw_obj, "classifier"):
            self.model = raw_obj.classifier
        else:
            self.model = raw_obj

        # Load metadata if present
        if metadata_path is None:
            metadata_path = self.model_path.parent / f"{self.model_path.stem}_metadata.json"
            if not metadata_path.exists():
                metadata_path = self.model_path.with_suffix(".json")
        else:
            metadata_path = Path(metadata_path)

        self.metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}

        self.model_name = self.metadata.get(
            "model_name", type(self.model).__name__
        )
        self.model_family = self.metadata.get("model_family", "Classifier")

        # Determine feature names
        if "feature_names" in self.metadata:
            self.feature_names = list(self.metadata["feature_names"])
        elif "features" in self.metadata and isinstance(self.metadata["features"], list):
            self.feature_names = list(self.metadata["features"])
        elif "hist_gradient_boosting" in str(self.model_path).lower():
            self.feature_names = list(HGB_FEATURE_NAMES)
        else:
            self.feature_names = list(SHARED_FEATURE_NAMES)

    def predict_candidate_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return positive class (class 1: ground truth match) probability scores.
        Input X has shape (M, num_features).
        Returns 1D array of shape (M,) with values in [0.0, 1.0].
        """
        if len(X) == 0:
            return np.array([], dtype=np.float64)

        X_mat = np.asarray(X, dtype=np.float64)

        # 1. Standard predict_proba
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(X_mat)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    return np.ascontiguousarray(proba[:, 1], dtype=np.float64)
                elif proba.ndim == 1:
                    return np.ascontiguousarray(proba, dtype=np.float64)
            except Exception:
                pass

        # 2. Decision function (SVM / linear)
        if hasattr(self.model, "decision_function"):
            try:
                df = self.model.decision_function(X_mat)
                sigmoid = 1.0 / (1.0 + np.exp(-np.clip(df, -25.0, 25.0)))
                return np.ascontiguousarray(sigmoid, dtype=np.float64)
            except Exception:
                pass

        # 3. Fallback to predict
        if hasattr(self.model, "predict"):
            preds = self.model.predict(X_mat)
            return np.ascontiguousarray(preds, dtype=np.float64)

        raise AttributeError(f"Model {type(self.model)} does not support scoring.")

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata for /model-info endpoint."""
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "model_type": str(type(self.model).__name__),
            "feature_names": self.feature_names,
            "num_features": len(self.feature_names),
            "metadata": self.metadata,
        }
