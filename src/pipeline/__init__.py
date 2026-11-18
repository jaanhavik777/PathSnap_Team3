"""
Pipeline package for PathSnap.
"""

from .model_adapter import ModelAdapter, HGB_FEATURE_NAMES, SHARED_FEATURE_NAMES
from .inference import predict_trip
from .evaluation import evaluate_prediction

__all__ = [
    "ModelAdapter",
    "HGB_FEATURE_NAMES",
    "SHARED_FEATURE_NAMES",
    "predict_trip",
    "evaluate_prediction",
]
