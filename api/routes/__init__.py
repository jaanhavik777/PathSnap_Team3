"""
API routes package for PathSnap.
"""

from .health import router as health_router
from .trips import router as trips_router
from .predict import router as predict_router
from .model_info import router as model_info_router

__all__ = [
    "health_router",
    "trips_router",
    "predict_router",
    "model_info_router",
]
