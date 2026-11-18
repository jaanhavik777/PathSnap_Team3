"""
Configuration and dependency injection for PathSnap FastAPI backend.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional
from src.pipeline.model_adapter import ModelAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_default_data_root() -> Path:
    """Find active data directory dynamically."""
    candidates = [
        REPO_ROOT / "data" / "raw" / "student_test",
        REPO_ROOT / "data" / "raw",
        REPO_ROOT.parent / "hack" / "Student folder" / "Student folder" / "student_test",
        REPO_ROOT.parent / "hack" / "Student folder" / "student_test",
    ]
    env_root = os.getenv("DATA_ROOT")
    if env_root:
        p = Path(env_root)
        if p.exists():
            return p

    for c in candidates:
        if c.exists() and any(c.glob("*/*.track")):
            return c
        if c.exists() and any(c.glob("*.track")):
            return c

    for c in candidates:
        if c.exists():
            return c

    return REPO_ROOT / "data" / "raw" / "student_test"


def get_default_model_path() -> Path:
    """Locate active classifier model artifact."""
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    candidates = [
        REPO_ROOT / "artifacts" / "models" / "hist_gradient_boosting_enhanced.joblib",
        REPO_ROOT / "artifacts" / "models" / "hist_gradient_boosting.joblib",
        REPO_ROOT / "artifacts" / "models" / "mlp" / "mlp_pipeline.joblib",
        REPO_ROOT.parent / "hack" / "artifacts" / "models" / "hist_gradient_boosting.joblib",
    ]
    for c in candidates:
        if c.exists():
            return c

    return REPO_ROOT / "artifacts" / "models" / "hist_gradient_boosting.joblib"


DATA_ROOT = get_default_data_root()
MODEL_PATH = get_default_model_path()
METADATA_PATH = Path(os.getenv("METADATA_PATH", str(MODEL_PATH.parent / f"{MODEL_PATH.stem}_metadata.json")))
EVALUATION_PARTITION = os.getenv("EVALUATION_PARTITION", "student_test")
ALLOWED_ORIGINS = [orig.strip() for orig in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

HMM_DEFAULT_CONFIG: Dict[str, float] = {
    "beta_meters": float(os.getenv("HMM_BETA", "15.0")),
    "sigma_z_meters": float(os.getenv("HMM_SIGMA_Z", "12.0")),
    "max_speed_mps": float(os.getenv("HMM_MAX_SPEED", "45.0")),
    "gap_threshold_sec": float(os.getenv("HMM_GAP_THRESHOLD", "10.0")),
}

_model_adapter: Optional[ModelAdapter] = None


def get_model_adapter() -> ModelAdapter:
    """Return initialized singleton ModelAdapter."""
    global _model_adapter
    if _model_adapter is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
        _model_adapter = ModelAdapter(MODEL_PATH, METADATA_PATH)
    return _model_adapter
