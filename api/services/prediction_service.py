"""
Prediction service coordinating pure ML + HMM inference without ground truth.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
from api.dependencies import DATA_ROOT, HMM_DEFAULT_CONFIG, get_model_adapter
from src.pipeline.inference import predict_trip


def execute_prediction(
    trip_id: str,
    data_dir: Path = DATA_ROOT,
    hmm_config: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Execute map-matching prediction pipeline on selected trip.
    Does NOT access ground truth .route files.
    """
    adapter = get_model_adapter()
    cfg = hmm_config or HMM_DEFAULT_CONFIG
    return predict_trip(
        trip_id=trip_id,
        data_dir=data_dir,
        model_adapter=adapter,
        hmm_config=cfg,
    )
