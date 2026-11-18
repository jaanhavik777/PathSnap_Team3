"""
Server-side ground truth evaluation service for PathSnap API.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Sequence
from api.dependencies import DATA_ROOT
from src.pipeline.evaluation import evaluate_prediction


def execute_server_evaluation(
    trip_id: str,
    predicted_arc_ids: Sequence[int],
    data_dir: Path = DATA_ROOT,
) -> Dict[str, Any]:
    """
    Perform isolated server-side evaluation comparing predictions to ground truth.
    Never exposes ground truth arc IDs or geometries.
    """
    return evaluate_prediction(
        trip_id=trip_id,
        predicted_arc_ids=predicted_arc_ids,
        data_dir=data_dir,
    )
