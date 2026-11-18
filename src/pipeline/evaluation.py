"""
Server-side ground-truth evaluation service for PathSnap.

Runs strictly on Laptop 1 post-prediction.
Ground-truth .route files are accessed on the server ONLY if available.
Ground-truth arc IDs and reference geometries are NEVER exposed in API responses.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from src.data.parsers import read_routes, read_nodes, read_arcs, read_tracks
from src.data.road_graph import RoadGraph
from src.data.alignment import align_reference_route_to_observations
from src.hmm.route_metrics import (
    calculate_route_continuity,
    calculate_invalid_transition_rate,
    calculate_top1_accuracy,
)


def find_reference_route_path(trip_id: str, data_dir: Union[str, Path]) -> Optional[Path]:
    """Find reference route path on server if present."""
    trip_dir = Path(data_dir) / trip_id
    route_file = trip_dir / f"{trip_id}.route"
    if route_file.exists():
        return route_file

    matches = list(Path(data_dir).glob(f"*/{trip_id}/{trip_id}.route"))
    if matches and matches[0].exists():
        return matches[0]

    parent_matches = list(Path(data_dir).parent.glob(f"**/{trip_id}/{trip_id}.route"))
    if parent_matches and parent_matches[0].exists():
        return parent_matches[0]

    return None


def evaluate_prediction(
    trip_id: str,
    predicted_arc_ids: Sequence[int],
    data_dir: Union[str, Path],
    road_graph: Optional[RoadGraph] = None,
) -> Dict[str, Any]:
    """
    Perform server-side accuracy calculation comparing predicted arcs to ground truth.

    Args:
        trip_id: Trip ID string
        predicted_arc_ids: Sequence of predicted road arc IDs
        data_dir: Data directory on server
        road_graph: Optional RoadGraph instance for continuity check

    Returns:
        Dictionary of aggregate evaluation metrics:
          - metrics_available (bool)
          - top1_accuracy (float or None)
          - route_continuity (float or None)
          - invalid_transition_rate (float or None)
          - num_eval_observations (int)
          - evaluation_reason (str)
    """
    route_path = find_reference_route_path(trip_id, data_dir)
    N = len(predicted_arc_ids)

    if route_path is None or not route_path.exists():
        return {
            "metrics_available": False,
            "top1_accuracy": None,
            "route_continuity": None,
            "invalid_transition_rate": None,
            "num_eval_observations": N,
            "evaluation_reason": "Reference ground-truth route is not available on server for this trip partition.",
        }

    try:
        route_df = read_routes(route_path)
        if route_df is None or len(route_df) == 0:
            return {
                "metrics_available": False,
                "top1_accuracy": None,
                "route_continuity": None,
                "invalid_transition_rate": None,
                "num_eval_observations": N,
                "evaluation_reason": "Reference route file is empty.",
            }

        raw_route = route_df["arc_id"].to_numpy(dtype=np.int64)

        # Build road graph if needed for alignment
        trip_dir = route_path.parent
        track_df = read_tracks(trip_dir / f"{trip_id}.track")
        lons = track_df["longitude"].to_numpy(dtype=np.float64)
        lats = track_df["latitude"].to_numpy(dtype=np.float64)
        timestamps = track_df["timestamp"].to_numpy(dtype=np.float64)

        if road_graph is None:
            nodes_df = read_nodes(trip_dir / f"{trip_id}.nodes")
            arcs_df = read_arcs(trip_dir / f"{trip_id}.arcs")
            node_coords = nodes_df[["longitude", "latitude"]].to_numpy(dtype=np.float64)
            arc_nodes = arcs_df[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
            road_graph = RoadGraph(node_coords, arc_nodes, trip_id=trip_id)

        # Align reference route to chronological GPS observations
        if len(raw_route) == N:
            gt_arcs = raw_route
        else:
            gt_arcs = align_reference_route_to_observations(
                lons, lats, timestamps, road_graph, raw_route
            )

        top1_acc = calculate_top1_accuracy(predicted_arc_ids, gt_arcs)
        continuity = calculate_route_continuity(predicted_arc_ids, road_graph)
        inv_rate = calculate_invalid_transition_rate(predicted_arc_ids, road_graph)

        return {
            "metrics_available": True,
            "top1_accuracy": round(float(top1_acc), 4),
            "route_continuity": round(float(continuity), 4),
            "invalid_transition_rate": round(float(inv_rate), 4),
            "num_eval_observations": int(N),
            "evaluation_reason": "Evaluated against verified server-side ground-truth route.",
        }

    except Exception as exc:
        return {
            "metrics_available": False,
            "top1_accuracy": None,
            "route_continuity": None,
            "invalid_transition_rate": None,
            "num_eval_observations": N,
            "evaluation_reason": f"Server-side evaluation error: {str(exc)}",
        }
