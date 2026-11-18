"""
Evaluation metrics for map-matched routes and candidate predictions in PathSnap.
"""

from typing import Any, List, Sequence
import numpy as np

try:
    from src.data.road_graph import RoadGraph
except ImportError:
    from ..data.road_graph import RoadGraph


def calculate_route_continuity(
    observation_arc_ids: Sequence[int],
    road_graph: RoadGraph,
) -> float:
    """
    Calculate fraction of consecutive observation transitions that are physically connected
    in the directed road graph (either same arc or u_next == v_prev).

    Returns:
        continuity: float in [0.0, 1.0].
    """
    valid_arcs = [a for a in observation_arc_ids if a >= 0]
    if len(valid_arcs) <= 1:
        return 1.0

    connected = 0
    total = len(valid_arcs) - 1

    for i in range(total):
        a1 = valid_arcs[i]
        a2 = valid_arcs[i + 1]
        if road_graph.is_connected_pair(a1, a2):
            connected += 1

    return float(connected / total) if total > 0 else 1.0


def calculate_invalid_transition_rate(
    observation_arc_ids: Sequence[int],
    road_graph: RoadGraph,
) -> float:
    """
    Calculate 1.0 - continuity: fraction of transitions that are disconnected jumps.
    """
    return float(1.0 - calculate_route_continuity(observation_arc_ids, road_graph))


def calculate_top1_accuracy(
    pred_arc_ids: Sequence[int],
    true_arc_ids: Sequence[int],
) -> float:
    """
    Calculate observation-level Top-1 arc matching accuracy against ground truth route.

    Returns:
        accuracy: float in [0.0, 1.0].
    """
    if len(pred_arc_ids) != len(true_arc_ids):
        raise ValueError(
            f"Length mismatch: pred ({len(pred_arc_ids)}) vs true ({len(true_arc_ids)})"
        )
    if len(pred_arc_ids) == 0:
        return 0.0

    preds = np.asarray(pred_arc_ids, dtype=np.int64)
    trues = np.asarray(true_arc_ids, dtype=np.int64)
    return float(np.mean(preds == trues))


def _get_arc_id(cand: Any) -> int:
    if hasattr(cand, "arc_id"):
        return int(cand.arc_id)
    if isinstance(cand, dict) and "arc_id" in cand:
        return int(cand["arc_id"])
    return int(cand)


def calculate_candidate_miss_rate(
    candidates_per_obs: List[List[Any]],
    true_arc_ids: Sequence[int],
) -> float:
    """
    Calculate candidate miss rate: fraction of observations where ground-truth arc
    was not included in the candidate set within search radius.

    Returns:
        miss_rate: float in [0.0, 1.0].
    """
    N = len(true_arc_ids)
    if N == 0:
        return 0.0

    misses = 0
    for t in range(N):
        true_arc = true_arc_ids[t]
        cands = candidates_per_obs[t] if t < len(candidates_per_obs) else []
        cand_arcs = {_get_arc_id(c) for c in cands}
        if true_arc not in cand_arcs:
            misses += 1

    return float(misses / N)
