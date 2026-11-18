"""
HMM transition scoring for PathSnap map matching.

Implements the Newson-Krumm road-network transition model:
  P(s_t | s_{t-1}) = (1 / beta) * exp(-|d_net(s_{t-1}, s_t) - d_gps(z_{t-1}, z_t)| / beta)

Features:
  - Directed network distance traversal along one-way arcs
  - Speed feasibility enforcement (max_speed_mps = 45 m/s = 162 km/h)
  - GPS gap handling for sensor dropouts
  - Disconnected candidate penalty
  - Cutoff-constrained Dijkstra search for high-throughput decoding
"""

from typing import Any
import numpy as np

try:
    from src.data.road_graph import RoadGraph
except ImportError:
    from ..data.road_graph import RoadGraph


def _get_arc_id(cand: Any) -> int:
    if hasattr(cand, "arc_id"):
        return int(cand.arc_id)
    if isinstance(cand, dict) and "arc_id" in cand:
        return int(cand["arc_id"])
    return int(cand)


def _get_proj_frac(cand: Any) -> float:
    if hasattr(cand, "projection_fraction"):
        return float(cand.projection_fraction)
    if isinstance(cand, dict) and "projection_fraction" in cand:
        return float(cand["projection_fraction"])
    return 0.5


def calculate_transition_log_prob(
    road_graph: RoadGraph,
    prev_cand: Any,
    curr_cand: Any,
    gps_dist: float,
    dt_sec: float,
    beta_meters: float = 15.0,
    max_speed_mps: float = 45.0,
    gap_threshold_sec: float = 10.0,
    min_log_prob: float = -30.0,
) -> float:
    """
    Calculate log transition probability P(curr_cand | prev_cand) between consecutive states.

    Args:
        road_graph: RoadGraph network instance for the trip.
        prev_cand: Predecessor candidate at t-1 (has .arc_id, .projection_fraction).
        curr_cand: Successor candidate at t (has .arc_id, .projection_fraction).
        gps_dist: Great-circle / Haversine GPS displacement between t-1 and t in meters.
        dt_sec: Time difference between t-1 and t in seconds.
        beta_meters: Transition distance scale parameter (default: 15.0m).
        max_speed_mps: Plausible maximum urban vehicle speed (default: 45.0 m/s).
        gap_threshold_sec: GPS dropout threshold (default: 10.0s).
        min_log_prob: Lower bound floor for log probabilities.

    Returns:
        log_transition: float log-probability value.
    """
    dt_sec = max(0.1, float(dt_sec))
    gps_dist = max(0.0, float(gps_dist))
    beta = max(1.0, float(beta_meters))

    # Extract arc IDs and projection fractions cleanly
    prev_arc = _get_arc_id(prev_cand)
    curr_arc = _get_arc_id(curr_cand)
    prev_frac = _get_proj_frac(prev_cand)
    curr_frac = _get_proj_frac(curr_cand)

    # 1. Handle large GPS gaps (sensor dropouts)
    if dt_sec > gap_threshold_sec:
        effective_beta = beta * (1.0 + np.log(1.0 + dt_sec))
        log_p = -gps_dist / effective_beta
        return float(max(log_p, min_log_prob))

    # 2. Directed road network shortest travel distance with localized search cutoff
    cutoff = max(60.0, min(250.0, gps_dist * 2.0 + 30.0))
    net_dist = road_graph.candidate_travel_distance(
        prev_arc,
        prev_frac,
        curr_arc,
        curr_frac,
        max_dist_cutoff=cutoff,
    )

    if np.isinf(net_dist):
        # Disconnected candidate pair in directed road network
        return float(min_log_prob)

    # 3. Speed feasibility check
    speed = net_dist / dt_sec
    if speed > max_speed_mps:
        excess_speed_pen = (speed - max_speed_mps) / 5.0
        log_p = -(abs(net_dist - gps_dist) / beta) - excess_speed_pen
        return float(max(log_p, min_log_prob))

    # 4. Standard Newson-Krumm exponential transition penalty
    dist_diff = abs(net_dist - gps_dist)
    log_p = -dist_diff / beta
    return float(max(log_p, min_log_prob))
