"""
Reference route alignment algorithm for PathSnap.

Maps chronological GPS observations (at ~1 Hz) to the sequential reference route arcs.
Because vehicles move through short road segments (median length ~10m) between GPS pings,
some intermediate arcs are traversed between two consecutive pings, while longer arcs
span multiple pings.

This module determines the most likely ground-truth arc for each GPS observation t
using forward-progress constrained dynamic matching.
"""

from typing import List, Optional, Tuple
import numpy as np

try:
    from src.data.road_graph import RoadGraph
    from src.data.spatial import (
        LocalProjector,
        project_point_to_segment_2d,
        angle_difference,
    )
except ImportError:
    from .road_graph import RoadGraph
    from .spatial import (
        LocalProjector,
        project_point_to_segment_2d,
        angle_difference,
    )


def align_reference_route_to_observations(
    lons: np.ndarray,
    lats: np.ndarray,
    timestamps: np.ndarray,
    road_graph: RoadGraph,
    reference_route: np.ndarray,
    search_window: int = 15,
) -> np.ndarray:
    """
    Align GPS observations to the reference route sequence.

    Args:
        lons: array of GPS longitudes, shape (N,)
        lats: array of GPS latitudes, shape (N,)
        timestamps: array of GPS timestamps, shape (N,)
        road_graph: RoadGraph of the trip
        reference_route: array of ground truth arc IDs, shape (K,)
        search_window: maximum forward arc lookahead in the reference route

    Returns:
        ground_truth_arc_ids: array of shape (N,), where index t contains
                              the assigned ground-truth arc ID for observation t.
    """
    N = len(lons)
    K = len(reference_route)
    if K == 0 or N == 0:
        return np.full(N, -1, dtype=np.int64)

    proj = LocalProjector(lons[0], lats[0])
    gps_x, gps_y = proj.forward_batch(lons, lats)

    # Pre-project all route arc endpoints to local Cartesian meters
    route_geom = []
    for aid in reference_route:
        u, v = road_graph.get_arc_nodes(int(aid))
        ax, ay = proj.forward(road_graph.node_coords[u, 0], road_graph.node_coords[u, 1])
        bx, by = proj.forward(road_graph.node_coords[v, 0], road_graph.node_coords[v, 1])
        bearing = road_graph.get_arc_bearing(int(aid))
        route_geom.append((ax, ay, bx, by, int(aid), bearing))

    aligned_arcs = np.zeros(N, dtype=np.int64)
    curr_k = 0

    for t in range(N):
        px, py = gps_x[t], gps_y[t]

        # Estimate vehicle travel heading
        if t < N - 1:
            vx, vy = gps_x[t + 1] - px, gps_y[t + 1] - py
        elif t > 0:
            vx, vy = px - gps_x[t - 1], py - gps_y[t - 1]
        else:
            vx, vy = 0.0, 0.0

        v_speed = np.hypot(vx, vy)
        v_heading = (
            (np.degrees(np.arctan2(vx, vy)) + 360.0) % 360.0
            if v_speed > 0.5
            else None
        )

        best_cost = float("inf")
        best_k = curr_k
        best_aid = route_geom[curr_k][4]

        # Search forward in a bounded window
        search_max = min(K, curr_k + search_window)
        for k in range(curr_k, search_max):
            ax, ay, bx, by, aid, bearing = route_geom[k]
            _, _, dist, frac = project_point_to_segment_2d(px, py, ax, ay, bx, by)

            heading_pen = 0.0
            if v_heading is not None:
                h_diff = angle_difference(v_heading, bearing)
                heading_pen = 15.0 * (h_diff / 180.0)

            # Progress penalty encourages forward advancement along the route
            progress_pen = 1.5 * (k - curr_k)
            cost = dist + heading_pen + progress_pen

            if cost < best_cost:
                best_cost = cost
                best_k = k
                best_aid = aid

        aligned_arcs[t] = best_aid
        # Update current pointer, allowing small backtrack (at most 1) if noisy
        curr_k = max(0, best_k - 1)

    return aligned_arcs
