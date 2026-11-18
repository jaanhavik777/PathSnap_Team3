"""
Complete route reconstruction and map geometry generation for PathSnap.

Takes observation-level predicted arc IDs and reconstructs:
  1. Complete continuous road arc sequence (interpolating intermediate arcs via directed graph search)
  2. Geographical polyline coordinates [latitude, longitude] for frontend visualization (Folium / Leaflet)
  3. Raw GPS geometry coordinates
  4. Diagnostics and continuity validation
"""

from dataclasses import dataclass
from typing import Any, List, Optional
import numpy as np

try:
    from src.data.road_graph import RoadGraph
except ImportError:
    from ..data.road_graph import RoadGraph


@dataclass
class ReconstructedRoute:
    """Represents a fully reconstructed road trajectory."""
    observation_arc_ids: List[int]
    reconstructed_arc_sequence: List[int]
    route_geometry: List[List[float]]  # List of [lat, lon] pairs
    gps_geometry: List[List[float]]    # List of [lat, lon] pairs
    continuity_ratio: float
    disconnected_jumps: int
    warnings: List[str]


def reconstruct_continuous_route(
    observation_arc_ids: List[int],
    road_graph: RoadGraph,
    gps_lons: np.ndarray,
    gps_lats: np.ndarray,
    max_gap_search_dist_m: float = 1000.0,
) -> ReconstructedRoute:
    """
    Reconstruct continuous road arc sequence and map geometries from observation-level predictions.

    Args:
        observation_arc_ids: Arc ID predicted for each GPS observation.
        road_graph: RoadGraph instance for the trip.
        gps_lons: Array of GPS longitudes.
        gps_lats: Array of GPS latitudes.
        max_gap_search_dist_m: Maximum distance threshold for path search between disconnected arcs.

    Returns:
        ReconstructedRoute instance.
    """
    N = len(observation_arc_ids)
    warnings: List[str] = []

    # GPS geometry as [lat, lon] pairs
    gps_geometry = [[float(gps_lats[i]), float(gps_lons[i])] for i in range(len(gps_lats))]

    if N == 0:
        return ReconstructedRoute([], [], [], gps_geometry, 1.0, 0, ["Empty observation arcs."])

    # Filter invalid (-1) arcs
    valid_obs_arcs = [a for a in observation_arc_ids if a >= 0]
    if len(valid_obs_arcs) == 0:
        return ReconstructedRoute(
            observation_arc_ids, [], [], gps_geometry, 0.0, 0, ["No valid arcs predicted."]
        )

    # Reconstruct continuous arc sequence
    reconstructed_arcs: List[int] = []
    reconstructed_arcs.append(valid_obs_arcs[0])

    disconnected_jumps = 0
    connected_steps = 0
    total_steps = 0

    for i in range(1, len(valid_obs_arcs)):
        prev_arc = valid_obs_arcs[i - 1]
        curr_arc = valid_obs_arcs[i]
        total_steps += 1

        if prev_arc == curr_arc:
            # Same arc continues
            connected_steps += 1
            continue

        u_prev, v_prev = road_graph.get_arc_nodes(prev_arc)
        u_curr, v_curr = road_graph.get_arc_nodes(curr_arc)

        if v_prev == u_curr:
            # Directly connected consecutive arcs
            reconstructed_arcs.append(curr_arc)
            connected_steps += 1
        else:
            # Try to find intermediate shortest path connecting v_prev to u_curr
            intermediate_arcs = road_graph.get_shortest_path_arcs(v_prev, u_curr)
            if intermediate_arcs is not None and len(intermediate_arcs) > 0:
                # Connected via intermediate arcs
                reconstructed_arcs.extend(intermediate_arcs)
                reconstructed_arcs.append(curr_arc)
                connected_steps += 1
            else:
                # Disconnected in directed road graph
                disconnected_jumps += 1
                warnings.append(
                    f"Disconnected road transition: arc {prev_arc} (node {v_prev}) -> arc {curr_arc} (node {u_curr})"
                )
                reconstructed_arcs.append(curr_arc)

    continuity_ratio = float(connected_steps / total_steps) if total_steps > 0 else 1.0

    # Build geographical coordinates [lat, lon] along reconstructed route
    route_geometry: List[List[float]] = []

    for arc_id in reconstructed_arcs:
        u, v = road_graph.get_arc_nodes(arc_id)
        u_lon, u_lat = road_graph.node_coords[u, 0], road_graph.node_coords[u, 1]
        v_lon, v_lat = road_graph.node_coords[v, 0], road_graph.node_coords[v, 1]

        u_pt = [float(u_lat), float(u_lon)]
        v_pt = [float(v_lat), float(v_lon)]

        if len(route_geometry) == 0:
            route_geometry.append(u_pt)
            route_geometry.append(v_pt)
        else:
            # Avoid duplicate vertex if connected to previous endpoint
            last_pt = route_geometry[-1]
            if abs(last_pt[0] - u_pt[0]) > 1e-7 or abs(last_pt[1] - u_pt[1]) > 1e-7:
                route_geometry.append(u_pt)
            route_geometry.append(v_pt)

    return ReconstructedRoute(
        observation_arc_ids=observation_arc_ids,
        reconstructed_arc_sequence=reconstructed_arcs,
        route_geometry=route_geometry,
        gps_geometry=gps_geometry,
        continuity_ratio=continuity_ratio,
        disconnected_jumps=disconnected_jumps,
        warnings=warnings,
    )
