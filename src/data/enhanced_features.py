"""
Enhanced, inference-safe candidate generation and feature extraction for PathSnap.

Incorporates:
  1. Dense spatial indexing along road segments (eliminates candidate misses on long segments).
  2. Exact 2D point-to-segment orthogonal projection in local metric plane.
  3. Absolute spatial & kinematic features:
     - distance_meters
     - heading_diff_deg
     - heading_cos, heading_sin
     - projection_fraction
     - segment_length_m
     - gps_speed_mps
     - step_distance_m
     - proximity_score
  4. Relative ranking features across candidate set of each observation:
     - dist_gap (distance - min_candidate_distance)
     - dist_ratio (distance / min_candidate_distance)
     - dist_rank (1-based rank by distance)
  5. Local graph topology features:
     - from_node_in_degree
     - to_node_out_degree

STRICTLY LEAK-FREE:
  - Uses only GPS observations (lon, lat, timestamp) and road network geometry (nodes, arcs).
  - Ground truth (.route) is NEVER accessed or referenced.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial import cKDTree

from src.data.road_graph import RoadGraph
from src.data.spatial import (
    LocalProjector,
    project_point_to_segment_2d,
    calculate_bearing,
    angle_difference,
    haversine_distance,
)
from src.hmm.states import CandidateState


ENHANCED_FEATURE_NAMES = [
    "dist_rank",              # 1-based candidate rank by distance
    "dist_gap",               # dist - min_dist (m)
    "dist_ratio",             # dist / min_dist
    "distance_meters",        # orthogonal distance to segment (m)
    "heading_cos",            # cos(heading_diff)
    "heading_diff_deg",       # angle difference [0, 180] deg
    "heading_sin",            # sin(heading_diff)
    "projection_fraction",    # fraction along segment [0, 1]
    "segment_length_m",       # physical arc length (m)
    "gps_speed_mps",          # estimated GPS travel speed (m/s)
    "step_distance_m",        # distance traveled from previous GPS point (m)
    "proximity_score",        # non-linear proximity: 1 / (1 + (d/12)^2)
    "from_node_in_degree",    # number of arcs entering start node
    "to_node_out_degree",     # number of arcs exiting end node
]


class EnhancedSpatialIndex:
    """
    High-recall spatial index placing adaptive seeds along arcs at most step_m apart.
    Guarantees no point near any segment is missed due to arc length.
    """
    def __init__(self, road_graph: RoadGraph, projector: LocalProjector, step_m: float = 20.0):
        self.road_graph = road_graph
        self.projector = projector
        self.step_m = step_m

        node_x, node_y = projector.forward_batch(
            road_graph.node_coords[:, 0], road_graph.node_coords[:, 1]
        )
        self.node_x = node_x
        self.node_y = node_y

        sample_pts = []
        sample_arc_ids = []

        for a in range(road_graph.num_arcs):
            u, v = road_graph.get_arc_nodes(a)
            ax, ay = node_x[u], node_y[u]
            bx, by = node_x[v], node_y[v]
            dx, dy = bx - ax, by - ay
            length = np.hypot(dx, dy)

            # Endpoints
            sample_pts.append([ax, ay])
            sample_arc_ids.append(a)
            sample_pts.append([bx, by])
            sample_arc_ids.append(a)

            # Intermediate seeds on longer segments
            if length > step_m:
                n_inter = int(np.ceil(length / step_m))
                for step in range(1, n_inter):
                    t = step / n_inter
                    sample_pts.append([ax + t * dx, ay + t * dy])
                    sample_arc_ids.append(a)

        self.kd_tree = cKDTree(np.array(sample_pts, dtype=np.float32))
        self.sample_arc_ids = np.array(sample_arc_ids, dtype=np.int32)

        # Precompute in/out degrees for network topology
        self.in_degrees = np.zeros(road_graph.num_nodes, dtype=np.int32)
        self.out_degrees = np.zeros(road_graph.num_nodes, dtype=np.int32)
        for a in range(road_graph.num_arcs):
            u, v = road_graph.get_arc_nodes(a)
            self.out_degrees[u] += 1
            self.in_degrees[v] += 1

    def query(
        self,
        gx: float,
        gy: float,
        radius_m: float = 75.0,
        max_cands: int = 12,
    ) -> List[Tuple[int, float, float]]:
        """
        Returns list of (arc_id, distance_m, projection_fraction) sorted by distance.
        """
        query_r = radius_m + (self.step_m * 0.5)
        indices = self.kd_tree.query_ball_point([gx, gy], r=query_r)
        if len(indices) == 0:
            _, nearest_idx = self.kd_tree.query([gx, gy], k=1)
            candidate_arcs = [int(self.sample_arc_ids[nearest_idx])]
        else:
            candidate_arcs = list(dict.fromkeys(self.sample_arc_ids[indices]))

        scored = []
        for aid in candidate_arcs:
            u, v = self.road_graph.get_arc_nodes(aid)
            ax, ay = self.node_x[u], self.node_y[u]
            bx, by = self.node_x[v], self.node_y[v]
            _, _, dist, frac = project_point_to_segment_2d(gx, gy, ax, ay, bx, by)
            if dist <= radius_m or len(scored) == 0:
                scored.append((aid, dist, frac))

        scored.sort(key=lambda x: x[1])
        return scored[:max_cands]


def extract_enhanced_candidates(
    road_graph: RoadGraph,
    lons: np.ndarray,
    lats: np.ndarray,
    timestamps: np.ndarray,
    search_radius_m: float = 75.0,
    max_cands: int = 12,
) -> Tuple[List[List[CandidateState]], np.ndarray, List[int]]:
    """
    Generate spatial candidates and 14 enhanced features for each observation.
    Returns:
      obs_candidates: List of List[CandidateState] per observation
      X_features: 2D numpy array of shape (total_candidates, 14)
      row_to_obs: List mapping each row in X_features back to observation index t
    """
    N = len(lons)
    mean_lon = float(np.mean(lons))
    mean_lat = float(np.mean(lats))
    projector = LocalProjector(mean_lon, mean_lat)
    spatial_index = EnhancedSpatialIndex(road_graph, projector, step_m=20.0)

    # Compute vehicle motion kinematics from GPS track
    obs_headings = np.zeros(N, dtype=np.float64)
    obs_speeds = np.zeros(N, dtype=np.float64)
    step_distances = np.zeros(N, dtype=np.float64)

    for i in range(N - 1):
        d = haversine_distance(lons[i], lats[i], lons[i + 1], lats[i + 1])
        dt = max(0.1, timestamps[i + 1] - timestamps[i])
        step_distances[i + 1] = d
        obs_speeds[i] = d / dt
        obs_headings[i] = calculate_bearing(lons[i], lats[i], lons[i + 1], lats[i + 1])
    obs_speeds[-1] = obs_speeds[-2] if N > 1 else 0.0
    obs_headings[-1] = obs_headings[-2] if N > 1 else 0.0
    step_distances[0] = step_distances[1] if N > 1 else 0.0

    gps_x, gps_y = projector.forward_batch(lons, lats)

    all_obs_candidates: List[List[CandidateState]] = []
    all_feature_rows: List[np.ndarray] = []
    feature_row_to_obs: List[int] = []

    for t in range(N):
        gx, gy = gps_x[t], gps_y[t]
        cands_raw = spatial_index.query(gx, gy, radius_m=search_radius_m, max_cands=max_cands)

        if not cands_raw:
            # Fallback to nearest
            cands_raw = spatial_index.query(gx, gy, radius_m=300.0, max_cands=1)

        min_dist = cands_raw[0][1] if cands_raw else 0.0

        cands_t: List[CandidateState] = []
        for rank, (arc_id, dist, frac) in enumerate(cands_raw, start=1):
            u, v = road_graph.get_arc_nodes(arc_id)
            arc_bearing = road_graph.get_arc_bearing(arc_id)
            head_diff = angle_difference(obs_headings[t], arc_bearing)
            seg_len = road_graph.get_arc_length(arc_id)
            speed = obs_speeds[t]
            step_d = step_distances[t]

            bearing_rad = np.radians(head_diff)
            bearing_cos = float(np.cos(bearing_rad))
            bearing_sin = float(np.sin(bearing_rad))
            prox_score = float(1.0 / (1.0 + (dist / 12.0) ** 2))

            dist_gap = float(dist - min_dist)
            dist_ratio = float(dist / (min_dist + 1e-3))
            dist_rank = float(rank)

            in_deg = float(spatial_index.in_degrees[u])
            out_deg = float(spatial_index.out_degrees[v])

            feat = np.array([
                dist_rank,
                dist_gap,
                dist_ratio,
                dist,
                bearing_cos,
                head_diff,
                bearing_sin,
                frac,
                seg_len,
                speed,
                step_d,
                prox_score,
                in_deg,
                out_deg,
            ], dtype=np.float32)

            cand = CandidateState(
                arc_id=int(arc_id),
                distance_meters=float(dist),
                projection_fraction=float(frac),
                heading_diff_deg=float(head_diff),
                segment_length_m=float(seg_len),
                gps_speed_mps=float(speed),
                proximity_score=float(prox_score),
                u_node=u,
                v_node=v,
            )
            cands_t.append(cand)
            all_feature_rows.append(feat)
            feature_row_to_obs.append(t)

        all_obs_candidates.append(cands_t)

    X_all = np.array(all_feature_rows, dtype=np.float32) if all_feature_rows else np.empty((0, len(ENHANCED_FEATURE_NAMES)), dtype=np.float32)
    return all_obs_candidates, X_all, feature_row_to_obs
