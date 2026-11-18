"""
End-to-end map-matching prediction pipeline for PathSnap.

Strictly isolated from reference routes (.route files are NEVER read).
Executes:
  1. GPS track and road network loading (include_route=False)
  2. Multi-point spatial candidate generation
  3. Feature extraction matching active model schema
  4. Probabilistic candidate classification via ModelAdapter
  5. HMM sequential decoding via Viterbi algorithm
  6. Graph-based continuous route reconstruction
  7. Map geometry assembly (GeoJSON [lon, lat] and Folium [lat, lon])
"""

from __future__ import annotations
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial import cKDTree

from src.data.parsers import load_trip
from src.data.road_graph import RoadGraph
from src.data.spatial import (
    LocalProjector,
    project_point_to_segment_2d,
    calculate_bearing,
    angle_difference,
    haversine_distance,
)
from src.hmm.states import CandidateState
from src.hmm.viterbi import ViterbiDecoder
from src.hmm.decode_route import reconstruct_continuous_route, ReconstructedRoute
from src.hmm.route_metrics import (
    calculate_route_continuity,
    calculate_invalid_transition_rate,
)
from src.pipeline.model_adapter import ModelAdapter, HGB_FEATURE_NAMES


def extract_hgb_candidates(
    road_graph: RoadGraph,
    lons: np.ndarray,
    lats: np.ndarray,
    timestamps: np.ndarray,
    search_radius_m: float = 65.0,
    max_cands: int = 10,
) -> Tuple[List[List[CandidateState]], np.ndarray, List[int]]:
    """
    Generate spatial candidates and 8 features for HistGradientBoosting model.
    """
    N = len(lons)
    mean_lon = float(np.mean(lons))
    mean_lat = float(np.mean(lats))
    projector = LocalProjector(mean_lon, mean_lat)

    node_x, node_y = projector.forward_batch(
        road_graph.node_coords[:, 0], road_graph.node_coords[:, 1]
    )

    arc_sample_pts = []
    arc_indices = []

    for a in range(road_graph.num_arcs):
        u, v = road_graph.get_arc_nodes(a)
        ux, uy = node_x[u], node_y[u]
        vx, vy = node_x[v], node_y[v]
        mx, my = 0.5 * (ux + vx), 0.5 * (uy + vy)

        arc_sample_pts.append([ux, uy])
        arc_indices.append(a)
        arc_sample_pts.append([vx, vy])
        arc_indices.append(a)
        arc_sample_pts.append([mx, my])
        arc_indices.append(a)

    kd_tree = cKDTree(np.array(arc_sample_pts))
    arc_indices = np.array(arc_indices)

    obs_headings = np.zeros(N, dtype=np.float64)
    obs_speeds = np.zeros(N, dtype=np.float64)

    for i in range(N - 1):
        d = haversine_distance(lons[i], lats[i], lons[i + 1], lats[i + 1])
        dt = max(0.1, timestamps[i + 1] - timestamps[i])
        obs_speeds[i] = d / dt
        obs_headings[i] = calculate_bearing(lons[i], lats[i], lons[i + 1], lats[i + 1])
    obs_speeds[-1] = obs_speeds[-2] if N > 1 else 0.0
    obs_headings[-1] = obs_headings[-2] if N > 1 else 0.0

    gps_x, gps_y = projector.forward_batch(lons, lats)

    all_obs_candidates: List[List[CandidateState]] = []
    all_feature_rows: List[np.ndarray] = []
    feature_row_to_obs: List[int] = []

    for t in range(N):
        gx, gy = gps_x[t], gps_y[t]
        query_r = search_radius_m

        neighbor_sample_indices = kd_tree.query_ball_point([gx, gy], r=query_r)
        nearby_arcs = set(arc_indices[neighbor_sample_indices]) if len(neighbor_sample_indices) else set()

        if not nearby_arcs:
            _, nearest_sample = kd_tree.query([gx, gy], k=1)
            nearby_arcs = {arc_indices[nearest_sample]}

        cands_t = []
        for arc_id in nearby_arcs:
            u, v = road_graph.get_arc_nodes(arc_id)
            ax, ay = node_x[u], node_y[u]
            bx, by = node_x[v], node_y[v]

            px, py, dist, frac = project_point_to_segment_2d(gx, gy, ax, ay, bx, by)
            arc_bearing = road_graph.get_arc_bearing(arc_id)
            head_diff = angle_difference(obs_headings[t], arc_bearing)
            seg_len = road_graph.get_arc_length(arc_id)
            speed = obs_speeds[t]

            bearing_rad = np.radians(head_diff)
            bearing_cos = float(np.cos(bearing_rad))
            bearing_sin = float(np.sin(bearing_rad))
            prox_score = float(1.0 / (1.0 + (dist / 15.0) ** 2))

            feat = np.array([
                dist,
                head_diff,
                frac,
                seg_len,
                speed,
                bearing_cos,
                bearing_sin,
                prox_score,
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
            cands_t.append((dist, cand, feat))

        cands_t.sort(key=lambda x: x[0])
        selected_t = cands_t[:max_cands]

        cand_objs = [x[1] for x in selected_t]
        all_obs_candidates.append(cand_objs)

        for _, _, f in selected_t:
            all_feature_rows.append(f)
            feature_row_to_obs.append(t)

    X_all = np.array(all_feature_rows, dtype=np.float32)
    return all_obs_candidates, X_all, feature_row_to_obs


def extract_shared_candidates(
    trip: Dict[str, Any],
    road_graph: RoadGraph,
    feature_names: List[str],
) -> Tuple[List[List[CandidateState]], np.ndarray, List[int]]:
    """
    Generate candidates using shared candidate pipeline (src.data.build_dataset).
    """
    from src.data.build_dataset import build_trip_dataset
    dataset_df = build_trip_dataset(trip, radius_m=50.0, max_candidates=8, include_labels=False)

    obs_series = dataset_df["observation_id"].to_numpy(dtype=np.int64)
    N = int(trip["track"].shape[0])

    all_obs_candidates: List[List[CandidateState]] = [[] for _ in range(N)]
    all_feature_rows: List[np.ndarray] = []
    feature_row_to_obs: List[int] = []

    feat_cols = [c for c in feature_names if c in dataset_df.columns]
    X_mat = dataset_df[feat_cols].to_numpy(dtype=np.float32)

    for i in range(len(dataset_df)):
        obs_id = int(obs_series[i])
        arc_id = int(dataset_df["arc_id"].iloc[i])
        dist = float(dataset_df["distance_m"].iloc[i])
        frac = float(dataset_df["projection_fraction"].iloc[i])

        cand = CandidateState(
            arc_id=arc_id,
            distance_meters=dist,
            projection_fraction=frac,
        )
        if obs_id < N:
            all_obs_candidates[obs_id].append(cand)
            all_feature_rows.append(X_mat[i])
            feature_row_to_obs.append(obs_id)

    return all_obs_candidates, np.array(all_feature_rows, dtype=np.float32), feature_row_to_obs


def predict_trip(
    trip_id: str,
    data_dir: Union[str, Path],
    model_adapter: ModelAdapter,
    hmm_config: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Execute full map-matching prediction for a trip WITHOUT ground-truth access.

    Args:
        trip_id: Trip ID string (e.g. '00000000')
        data_dir: Path to directory containing trip folders
        model_adapter: Fitted ModelAdapter wrapper
        hmm_config: Optional dict of HMM tuning parameters

    Returns:
        Structured prediction result dictionary.
    """
    t_start = time.perf_counter()
    trip_folder = Path(data_dir) / trip_id
    if not trip_folder.is_dir():
        # Check subdirectories
        candidates = list(Path(data_dir).glob(f"*/{trip_id}"))
        if candidates and candidates[0].is_dir():
            trip_folder = candidates[0]
        else:
            raise FileNotFoundError(f"Trip folder '{trip_id}' not found in {data_dir}")

    # 1. Load Trip Data (include_route=False ensures ZERO ground-truth leakage)
    t0_load = time.perf_counter()
    trip_data = load_trip(trip_folder, include_route=False)
    t_load = (time.perf_counter() - t0_load) * 1000

    track_df = trip_data["track"]
    nodes_df = trip_data["nodes"]
    arcs_df = trip_data["arcs"]

    N = len(track_df)
    lons = track_df["longitude"].to_numpy(dtype=np.float64)
    lats = track_df["latitude"].to_numpy(dtype=np.float64)
    timestamps = track_df["timestamp"].to_numpy(dtype=np.float64)

    # 2. Construct RoadGraph
    t0_graph = time.perf_counter()
    node_coords = nodes_df[["longitude", "latitude"]].to_numpy(dtype=np.float64)
    arc_nodes = arcs_df[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)
    road_graph = RoadGraph(node_coords, arc_nodes, trip_id=trip_id)
    t_graph = (time.perf_counter() - t0_graph) * 1000

    # 3. Generate Candidates & Features
    t0_cand = time.perf_counter()
    use_hgb_schema = len(model_adapter.feature_names) == 8 or "hist_gradient" in model_adapter.model_name.lower()
    if use_hgb_schema:
        obs_cands, X_features, obs_indices = extract_hgb_candidates(
            road_graph, lons, lats, timestamps, search_radius_m=65.0, max_cands=10
        )
    else:
        obs_cands, X_features, obs_indices = extract_shared_candidates(
            trip_data, road_graph, model_adapter.feature_names
        )
    t_cand = (time.perf_counter() - t0_cand) * 1000

    # 4. Predict Candidate Probabilities
    t0_infer = time.perf_counter()
    cand_scores = model_adapter.predict_candidate_scores(X_features)
    t_infer = (time.perf_counter() - t0_infer) * 1000

    # Group scores by observation
    scores_per_obs: List[np.ndarray] = []
    idx = 0
    for t in range(N):
        cnt = len(obs_cands[t])
        scores_per_obs.append(cand_scores[idx : idx + cnt])
        idx += cnt

    # 5. HMM Viterbi Decoding
    t0_vit = time.perf_counter()
    cfg = hmm_config or {}
    decoder = ViterbiDecoder(
        road_graph=road_graph,
        beta_meters=cfg.get("beta_meters", 15.0),
        sigma_z_meters=cfg.get("sigma_z_meters", 12.0),
        max_speed_mps=cfg.get("max_speed_mps", 45.0),
        gap_threshold_sec=cfg.get("gap_threshold_sec", 10.0),
    )
    selected_cands, predicted_arcs, log_score, viterbi_warnings = decoder.decode(
        obs_cands, scores_per_obs, lons, lats, timestamps
    )
    t_vit = (time.perf_counter() - t0_vit) * 1000

    # 6. Reconstruct Continuous Route
    t0_recon = time.perf_counter()
    recon: ReconstructedRoute = reconstruct_continuous_route(
        observation_arc_ids=predicted_arcs,
        road_graph=road_graph,
        gps_lons=lons,
        gps_lats=lats,
    )
    t_recon = (time.perf_counter() - t0_recon) * 1000
    t_total = (time.perf_counter() - t_start) * 1000

    # 7. Route Continuity & Transition Rate
    route_continuity = calculate_route_continuity(predicted_arcs, road_graph)
    invalid_rate = calculate_invalid_transition_rate(predicted_arcs, road_graph)

    # 8. Assemble Geometry for Map Display
    # GPS points in [{lat, lon}] and [[lat, lon]]
    gps_points = [{"lat": round(float(lats[i]), 6), "lon": round(float(lons[i]), 6)} for i in range(N)]
    gps_geometry = [[round(float(lats[i]), 6), round(float(lons[i]), 6)] for i in range(N)]

    # Predicted route segments formatted for GeoJSON ([lon, lat])
    predicted_route_geojson = []
    seen_route_arcs = set()
    for aid in recon.reconstructed_arc_sequence:
        if aid not in seen_route_arcs and 0 <= aid < road_graph.num_arcs:
            seen_route_arcs.add(aid)
            u, v = road_graph.get_arc_nodes(aid)
            u_lon, u_lat = road_graph.node_coords[u, 0], road_graph.node_coords[u, 1]
            v_lon, v_lat = road_graph.node_coords[v, 0], road_graph.node_coords[v, 1]
            predicted_route_geojson.append({
                "arc_id": int(aid),
                "coordinates": [
                    [round(float(u_lon), 6), round(float(u_lat), 6)],
                    [round(float(v_lon), 6), round(float(v_lat), 6)],
                ],
            })

    # Road network nearby segments for background layer
    road_segments_geojson = []
    sample_candidate_arcs = set(aid for cands in obs_cands for cand in cands for aid in [cand.arc_id])
    for aid in list(sample_candidate_arcs)[:200]:
        if 0 <= aid < road_graph.num_arcs:
            u, v = road_graph.get_arc_nodes(aid)
            u_lon, u_lat = road_graph.node_coords[u, 0], road_graph.node_coords[u, 1]
            v_lon, v_lat = road_graph.node_coords[v, 0], road_graph.node_coords[v, 1]
            road_segments_geojson.append({
                "arc_id": int(aid),
                "coordinates": [
                    [round(float(u_lon), 6), round(float(u_lat), 6)],
                    [round(float(v_lon), 6), round(float(v_lat), 6)],
                ],
            })

    all_warnings = viterbi_warnings + recon.warnings

    return {
        "trip_id": str(trip_id),
        "status": "SUCCESS" if len(recon.warnings) == 0 else "WARNING",
        "model_name": model_adapter.model_name,
        "observation_count": int(N),
        "predicted_arc_ids": predicted_arcs,
        "reconstructed_arc_sequence": recon.reconstructed_arc_sequence,
        "route_geometry": recon.route_geometry,
        "gps_geometry": gps_geometry,
        "gps_points": gps_points,
        "predicted_route": predicted_route_geojson,
        "road_segments": road_segments_geojson,
        "route_continuity": float(route_continuity),
        "invalid_transition_rate": float(invalid_rate),
        "disconnected_jumps": int(recon.disconnected_jumps),
        "viterbi_log_score": float(log_score),
        "inference_time_ms": float(round(t_total, 2)),
        "timing": {
            "load_time_ms": float(round(t_load, 2)),
            "graph_build_ms": float(round(t_graph, 2)),
            "candidate_gen_ms": float(round(t_cand, 2)),
            "model_infer_ms": float(round(t_infer, 2)),
            "viterbi_ms": float(round(t_vit, 2)),
            "reconstruct_ms": float(round(t_recon, 2)),
            "total_ms": float(round(t_total, 2)),
        },
        "warnings": all_warnings,
    }
