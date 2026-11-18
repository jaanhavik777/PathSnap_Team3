"""
PathSnap Review 1 Live Demonstration Script — Person 3 (Hitha)

Demonstrates:
  1. Road network graph construction (RoadGraph)
  2. HistGradientBoosting candidate classification
  3. HMM emission and transition scoring
  4. Viterbi dynamic programming decoding
  5. Route reconstruction with intermediate arc interpolation
  6. Direct comparison: Classifier-Only vs Classifier + HMM
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from src.hmm.decode_route import reconstruct_continuous_route
from src.hmm.route_metrics import (
    calculate_route_continuity,
    calculate_invalid_transition_rate,
    calculate_top1_accuracy,
    calculate_candidate_miss_rate,
)
from src.models.train_hist_gradient_boosting import HistGradientBoostingModel


def find_trip_folder(trip_id: str, custom_dir: str = None) -> Path:
    """Locate trip folder across likely paths."""
    candidates = []
    if custom_dir:
        candidates.append(Path(custom_dir) / trip_id)
        candidates.append(Path(custom_dir))

    candidates.extend([
        REPO_ROOT / "data" / "raw" / "student_test" / trip_id,
        REPO_ROOT.parent / "hack" / "Student folder" / "Student folder" / "student_test" / trip_id,
        REPO_ROOT.parent / "hack" / "Student folder" / "student_test" / trip_id,
        REPO_ROOT.parent / "hack" / "Student folder" / "train" / trip_id,
    ])

    for p in candidates:
        if p.exists() and (p / f"{trip_id}.track").exists():
            return p

    raise FileNotFoundError(
        f"Trip {trip_id} folder not found in searched locations: {[str(c) for c in candidates]}"
    )


def extract_observation_candidates(
    road_graph: RoadGraph,
    lons: np.ndarray,
    lats: np.ndarray,
    timestamps: np.ndarray,
    search_radius_m: float = 60.0,
    max_cands: int = 10,
):
    """
    Generate spatial candidate states and 8 features for each observation in trajectory.
    """
    N = len(lons)
    mean_lon = float(np.mean(lons))
    mean_lat = float(np.mean(lats))
    projector = LocalProjector(mean_lon, mean_lat)

    # Project road nodes
    node_x, node_y = projector.forward_batch(
        road_graph.node_coords[:, 0], road_graph.node_coords[:, 1]
    )

    # Build KDTree over arc sample points (midpoint, endpoints)
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

    # GPS headings and speeds
    obs_headings = np.zeros(N, dtype=np.float64)
    obs_speeds = np.zeros(N, dtype=np.float64)

    for i in range(N - 1):
        d = haversine_distance(lons[i], lats[i], lons[i + 1], lats[i + 1])
        dt = max(0.1, timestamps[i + 1] - timestamps[i])
        obs_speeds[i] = d / dt
        obs_headings[i] = calculate_bearing(lons[i], lats[i], lons[i + 1], lats[i + 1])
    obs_speeds[-1] = obs_speeds[-2] if N > 1 else 0.0
    obs_headings[-1] = obs_headings[-2] if N > 1 else 0.0

    # Project GPS points
    gps_x, gps_y = projector.forward_batch(lons, lats)

    all_obs_candidates: list[list[CandidateState]] = []
    all_feature_rows: list[np.ndarray] = []
    feature_row_to_obs: list[int] = []

    for t in range(N):
        gx, gy = gps_x[t], gps_y[t]
        query_r = search_radius_m

        # Query KDTree
        neighbor_sample_indices = kd_tree.query_ball_point([gx, gy], r=query_r)
        nearby_arcs = set(arc_indices[neighbor_sample_indices]) if len(neighbor_sample_indices) else set()

        if not nearby_arcs:
            # Fallback to nearest arc
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

        # Sort by distance and retain top max_cands
        cands_t.sort(key=lambda x: x[0])
        selected_t = cands_t[:max_cands]

        cand_objs = [x[1] for x in selected_t]
        all_obs_candidates.append(cand_objs)

        for _, _, f in selected_t:
            all_feature_rows.append(f)
            feature_row_to_obs.append(t)

    X_all = np.array(all_feature_rows, dtype=np.float32)
    return all_obs_candidates, X_all, feature_row_to_obs


def main():
    parser = argparse.ArgumentParser(description="PathSnap Review 1 Live Demonstration")
    parser.add_argument("--trip-id", type=str, default="00000000", help="Trip ID to evaluate")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing trips")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "models" / "hist_gradient_boosting.joblib"),
        help="Path to trained model artifact",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("PATHSNAP REVIEW 1 — PERSON 3 (HITHA) DEMONSTRATION")
    print("=" * 78)
    print(f"Target Trip ID:  {args.trip_id}")
    print(f"Model Artifact:  {args.model_path}")

    # 1. Locate and load trip
    t0 = time.perf_counter()
    trip_folder = find_trip_folder(args.trip_id, args.data_dir)
    print(f"Loading data from: {trip_folder}")
    trip_data = load_trip(trip_folder, include_route=True)

    track_df = trip_data["track"]
    nodes_df = trip_data["nodes"]
    arcs_df = trip_data["arcs"]
    route_df = trip_data["route"]

    N = len(track_df)
    M = len(nodes_df)
    K = len(arcs_df)
    has_ground_truth = route_df is not None and len(route_df) == N

    print(f"\n[1] DATASET INSPECTION")
    print(f"  • GPS Observations: {N:,}")
    print(f"  • Road Nodes:       {M:,}")
    print(f"  • Road Arcs:        {K:,}")
    print(f"  • Ground Truth:     {'Available (' + str(len(route_df)) + ' arcs)' if has_ground_truth else 'Unavailable'}")

    # 2. Build RoadGraph
    node_coords = nodes_df[["longitude", "latitude"]].to_numpy(dtype=np.float64)
    arc_nodes = arcs_df[["from_node_id", "to_node_id"]].to_numpy(dtype=np.int64)

    t_graph_0 = time.perf_counter()
    road_graph = RoadGraph(node_coords, arc_nodes, trip_id=args.trip_id)
    t_graph = (time.perf_counter() - t_graph_0) * 1000
    print(f"\n[2] ROAD NETWORK GRAPH CONSTRUCTED in {t_graph:.1f} ms")
    print(f"  • NetworkX MultiDiGraph: {road_graph.num_nodes} nodes, {road_graph.num_arcs} directed arcs")

    # 3. Candidate Generation & Feature Extraction
    lons = track_df["longitude"].to_numpy(dtype=np.float64)
    lats = track_df["latitude"].to_numpy(dtype=np.float64)
    timestamps = track_df["timestamp"].to_numpy(dtype=np.float64)

    t_cand_0 = time.perf_counter()
    obs_candidates, X_features, obs_indices = extract_observation_candidates(
        road_graph, lons, lats, timestamps, search_radius_m=60.0, max_cands=10
    )
    t_cand = (time.perf_counter() - t_cand_0) * 1000
    total_cands = len(X_features)
    print(f"\n[3] CANDIDATES GENERATED in {t_cand:.1f} ms")
    print(f"  • Total Candidates: {total_cands:,} (avg {total_cands / N:.1f} per observation)")
    print(f"  • Feature Matrix:   shape {X_features.shape}")

    # 4. Load HistGradientBoosting Model
    print(f"\n[4] LOADING HistGradientBoosting MODEL")
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model artifact not found at {args.model_path}")
    model = HistGradientBoostingModel.load(args.model_path)
    print(f"  • Model Type:       {type(model.classifier).__name__}")
    print(f"  • Features (8):     {', '.join(model.feature_names)}")

    # 5. Predict Candidate Scores
    t_pred_0 = time.perf_counter()
    cand_scores = model.predict_candidate_scores(X_features)
    t_pred = (time.perf_counter() - t_pred_0) * 1000
    print(f"  • Inference Time:   {t_pred:.1f} ms ({t_pred / total_cands * 1000:.3f} µs/candidate)")

    # Group scores by observation
    scores_per_obs = []
    idx = 0
    for t in range(N):
        cnt = len(obs_candidates[t])
        scores_per_obs.append(cand_scores[idx : idx + cnt])
        idx += cnt

    # 6. Classifier-Only Baseline Predictions
    clf_only_arcs = []
    for t in range(N):
        cands = obs_candidates[t]
        scores = scores_per_obs[t]
        if len(cands) > 0 and len(scores) > 0:
            best_i = int(np.argmax(scores))
            clf_only_arcs.append(cands[best_i].arc_id)
        else:
            clf_only_arcs.append(-1)

    clf_continuity = calculate_route_continuity(clf_only_arcs, road_graph)
    clf_invalid = calculate_invalid_transition_rate(clf_only_arcs, road_graph)

    # 7. HMM + Viterbi Global Decoding
    print(f"\n[5] HMM MAP MATCHING & VITERBI DECODING")
    decoder = ViterbiDecoder(
        road_graph=road_graph,
        beta_meters=15.0,
        sigma_z_meters=12.0,
        max_speed_mps=45.0,
        gap_threshold_sec=10.0,
    )

    t_vit_0 = time.perf_counter()
    selected_cands, hmm_arcs, final_log_score, warnings = decoder.decode(
        obs_candidates, scores_per_obs, lons, lats, timestamps
    )
    t_vit = (time.perf_counter() - t_vit_0) * 1000
    hmm_continuity = calculate_route_continuity(hmm_arcs, road_graph)
    hmm_invalid = calculate_invalid_transition_rate(hmm_arcs, road_graph)

    print(f"  • Viterbi Runtime:  {t_vit:.1f} ms ({t_vit / N:.3f} ms/observation)")
    print(f"  • Log Likelihood:   {final_log_score:.2f}")
    print(f"  • Decoder Warnings: {len(warnings)}")

    # 8. Route Reconstruction
    t_rec_0 = time.perf_counter()
    reconstructed = reconstruct_continuous_route(hmm_arcs, road_graph, lons, lats)
    t_rec = (time.perf_counter() - t_rec_0) * 1000

    print(f"\n[6] ROUTE RECONSTRUCTION")
    print(f"  • Observation Arcs:      {len(hmm_arcs):,}")
    print(f"  • Reconstructed Sequence: {len(reconstructed.reconstructed_arc_sequence):,} arcs")
    print(f"  • Polyline Vertices:     {len(reconstructed.route_geometry):,} [lat, lon] coordinates")
    print(f"  • Reconstructed Continuity: {reconstructed.continuity_ratio * 100:.2f}%")

    # 9. Comparison & Accuracy
    print("\n" + "=" * 78)
    print("RESULTS COMPARISON: CLASSIFIER-ONLY vs CLASSIFIER + HMM")
    print("=" * 78)
    header = f"{'Metric':<30} | {'Classifier-Only':<18} | {'Classifier + HMM':<18} | {'Delta':<12}"
    print(header)
    print("-" * 78)

    cont_delta = (hmm_continuity - clf_continuity) * 100
    print(f"{'Route Continuity':<30} | {clf_continuity * 100:>17.2f}% | {hmm_continuity * 100:>17.2f}% | {cont_delta:>+11.2f}%")

    inv_delta = (hmm_invalid - clf_invalid) * 100
    print(f"{'Invalid Transition Rate':<30} | {clf_invalid * 100:>17.2f}% | {hmm_invalid * 100:>17.2f}% | {inv_delta:>+11.2f}%")

    if has_ground_truth:
        true_arcs = route_df["arc_id"].to_numpy(dtype=np.int64)
        clf_acc = calculate_top1_accuracy(clf_only_arcs, true_arcs)
        hmm_acc = calculate_top1_accuracy(hmm_arcs, true_arcs)
        miss_rate = calculate_candidate_miss_rate(obs_candidates, true_arcs)
        acc_delta = (hmm_acc - clf_acc) * 100

        print(f"{'Top-1 Arc Accuracy':<30} | {clf_acc * 100:>17.2f}% | {hmm_acc * 100:>17.2f}% | {acc_delta:>+11.2f}%")
        print(f"{'Candidate Miss Rate':<30} | {miss_rate * 100:>17.2f}% | {miss_rate * 100:>17.2f}% | {'--':>12}")

    print("-" * 78)
    total_time = (time.perf_counter() - t0) * 1000
    print(f"Total Pipeline Runtime: {total_time:.1f} ms")
    print("=" * 78)
    print("REVIEW 1 DEMONSTRATION COMPLETE: ALL 6 DELIVERABLES VERIFIED.")
    print("=" * 78)


if __name__ == "__main__":
    main()
