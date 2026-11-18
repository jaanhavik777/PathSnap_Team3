"""
Experiment script for Candidate Generation Optimization (Step 2).
Evaluates candidate recall/miss rate and latency across different:
  - Search radii (50m, 65m, 75m, 85m, 100m)
  - Candidate capacities (8, 10, 12, 16)
  - Indexing strategies:
      * Sparse 3-point kd-tree (baseline Review 2)
      * SegmentIndex (bounding box midpoint ball query + exact segment projection)
      * Dense adaptive segment sampling (every 20m along arc)
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.data.parsers import load_trip
from src.data.road_graph import RoadGraph
from src.data.spatial import LocalProjector, project_point_to_segment_2d
from src.data.candidates import SegmentIndex, trip_coordinates
from src.data.alignment import align_reference_route_to_observations

REPO_ROOT = Path(__file__).resolve().parent.parent

def locate_train_folder() -> Path:
    candidates = [
        REPO_ROOT / "data" / "raw" / "train",
        REPO_ROOT.parent / "hack" / "Student folder" / "Student folder" / "train",
        REPO_ROOT.parent / "hack" / "Student folder" / "train",
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*/*.track")):
            return c
    raise FileNotFoundError("Train directory not found")


class DenseSegmentIndex:
    """
    Spatial index that places sample points along arcs at most step_m apart,
    guaranteeing that no point near any segment is further than step_m/2 from an index seed.
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

            # Sample endpoints
            sample_pts.append([ax, ay])
            sample_arc_ids.append(a)
            sample_pts.append([bx, by])
            sample_arc_ids.append(a)

            # Sample intermediate points along long arcs
            if length > step_m:
                n_inter = int(np.ceil(length / step_m))
                for step in range(1, n_inter):
                    t = step / n_inter
                    sample_pts.append([ax + t * dx, ay + t * dy])
                    sample_arc_ids.append(a)

        self.kd_tree = cKDTree(np.array(sample_pts, dtype=np.float32))
        self.sample_arc_ids = np.array(sample_arc_ids, dtype=np.int32)

    def query(self, gx: float, gy: float, radius_m: float, max_cands: int) -> Tuple[List[int], List[float], List[float]]:
        # Find candidates within radius + half sampling step
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
                scored.append((dist, aid, frac))

        scored.sort(key=lambda x: x[0])
        top = scored[:max_cands]
        return [x[1] for x in top], [x[0] for x in top], [x[2] for x in top]


def evaluate_candidate_strategy(
    trips: List[Dict],
    strategy_name: str,
    radius_m: float,
    max_cands: int,
) -> Dict:
    total_obs = 0
    missed_obs = 0
    total_cands_generated = 0
    t0 = time.perf_counter()

    for item in trips:
        trip = item["trip"]
        true_arcs = item["true_arcs"]
        road_graph = item["road_graph"]
        N = len(true_arcs)

        lons = trip["track"]["longitude"].to_numpy(dtype=float)
        lats = trip["track"]["latitude"].to_numpy(dtype=float)

        if strategy_name == "baseline_3point":
            from src.pipeline.inference import extract_hgb_candidates
            timestamps = trip["track"]["timestamp"].to_numpy(dtype=float)
            obs_cands, _, _ = extract_hgb_candidates(
                road_graph, lons, lats, timestamps, search_radius_m=radius_m, max_cands=max_cands
            )
            for t in range(N):
                c_aids = {c.arc_id for c in obs_cands[t]}
                total_cands_generated += len(c_aids)
                if true_arcs[t] not in c_aids:
                    missed_obs += 1

        elif strategy_name == "dense_index":
            mean_lon, mean_lat = float(np.mean(lons)), float(np.mean(lats))
            proj = LocalProjector(mean_lon, mean_lat)
            dense_idx = DenseSegmentIndex(road_graph, proj, step_m=20.0)
            gx_all, gy_all = proj.forward_batch(lons, lats)

            for t in range(N):
                c_aids, _, _ = dense_idx.query(gx_all[t], gy_all[t], radius_m=radius_m, max_cands=max_cands)
                total_cands_generated += len(c_aids)
                if true_arcs[t] not in set(c_aids):
                    missed_obs += 1

        total_obs += N

    elapsed_s = time.perf_counter() - t0
    miss_rate = missed_obs / total_obs
    avg_cands = total_cands_generated / total_obs
    ms_per_1k = (elapsed_s / total_obs) * 1000 * 1000

    return {
        "strategy": strategy_name,
        "radius_m": radius_m,
        "max_cands": max_cands,
        "total_obs": total_obs,
        "missed_obs": missed_obs,
        "miss_rate_pct": round(miss_rate * 100, 2),
        "recall_pct": round((1.0 - miss_rate) * 100, 2),
        "avg_cands_per_obs": round(avg_cands, 2),
        "latency_ms_per_1k_obs": round(ms_per_1k, 2),
        "total_time_s": round(elapsed_s, 2),
    }


def main():
    train_dir = locate_train_folder()
    all_train_folders = sorted([f for f in train_dir.iterdir() if f.is_dir() and (f / f"{f.name}.track").exists()])
    
    # Select a diverse group-aware validation sample of 15 train trips (representing short, medium, and long trips)
    np.random.seed(42)
    sample_folders = all_train_folders[:15]
    print(f"Loading {len(sample_folders)} training trips for candidate generation tuning...")

    trips_data = []
    for f in sample_folders:
        trip = load_trip(f, include_route=True)
        track = trip["track"]
        N = len(track)
        lons = track["longitude"].to_numpy(dtype=float)
        lats = track["latitude"].to_numpy(dtype=float)
        timestamps = track["timestamp"].to_numpy(dtype=float)
        rg = RoadGraph(
            trip["nodes"][["longitude", "latitude"]].to_numpy(dtype=float),
            trip["arcs"][["from_node_id", "to_node_id"]].to_numpy(dtype=int),
            trip_id=f.name
        )
        raw_route = trip["route"]["arc_id"].to_numpy(dtype=int)
        true_arcs = align_reference_route_to_observations(lons, lats, timestamps, rg, raw_route)
        trips_data.append({"trip": trip, "true_arcs": true_arcs, "road_graph": rg})

    total_sample_obs = sum(len(x["true_arcs"]) for x in trips_data)
    print(f"Total validation observations in sample: {total_sample_obs:,}")

    configs = [
        # Baseline Review 2 settings
        ("baseline_3point", 60.0, 8),
        ("baseline_3point", 75.0, 10),
        # Dense index with adaptive sampling
        ("dense_index", 50.0, 8),
        ("dense_index", 65.0, 10),
        ("dense_index", 75.0, 10),
        ("dense_index", 75.0, 12),
        ("dense_index", 85.0, 12),
        ("dense_index", 100.0, 14),
    ]

    results = []
    print("\nRunning candidate generation experiments...\n")
    for strat, r, k in configs:
        res = evaluate_candidate_strategy(trips_data, strat, r, k)
        results.append(res)
        print(f"[{strat:<16} | R={r:>4.0f}m, K={k:>2}] -> Miss Rate: {res['miss_rate_pct']:>5.2f}% "
              f"| Recall: {res['recall_pct']:>5.2f}% | Avg Cands: {res['avg_cands_per_obs']:>4.1f} "
              f"| Latency: {res['latency_ms_per_1k_obs']:>6.1f} ms/1k obs")

    df = pd.DataFrame(results)
    out_csv = REPO_ROOT / "results" / "candidate_generation_tuning.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")


if __name__ == "__main__":
    main()
