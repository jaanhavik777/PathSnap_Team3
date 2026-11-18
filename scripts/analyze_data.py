"""Summarize GPS quality and directed road graphs for the two development partitions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.data.parsers import load_trip


def analyze_trip(trip, partition):
    track, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    starts = arcs["from_node_id"].to_numpy()
    ends = arcs["to_node_id"].to_numpy()
    graph = coo_matrix(
        (np.ones(len(arcs), dtype=bool), (starts, ends)), shape=(len(nodes), len(nodes)),
    ).tocsr()
    weak_count, component = connected_components(graph, directed=True, connection="weak")
    strong_count = connected_components(graph, directed=True, connection="strong", return_labels=False)
    largest = np.bincount(component).max(initial=0)
    delta = track["timestamp"].diff().dropna()
    route = trip["route"]["arc_id"].to_numpy() if trip["route"] is not None else np.empty(0, dtype=int)
    changes = route[1:] != route[:-1]
    breaks = changes & (ends[route[:-1]] != starts[route[1:]])
    return {
        "partition": partition, "trip_id": trip["trip_id"],
        "observations": len(track), "nodes": len(nodes), "arcs": len(arcs),
        "missing_gps": int(track.isna().any(axis=1).sum()),
        "nonpositive_steps": int(delta.le(0).sum()), "gaps_over_5s": int(delta.gt(5).sum()),
        "gaps_over_60s": int(delta.gt(60).sum()),
        "max_gap_s": float(delta.max()) if len(delta) else 0.0,
        "isolated_nodes": len(nodes) - len(np.union1d(starts, ends)),
        "weak_components": weak_count, "strong_components": strong_count,
        "largest_weak_component_fraction": largest / len(nodes) if len(nodes) else np.nan,
        "reference_transitions": max(0, len(route) - 1), "reference_breaks": int(breaks.sum()),
        "min_lon": track.longitude.min(), "max_lon": track.longitude.max(),
        "min_lat": track.latitude.min(), "max_lat": track.latitude.max(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/results/dataset_eda.csv"))
    args = parser.parse_args()
    records = []
    for partition in ("train", "student_test"):
        root = args.data_root / partition
        tracks = sorted(path for path in root.glob("*/*.track") if path.stem == path.parent.name)
        if not tracks:
            parser.error(f"No trips found under {root}")
        records.extend(analyze_trip(load_trip(path.parent), partition) for path in tracks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.out, index=False)
    print(f"Saved {len(records)} trip summaries to {args.out}")


if __name__ == "__main__":
    main()
