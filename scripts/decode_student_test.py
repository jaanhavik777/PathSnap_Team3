"""Decode student-test candidate scores and compare routes with references."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.parsers import load_trip, read_routes
from src.data.road_graph import RoadGraph
from src.hmm import CandidateState, reconstruct_continuous_route, viterbi_decode


def decode_scores(trip, scores):
    """Adapt the scorer's CSV rows to the team's Viterbi decoder."""
    track = trip["track"]
    ordered = scores.sort_values(["observation_id", "candidate_rank", "arc_id"])
    groups = list(ordered.groupby("observation_id", sort=False, dropna=False))
    if [obs for obs, _ in groups] != list(range(len(track))):
        raise ValueError("Expected candidates for every observation in the trip")
    if ordered.duplicated(["observation_id", "arc_id"]).any():
        raise ValueError("Duplicate observation and arc keys")
    for column in ("candidate_score", "projection_fraction"):
        if not ordered[column].between(0, 1).all():
            raise ValueError(f"Expected {column} in [0, 1]")
    if not np.isfinite(ordered["distance_m"]).all() or (ordered["distance_m"] < 0).any():
        raise ValueError("Expected finite, nonnegative distances")
    if not ordered["arc_id"].isin(trip["arcs"]["arc_id"]).all():
        raise ValueError("Candidate arc ID is absent from the road graph")

    graph = RoadGraph(
        trip["nodes"][["longitude", "latitude"]].to_numpy(),
        trip["arcs"][["from_node_id", "to_node_id"]].to_numpy(),
        trip_id=trip["trip_id"],
    )
    states = [
        [CandidateState(int(arc), distance, fraction) for arc, distance, fraction in
         frame[["arc_id", "distance_m", "projection_fraction"]].itertuples(index=False, name=None)]
        for _, frame in groups
    ]
    probabilities = [frame["candidate_score"].to_numpy() for _, frame in groups]
    lons = track["longitude"].to_numpy()
    lats = track["latitude"].to_numpy()
    _, selected, _, _ = viterbi_decode(
        graph, states, probabilities, lons, lats, track["timestamp"].to_numpy(),
    )
    return reconstruct_continuous_route(selected, graph, lons, lats)


def route_metrics(predicted, reference, *, route_breaks):
    if not reference:
        raise ValueError("Route metrics require a nonempty reference")
    overlap = sum((Counter(predicted) & Counter(reference)).values())
    return {
        "predicted_arcs": len(predicted),
        "reference_arcs": len(reference),
        "matched_arc_occurrences": overlap,
        "arc_precision": overlap / len(predicted) if predicted else 0.0,
        "arc_recall": overlap / len(reference),
        "arc_f1": 2 * overlap / (len(predicted) + len(reference)),
        "exact_route_match": route_breaks == 0 and predicted == reference,
        "route_breaks": route_breaks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("artifacts/baselines/student_test_scores.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/route_eval"))
    args = parser.parse_args()
    root = args.data_root / "student_test"
    folders = {path.parent.name: path.parent for path in root.glob("*/*.track") if path.stem == path.parent.name}
    scores = pd.read_csv(args.scores, dtype={"trip_id": str})
    if not folders or scores["trip_id"].isna().any() or set(scores["trip_id"]) != set(folders):
        parser.error("Score CSV must include exactly the student-test trips under --data-root")

    routes, metrics = [], []
    for trip_id, rows in scores.groupby("trip_id", sort=True):
        folder = folders[trip_id]
        trip = load_trip(folder, include_route=False)
        decoded = decode_scores(trip, rows)
        reference = read_routes(folder / f"{trip_id}.route")["arc_id"].tolist()
        predicted = decoded.reconstructed_arc_sequence
        metrics.append({"trip_id": trip_id, **route_metrics(
            predicted, reference, route_breaks=decoded.disconnected_jumps,
        )})
        routes.extend((trip_id, position, arc) for position, arc in enumerate(predicted))
        print(f"Decoded {trip_id}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(routes, columns=["trip_id", "arc_order", "arc_id"]).to_csv(
        args.out_dir / "decoded_routes.csv", index=False,
    )
    pd.DataFrame(metrics).to_csv(args.out_dir / "route_metrics.csv", index=False)


if __name__ == "__main__":
    main()
