"""Export an MLP prediction map for a train/student_test trip only."""
import argparse
from pathlib import Path

import joblib

from src.data.build_dataset import build_trip_dataset
from src.data.parsers import load_trip
from src.data.person4_maps import build_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trip-folder", required=True, help="Permitted train/student_test trip folder")
    parser.add_argument("--model", default="artifacts/models/mlp/mlp_pipeline.joblib")
    parser.add_argument("--output", default="artifacts/maps/development_map.html")
    parser.add_argument("--include-reference", action="store_true",
                        help="Include ground truth for train/student_test report maps only")
    args = parser.parse_args()
    folder = Path(args.trip_folder)
    if folder.parent.name.lower() not in {"train", "student_test"}:
        raise SystemExit("Refusing a reference/development map outside train or student_test")

    trip = load_trip(folder, include_route=args.include_reference)
    bundle = joblib.load(args.model)
    candidates = build_trip_dataset(
        trip, radius_m=bundle["candidate_config"]["radius_m"],
        max_candidates=bundle["candidate_config"]["max_candidates"],
        include_labels=False,
    )
    if candidates.empty:
        raise SystemExit("No road candidates are available for this trip")
    candidates["score"] = bundle["model"].predict_proba(
        candidates[bundle["feature_columns"]]
    )[:, 1]
    ranked = candidates.sort_values(["score", "candidate_rank"], ascending=[False, True])
    predicted = ranked.drop_duplicates("observation_id").sort_values("observation_id")["arc_id"].tolist()
    reference = trip["route"]["arc_id"].tolist() if args.include_reference and trip["route"] is not None else []
    map_view = build_map(trip, predicted, reference, include_reference=args.include_reference)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    map_view.save(str(output))
    print(f"Wrote {output} for trip {trip['trip_id']} ({len(trip['track'])} GPS rows)")


if __name__ == "__main__":
    main()
