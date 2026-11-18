"""Count how often the candidate generator contains an independent weak match."""

import argparse
from pathlib import Path

import pandas as pd

from src.data.candidates import generate_candidates
from src.data.labels import match_reference_arcs
from src.data.parsers import load_trip


def evaluate_trip(trip, *, radius_m=50.0, max_candidates=8):
    matches = match_reference_arcs(trip)
    candidates = generate_candidates(trip, radius_m=radius_m, max_candidates=max_candidates)
    keys = set(candidates[["observation_id", "arc_id"]].itertuples(index=False, name=None))
    hits = sum(match in keys for match in matches.items())
    return {
        "trip_id": trip["trip_id"],
        "observations": len(trip["track"]),
        "reference_matches": len(matches),
        "candidate_hits": hits,
        "candidate_recall": hits / len(matches) if matches else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--partition", choices=("train", "student_test"), default="train")
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.data_root / args.partition
    tracks = sorted(path for path in root.glob("*/*.track") if path.stem == path.parent.name)
    if not tracks:
        parser.error(f"No trips found under {root}")
    records = []
    for path in tracks:
        trip = load_trip(path.parent)
        if trip["route"] is None:
            parser.error(f"Missing reference route for {path.parent.name}")
        records.append(evaluate_trip(trip, radius_m=args.radius_m, max_candidates=args.max_candidates))
    report = pd.DataFrame(records)
    output = args.out or Path("artifacts/results") / f"{args.partition}_candidate_recall.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    observations, matches, hits = report[["observations", "reference_matches", "candidate_hits"]].sum()
    print(f"Reference matches: {matches:,}/{observations:,} observations")
    print(f"Candidate recall: {hits / matches:.2%} ({hits:,}/{matches:,})" if matches
          else "Candidate recall is undefined with no reference matches")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
