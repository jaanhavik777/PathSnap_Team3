"""Write candidate features for a partition or selected trips to CSV."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.data.build_dataset import build_trip_dataset
from src.data.parsers import load_trip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--partition", default="train", choices=("train", "student_test", "evaluator_test"))
    parser.add_argument("--trip", action="append", help="Trip ID, repeatable")
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--labels", action="store_true", help="Include weak labels on train only")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.labels and args.partition != "train":
        parser.error("--labels requires the train partition")

    root = args.data_root / args.partition
    trip_ids = list(dict.fromkeys(args.trip or []))
    folders = [root / trip_id for trip_id in trip_ids]
    if not folders and root.is_dir():
        folders = sorted(path.parent for path in root.glob("*/*.track") if path.stem == path.parent.name)
    if not folders:
        parser.error(f"No trips found in {root}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=args.out.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            for index, folder in enumerate(folders):
                trip = load_trip(folder, include_route=args.labels)
                if args.labels and trip["route"] is None:
                    raise ValueError(f"Missing reference route for {folder}")
                table = build_trip_dataset(
                    trip, radius_m=args.radius_m, max_candidates=args.max_candidates,
                    include_labels=args.labels,
                )
                table.to_csv(output, index=False, header=index == 0)
            output.close()
            temporary.replace(args.out)
        finally:
            output.close()
            temporary.unlink(missing_ok=True)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
