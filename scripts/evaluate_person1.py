"""Score and decode Logistic Regression and KNN on a held-out partition."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

from scripts.decode_student_test import decode_scores, route_metrics
from src.data.features import FEATURE_COLUMNS
from src.data.parsers import load_trip, read_routes

OBSERVATION_KEY = ["trip_id", "observation_id"]
SCORE_COLUMNS = [*OBSERVATION_KEY, "arc_id", "candidate_rank", "distance_m", "projection_fraction"]
MODELS = ("logistic_regression", "knn")
INPUT_FILES = {
    "student_test": "student_test_candidates_labeled.csv",
    "evaluator_test": "evaluator_test_candidates.csv",
}


def candidate_metrics(scores):
    labeled = scores.dropna(subset=["label"])
    if labeled.empty:
        raise ValueError("Student-test evaluation needs labeled observations")
    if not labeled["label"].isin([0, 1]).all():
        raise ValueError("Candidate labels must be binary or blank")
    grouped = labeled.groupby(OBSERVATION_KEY)["label"]
    labeled_sizes = grouped.size()
    group_sizes = scores.groupby(OBSERVATION_KEY).size()
    if not grouped.sum().eq(1).all() or not labeled_sizes.eq(group_sizes.reindex(labeled_sizes.index)).all():
        raise ValueError("Labeled observations require all candidate labels and exactly one positive")

    ranked = labeled.sort_values(
        ["candidate_score", "candidate_rank", "arc_id"], ascending=[False, True, True],
    )
    top = ranked.drop_duplicates(OBSERVATION_KEY)
    predicted = labeled["candidate_score"].ge(0.5)
    return {
        "observations": len(group_sizes),
        "labeled_observations": len(top),
        "label_coverage": len(top) / len(group_sizes),
        "top1": float(top["label"].mean()),
        "candidate_f1_at_0_5": f1_score(labeled["label"], predicted, zero_division=0),
        "candidate_macro_f1_at_0_5": f1_score(labeled["label"], predicted, labels=[0, 1], average="macro", zero_division=0),
        "candidate_log_loss": log_loss(labeled["label"], labeled["candidate_score"], labels=[0, 1]),
    }


def score_candidates(model, candidates):
    input_columns = list(FEATURE_COLUMNS)
    positive = list(model.classes_).index(1)
    scores = candidates[SCORE_COLUMNS].copy()
    scores["candidate_score"] = model.predict_proba(candidates[input_columns])[:, positive]
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=INPUT_FILES, default="student_test")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    evaluate = args.partition == "student_test"
    input_path = args.features or Path("data/processed") / INPUT_FILES[args.partition]
    out_dir = args.out_dir or Path("artifacts/person1") / args.partition
    root = args.data_root / args.partition
    folders = {path.parent.name: path.parent for path in root.glob("*/*.track") if path.stem == path.parent.name}
    candidates = pd.read_csv(input_path, dtype={"trip_id": str})
    keys = candidates[[*OBSERVATION_KEY, "arc_id"]]
    if keys.isna().any().any() or keys.duplicated().any():
        parser.error("Missing or duplicate candidate keys")
    if not folders or set(keys["trip_id"]) != set(folders):
        parser.error(f"Candidate CSV must include exactly the trips under {root}")
    if evaluate and "label" not in candidates:
        parser.error("Student-test metrics require the labeled candidate CSV")
    if not evaluate and "label" in candidates:
        parser.error("Evaluator inference requires an unlabeled candidate CSV")

    scored = {}
    summary = {}
    decoded_rows = {name: [] for name in MODELS}
    route_reports = {name: [] for name in MODELS}
    for name in MODELS:
        model = joblib.load(args.model_dir / f"{name}.joblib")
        scores = score_candidates(model, candidates)
        if evaluate:
            summary[name] = candidate_metrics(scores.assign(label=candidates["label"]))
        scored[name] = scores.groupby("trip_id", sort=False)

    for trip_id, folder in sorted(folders.items()):
        trip = load_trip(folder, include_route=False)
        decoded = {}
        for name in MODELS:
            route = decode_scores(trip, scored[name].get_group(trip_id))
            predicted = route.reconstructed_arc_sequence
            decoded[name] = predicted
            decoded_rows[name].extend((trip_id, order, arc) for order, arc in enumerate(predicted))
            route_reports[name].append({
                "trip_id": trip_id,
                "route_continuity": route.continuity_ratio,
                "route_breaks": route.disconnected_jumps,
            })
        if evaluate:
            reference = read_routes(folder / f"{trip_id}.route")["arc_id"].tolist()
            for name in MODELS:
                report = route_reports[name][-1]
                report.update(route_metrics(decoded[name], reference, route_breaks=report["route_breaks"]))
        print(f"Decoded {args.partition} trip {trip_id} with both models", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for name in MODELS:
        model_dir = out_dir / name
        model_dir.mkdir(exist_ok=True)
        scored[name].obj.to_csv(model_dir / "candidate_scores.csv", index=False)
        pd.DataFrame(decoded_rows[name], columns=["trip_id", "arc_order", "arc_id"]).to_csv(
            model_dir / "decoded_routes.csv", index=False,
        )
        reports = pd.DataFrame(route_reports[name])
        reports.to_csv(model_dir / "route_metrics.csv", index=False)
        if evaluate:
            summary_rows.append({
                "model": name,
                **summary[name],
                "mean_route_continuity": float(reports["route_continuity"].mean()),
                "mean_route_arc_f1": float(reports["arc_f1"].mean()),
                "exact_route_matches": int(reports["exact_route_match"].sum()),
            })
    if evaluate:
        table = pd.DataFrame(summary_rows)
        table.to_csv(out_dir / "summary.csv", index=False)
        print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
