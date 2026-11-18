"""Publish the Logistic Regression and KNN results from completed runs."""

import argparse
from os.path import relpath
from pathlib import Path
from shutil import copyfile

import pandas as pd
from sklearn.metrics import f1_score, log_loss

from scripts.evaluate_person1 import MODELS
from src.data.parsers import read_arcs

EVALUATION_ROOTS = tuple(Path("artifacts") / name for name in ("baseline_evaluation", "jaanhavi", "person1"))
KEYS = ["trip_id", "observation_id", "arc_id"]


def direct_continuity(top, arcs_by_trip):
    """Return pooled counts using the RF pre-decoder continuity definition."""
    connected = total = 0
    for trip_id, trip in top.sort_values(["trip_id", "observation_id"]).groupby("trip_id"):
        arcs = trip["arc_id"].to_numpy()
        endpoints = arcs_by_trip[trip_id].loc[arcs]
        linked = (arcs[:-1] == arcs[1:]) | (
            endpoints["to_node_id"].to_numpy()[:-1] == endpoints["from_node_id"].to_numpy()[1:]
        )
        connected += int(linked.sum())
        total += len(linked)
    return connected, total


def comparison_metrics(student, features, data_root):
    labels = pd.read_csv(features, usecols=[*KEYS, "label"], dtype={"trip_id": str})
    arcs_by_trip = {
        trip: read_arcs(data_root / "student_test" / trip / f"{trip}.arcs").set_index("arc_id")
        for trip in labels["trip_id"].unique()
    }
    records = []
    for model in MODELS:
        scores = pd.read_csv(student / model / "candidate_scores.csv", dtype={"trip_id": str})
        joined = labels.merge(scores, on=KEYS, validate="one_to_one")
        if len(joined) != len(labels) or len(joined) != len(scores):
            raise ValueError(f"Student-test score keys differ from the labeled export for {model}")
        labeled = joined.dropna(subset=["label"])
        if labeled.empty or not labeled["candidate_score"].between(0, 1).all():
            raise ValueError(f"No labeled scores or invalid probabilities for {model}")
        top = labeled.sort_values(
            ["candidate_score", "candidate_rank", "arc_id"], ascending=[False, True, True],
        ).drop_duplicates(KEYS[:2])
        connected, total = direct_continuity(top, arcs_by_trip)
        records.append({
            "model": model,
            "top1_direct_connected_pairs": connected,
            "top1_direct_total_pairs": total,
            "top1_direct_continuity": connected / total if total else 0.0,
            "candidate_macro_f1_at_0_5": f1_score(
                labeled["label"], labeled["candidate_score"].ge(0.5),
                labels=[0, 1], average="macro", zero_division=0,
            ),
            "candidate_log_loss_rf_clip": log_loss(
                labeled["label"], labeled["candidate_score"].clip(1e-7, 1 - 1e-7), labels=[0, 1],
            ),
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--recall", type=Path)
    parser.add_argument("--features", type=Path, default=Path("data/processed/student_test_candidates_labeled.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/results"))
    args = parser.parse_args()
    evaluation_root = args.evaluation_root or next(
        (root for root in EVALUATION_ROOTS if (root / "student_test/summary.csv").is_file()),
        EVALUATION_ROOTS[0],
    )
    student = evaluation_root / "student_test"
    summary = pd.read_csv(student / "summary.csv")
    if len(summary) != len(MODELS) or set(summary["model"]) != set(MODELS):
        parser.error("Expected one student-test summary row per baseline")
    recall_path = args.recall or next(
        (path for path in (
            Path("artifacts/results/train_candidate_recall.csv"),
            args.baseline_dir / "train_candidate_recall.csv",
            Path("artifacts/results/training_candidate_recall.csv"),
        ) if path.is_file()), Path("artifacts/results/train_candidate_recall.csv"),
    )
    recall = pd.read_csv(recall_path, dtype={"trip_id": str})
    totals = recall[["observations", "reference_matches", "candidate_hits"]].sum()
    if not 0 <= totals.candidate_hits <= totals.reference_matches <= totals.observations or not totals.observations:
        parser.error("Training candidate recall counts are invalid")
    if not totals.reference_matches:
        recall_text = "undefined with zero weak reference matches"
    else:
        recall_text = f"{totals.candidate_hits / totals.reference_matches:.2%} ({totals.candidate_hits:,}/{totals.reference_matches:,})"

    report_tables = {}
    for partition in ("student_test", "evaluator_test"):
        folder = evaluation_root / partition
        if partition == "evaluator_test" and not folder.exists():
            continue
        report_tables[partition] = pd.concat(
            [pd.read_csv(folder / model / "route_metrics.csv", dtype={"trip_id": str}).assign(model=model)
             for model in MODELS],
            ignore_index=True,
        )
    folds = args.baseline_dir / "fold_metrics.csv"
    if not folds.is_file():
        parser.error(f"Missing baseline folds at {folds}")

    output = args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = summary.drop(columns="candidate_macro_f1_at_0_5", errors="ignore").merge(
        comparison_metrics(student, args.features, args.data_root), on="model", validate="one_to_one",
    )
    summary.to_csv(output / "student_test_baseline_summary.csv", index=False)
    copyfile(folds, output / "baseline_cross_validation.csv")
    recall.to_csv(output / "training_candidate_recall.csv", index=False)
    table_names = {
        "student_test": "student_test_baseline_route_metrics.csv",
        "evaluator_test": "evaluator_baseline_route_diagnostics.csv",
    }
    for partition, table in report_tables.items():
        table.to_csv(output / table_names[partition], index=False)

    lines = [
        "# PathSnap baseline results", "",
        f"Training weak-label coverage was {totals.candidate_hits / totals.observations:.2%} "
        f"({totals.candidate_hits:,}/{totals.observations:,} observations).",
        f"Training candidate recall was {recall_text}.", "",
        "- [Student-test comparison](student_test_baseline_summary.csv)",
        "- [Training cross-validation](baseline_cross_validation.csv)",
        "- [Training candidate recall by trip](training_candidate_recall.csv)",
    ]
    for partition in report_tables:
        lines.append(f"- [{partition} per-trip results]({table_names[partition]})")
        for model in MODELS:
            folder = evaluation_root / partition / model
            routes = Path(relpath(folder / "decoded_routes.csv", start=output)).as_posix()
            scores = Path(relpath(folder / "candidate_scores.csv", start=output)).as_posix()
            lines.append(f"- {partition} {model} [scores]({scores}) and [decoded routes]({routes})")
    lines += [
        "", "Student-test candidate metrics use weak labels. Use top1_direct_continuity for the "
        "Random Forest comparison and mean_route_continuity for post-Viterbi diagnostics. "
        "Evaluator outputs use only candidate scores and graph diagnostics.",
    ]
    (output / "baseline_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Published results in {output}")


if __name__ == "__main__":
    main()
