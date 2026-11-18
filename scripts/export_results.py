"""Publish the Logistic Regression and KNN results from completed runs."""

import argparse
from os.path import relpath
from pathlib import Path
from shutil import copyfile

import pandas as pd

from scripts.evaluate_person1 import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--evaluation-root", type=Path, default=Path("artifacts/person1"))
    parser.add_argument("--recall", type=Path, default=Path("artifacts/results/train_candidate_recall.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/results"))
    args = parser.parse_args()
    student = args.evaluation_root / "student_test"
    summary = pd.read_csv(student / "summary.csv")
    if len(summary) != len(MODELS) or set(summary["model"]) != set(MODELS):
        parser.error("Expected one student-test summary row per baseline")
    recall = pd.read_csv(args.recall, dtype={"trip_id": str})
    totals = recall[["observations", "reference_matches", "candidate_hits"]].sum()
    if not 0 <= totals.candidate_hits <= totals.reference_matches <= totals.observations or not totals.observations:
        parser.error("Training candidate recall counts are invalid")
    if not totals.reference_matches:
        recall_text = "undefined with zero weak reference matches"
    else:
        recall_text = f"{totals.candidate_hits / totals.reference_matches:.2%} ({totals.candidate_hits:,}/{totals.reference_matches:,})"

    report_tables = {}
    for partition in ("student_test", "evaluator_test"):
        folder = args.evaluation_root / partition
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
    copyfile(student / "summary.csv", output / "student_test_baseline_summary.csv")
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
            folder = args.evaluation_root / partition / model
            routes = Path(relpath(folder / "decoded_routes.csv", start=output)).as_posix()
            scores = Path(relpath(folder / "candidate_scores.csv", start=output)).as_posix()
            lines.append(f"- {partition} {model} [scores]({scores}) and [decoded routes]({routes})")
    lines += [
        "", "Student-test candidate metrics use weak labels. Route continuity measures graph connectivity. "
        "Evaluator outputs use only candidate scores and graph diagnostics.",
    ]
    (output / "baseline_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Published results in {output}")


if __name__ == "__main__":
    main()
