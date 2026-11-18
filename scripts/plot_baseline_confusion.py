"""Export aggregate confusion matrices for weakly labeled student-test candidates."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from scripts.export_results import EVALUATION_ROOTS, KEYS
from scripts.evaluate_person1 import MODELS


def candidate_confusion(labels, scores):
    joined = labels.merge(scores, on=KEYS, validate="one_to_one")
    if len(joined) != len(labels) or len(joined) != len(scores):
        raise ValueError("Score keys differ from the student-test feature export")
    labeled = joined.dropna(subset=["label"])
    if labeled.empty or not labeled["label"].isin([0, 1]).all():
        raise ValueError("Expected binary weak labels on student-test candidates")
    if not labeled["candidate_score"].between(0, 1).all():
        raise ValueError("Candidate probabilities must be between zero and one")
    return confusion_matrix(labeled["label"], labeled["candidate_score"].ge(0.5), labels=[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/processed/student_test_candidates_labeled.csv"))
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--figure", type=Path, default=Path("reports/figures/baselines/student_test_confusion_matrices.png"))
    parser.add_argument("--counts", type=Path, default=Path("reports/data/student_test_confusion_counts.csv"))
    args = parser.parse_args()

    root = args.evaluation_root or next(
        (path for path in EVALUATION_ROOTS if (path / "student_test/summary.csv").is_file()),
        EVALUATION_ROOTS[0],
    )
    labels = pd.read_csv(args.features, usecols=[*KEYS, "label"], dtype={"trip_id": str})
    names = ("Logistic Regression", "KNN")
    matrices = []
    for model in MODELS:
        path = root / "student_test" / model / "candidate_scores.csv"
        scores = pd.read_csv(path, usecols=[*KEYS, "candidate_score"], dtype={"trip_id": str})
        matrices.append(candidate_confusion(labels, scores))

    figure, axes = plt.subplots(1, 2, figsize=(8.5, 4.5), layout="constrained")
    vmax = max(matrix.max() for matrix in matrices)
    rows = []
    for axis, model, name, matrix in zip(axes, MODELS, names, matrices):
        display = ConfusionMatrixDisplay(matrix, display_labels=["Negative", "Positive"])
        display.plot(ax=axis, cmap="Blues", colorbar=False, values_format=",d", im_kw={"vmin": 0, "vmax": vmax})
        axis.set_title(name)
        tn, fp, fn, tp = matrix.ravel()
        rows.append({"model": model, "threshold": 0.5, "true_negative": tn,
                     "false_positive": fp, "false_negative": fn, "true_positive": tp})
    figure.suptitle("Student-test candidate confusion matrices")
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.counts.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=170, facecolor="white")
    plt.close(figure)
    pd.DataFrame(rows).to_csv(args.counts, index=False)
    print(f"Saved {args.figure} and {args.counts}")


if __name__ == "__main__":
    main()
