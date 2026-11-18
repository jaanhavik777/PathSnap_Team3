"""Train and export the Logistic Regression and KNN candidate baselines."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data.features import FEATURE_COLUMNS
from src.models.cross_validate import cross_validate, fit_baseline
from src.models.train_knn import make_knn
from src.models.train_logistic_regression import make_logistic_regression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train_candidates.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--knn-max-fit-rows", type=int, default=25_000)
    args = parser.parse_args()
    if args.knn_max_fit_rows < 15:
        parser.error("--knn-max-fit-rows must be at least 15")

    columns = ["trip_id", "observation_id", "label", *FEATURE_COLUMNS]
    rows = pd.read_csv(args.train, usecols=columns, dtype={"trip_id": str})
    rows = rows.dropna(subset=["label"])
    if not rows["label"].isin([0, 1]).all() or rows["trip_id"].isna().any():
        parser.error("Training CSV requires binary labels and a trip ID on each labeled row")
    if args.folds < 2 or rows["trip_id"].nunique() < args.folds:
        parser.error("--folds must be between 2 and the number of labeled trips")
    labeled_groups = rows.groupby(["trip_id", "observation_id"])["label"]
    if not labeled_groups.sum().eq(1).all():
        parser.error("Expected exactly one positive per labeled observation")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    baselines = (
        ("logistic_regression", make_logistic_regression(), None),
        ("knn", make_knn(), args.knn_max_fit_rows),
    )
    for name, estimator, fit_limit in baselines:
        for result in cross_validate(estimator, rows, folds=args.folds, max_fit_rows=fit_limit):
            results.append({"model": name, **result})
            print(
                f'{name} fold {result["fold"]}: AP={result["average_precision"]:.4f}, '
                f'top-1={result["observation_top1"]:.4f}',
                flush=True,
            )
        model = fit_baseline(estimator, rows, max_fit_rows=fit_limit)
        joblib.dump(model, args.out_dir / f"{name}.joblib")
    pd.DataFrame(results).to_csv(args.out_dir / "fold_metrics.csv", index=False)


if __name__ == "__main__":
    main()
