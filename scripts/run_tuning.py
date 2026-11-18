"""Tune the Logistic Regression and KNN baselines using training trips only."""

import argparse
import json
from pathlib import Path
from statistics import mean

import joblib
import optuna

from scripts.train_all import read_training_rows
from src.models.cross_validate import cross_validate, fit_baseline
from src.models.train_knn import make_knn
from src.models.train_logistic_regression import make_logistic_regression


def logistic_trial(trial):
    return make_logistic_regression().set_params(
        logisticregression__C=trial.suggest_float("C", 0.001, 100, log=True),
    )


def knn_trial(trial):
    return make_knn().set_params(
        kneighborsclassifier__n_neighbors=trial.suggest_int("n_neighbors", 3, 31, step=2),
        kneighborsclassifier__weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
    )


def tune_model(rows, make_model, baseline, *, folds, trials, seed, max_fit_rows=None):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(baseline)

    def objective(trial):
        results = list(cross_validate(make_model(trial), rows, folds=folds, max_fit_rows=max_fit_rows))
        trial.set_user_attr("fold_metrics", results)
        return mean(result["observation_top1"] for result in results)

    study.optimize(objective, n_trials=trials)
    best = study.best_trial
    model = fit_baseline(
        make_model(optuna.trial.FixedTrial(best.params)), rows, max_fit_rows=max_fit_rows,
    )
    return model, study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train_candidates.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/tuned_baselines"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=8, help="Trials per model, including its baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-max-fit-rows", type=int, default=25_000)
    args = parser.parse_args()
    if args.trials < 1 or args.knn_max_fit_rows < 15:
        parser.error("Use at least one trial and at least 15 KNN fit rows")
    try:
        rows = read_training_rows(args.train, args.folds)
    except ValueError as error:
        parser.error(str(error))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = (
        ("logistic_regression", logistic_trial, {"C": 1.0}, None),
        ("knn", knn_trial, {"n_neighbors": 15, "weights": "distance"}, args.knn_max_fit_rows),
    )
    for name, make_model, baseline, limit in models:
        model, study = tune_model(
            rows, make_model, baseline, folds=args.folds, trials=args.trials,
            seed=args.seed, max_fit_rows=limit,
        )
        joblib.dump(model, args.out_dir / f"{name}.joblib")
        study.trials_dataframe().to_csv(args.out_dir / f"{name}_trials.csv", index=False)
        best = study.best_trial
        metadata = {
            "objective": "mean_training_group_cv_observation_top1",
            "folds": args.folds,
            "seed": args.seed,
            "max_fit_rows": limit,
            "trial": best.number,
            "parameters": best.params,
            "mean_top1": best.value,
            "fold_metrics": best.user_attrs["fold_metrics"],
        }
        (args.out_dir / f"{name}_best.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
