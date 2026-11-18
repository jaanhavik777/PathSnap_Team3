"""Train/evaluate the MLP using the shared candidate and weak-label pipeline."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.build_dataset import build_trip_dataset
from src.data.features import FEATURE_COLUMNS
from src.data.parsers import load_trip

OBSERVATION_KEY = ["trip_id", "observation_id"]


def make_model(hidden=(32,), alpha=0.001, seed=42, max_iter=40):
    """Median-impute and scale shared features before the MLP classifier."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=tuple(hidden), activation="relu", solver="adam",
            alpha=alpha, batch_size=512, learning_rate_init=0.001,
            max_iter=max_iter, early_stopping=False, random_state=seed,
        )),
    ])


def partition_frame(data_root, partition, *, radius_m=50.0, max_candidates=8):
    """Build labeled rows using the shared parser, features, and route labeler."""
    partition_root = Path(data_root) / partition
    if not partition_root.is_dir():
        raise FileNotFoundError(f"Missing permitted partition: {partition_root}")
    frames = []
    for folder in sorted(path for path in partition_root.iterdir() if path.is_dir()):
        trip = load_trip(folder, include_route=True)
        table = build_trip_dataset(
            trip, radius_m=radius_m, max_candidates=max_candidates,
            include_labels=True,
        )
        # The shared labeler leaves unresolved observations nullable. Do not
        # turn those rows into negative examples.
        table = table.dropna(subset=["label"]).copy()
        if table.empty:
            continue
        table["label"] = table["label"].astype("int8")
        frames.append(table)
    if not frames:
        raise ValueError(f"No labeled candidate observations found in {partition_root}")
    return pd.concat(frames, ignore_index=True)


def _metrics(model, rows):
    if rows.empty:
        return {"status": "unavailable_no_labeled_candidates"}
    features = rows[list(FEATURE_COLUMNS)]
    labels = rows["label"].to_numpy(dtype=int)
    predicted = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    ranked = rows[OBSERVATION_KEY + ["candidate_rank", "label"]].assign(score=probabilities)
    best = ranked.sort_values(["score", "candidate_rank"], ascending=[False, True])
    best = best.drop_duplicates(OBSERVATION_KEY)
    return {
        "candidate_accuracy": float(accuracy_score(labels, predicted)),
        "candidate_macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "log_loss": float(log_loss(labels, model.predict_proba(features), labels=[0, 1])),
        "observation_top1_accuracy": float(np.mean(best["label"].to_numpy() == 1)),
        "labeled_observations": int(len(best)),
        "candidate_rows": int(len(rows)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def grouped_cv(rows, config, *, folds=3):
    """Group whole trips into held-out folds; fit preprocessing per fold."""
    groups = rows["trip_id"]
    if groups.nunique() < folds:
        raise ValueError("Need at least one labeled trip per GroupKFold fold")
    splitter = GroupKFold(n_splits=folds)
    fold_results = []
    for fit_idx, valid_idx in splitter.split(rows, groups=groups):
        train, valid = rows.iloc[fit_idx], rows.iloc[valid_idx]
        model = make_model(**config)
        model.fit(train[list(FEATURE_COLUMNS)], train["label"])
        train_metrics = _metrics(model, train)
        valid_metrics = _metrics(model, valid)
        fold_results.append({
            "macro_f1": valid_metrics["candidate_macro_f1"],
            "train_macro_f1": train_metrics["candidate_macro_f1"],
            "top1": valid_metrics["observation_top1_accuracy"],
        })
    values = [result["macro_f1"] for result in fold_results]
    return {
        "cv_macro_f1_mean": float(np.mean(values)),
        "cv_macro_f1_std": float(np.std(values)),
        "cv_train_macro_f1_mean": float(np.mean([r["train_macro_f1"] for r in fold_results])),
        "cv_observation_top1_mean": float(np.mean([r["top1"] for r in fold_results])),
        "fold_scores": values,
    }


def run(data_root, out_root, seed=42, *, radius_m=50.0, max_candidates=8, folds=3):
    output = Path(out_root)
    output.mkdir(parents=True, exist_ok=True)
    options = {"radius_m": radius_m, "max_candidates": max_candidates}
    train = partition_frame(data_root, "train", **options)
    config = {"hidden": (32,), "alpha": 0.001, "seed": seed, "max_iter": 40}
    cv = grouped_cv(train, config, folds=folds)

    model = make_model(**config)
    started = time.perf_counter()
    model.fit(train[list(FEATURE_COLUMNS)], train["label"])
    training_time = time.perf_counter() - started

    student = partition_frame(data_root, "student_test", **options)
    started = time.perf_counter()
    student_metrics = _metrics(model, student)
    inference_time = time.perf_counter() - started

    bundle = {
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "config": config,
        "candidate_config": options,
        "seed": seed,
    }
    joblib.dump(bundle, output / "mlp_pipeline.joblib")
    summary = {
        "model": "MLPClassifier",
        "config": config,
        "candidate_config": options,
        "feature_columns": list(FEATURE_COLUMNS),
        "seed": seed,
        "training_rows": int(len(train)),
        "training_trips": int(train.trip_id.nunique()),
        "positive_rows": int(train.label.sum()),
        "cv": cv,
        "student_test": student_metrics,
        "student_trips_evaluated": int(student.trip_id.nunique()),
        "training_time_s": training_time,
        "student_inference_time_s": inference_time,
        "training_iterations": int(model.named_steps["mlp"].n_iter_),
        "training_converged_before_limit": bool(model.named_steps["mlp"].n_iter_ < config["max_iter"]),
        "training_loss_curve": [float(value) for value in model.named_steps["mlp"].loss_curve_],
        "labeling_note": "Uses src.data.build_dataset and src.data.labels; nullable unresolved observations are excluded.",
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([{
        "configuration": "32-unit ReLU, alpha=0.001",
        **cv,
        **{f"student_{key}": value for key, value in student_metrics.items()
           if not isinstance(value, (list, dict))},
        "training_time_s": training_time,
        "student_inference_time_s": inference_time,
    }]).to_csv(output / "metrics.csv", index=False)
    _plots(summary, output)
    return summary


def _plots(summary, output):
    import matplotlib.pyplot as plt

    metrics = summary["student_test"]
    plt.figure()
    plt.plot(summary["training_loss_curve"])
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.title("MLP training loss")
    plt.tight_layout()
    plt.savefig(output / "training_loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(metrics["confusion_matrix"], cmap="Blues")
    plt.xticks([0, 1], ["non-match", "match"])
    plt.yticks([0, 1], ["non-match", "match"])
    plt.xlabel("Predicted")
    plt.ylabel("Reference")
    plt.title("Student-test candidate confusion matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close()

    values = [summary["cv"]["cv_macro_f1_mean"], metrics["candidate_macro_f1"],
              metrics["observation_top1_accuracy"]]
    plt.figure()
    plt.bar(["Grouped CV macro F1", "Student candidate macro F1", "Student observation Top-1"],
            values, color=["#5969d9", "#11a579", "#f2a541"])
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output / "performance.png", dpi=150)
    plt.close()
