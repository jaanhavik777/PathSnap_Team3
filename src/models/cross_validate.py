"""Evaluate candidate models on held-out training trips."""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, train_test_split

from src.data.features import FEATURE_COLUMNS

OBSERVATION_KEY = ["trip_id", "observation_id"]


def fit_baseline(estimator, rows, *, max_fit_rows=None):
    """Fit preprocessing and model on the supplied training rows only."""
    labels = rows["label"]
    if labels.nunique() != 2:
        raise ValueError("Training rows must contain both labels")
    if max_fit_rows is not None and len(rows) > max_fit_rows:
        rows, _ = train_test_split(
            rows, train_size=max_fit_rows, stratify=labels, random_state=42,
        )
    model = clone(estimator)
    neighbors = model.get_params().get("kneighborsclassifier__n_neighbors")
    if neighbors is not None and neighbors > len(rows):
        model.set_params(kneighborsclassifier__n_neighbors=len(rows))
    return model.fit(rows[list(FEATURE_COLUMNS)], rows["label"])


def cross_validate(estimator, rows, *, folds=5, max_fit_rows=None):
    """Return per-fold weak-label average precision and observation top-1 accuracy."""
    if folds < 2 or rows["trip_id"].nunique() < folds:
        raise ValueError("Cross-validation requires at least as many labeled trips as folds")
    if max_fit_rows is not None and max_fit_rows < 15:
        raise ValueError("max_fit_rows must be at least 15")
    splitter = GroupKFold(n_splits=folds)
    for fold, (fit_idx, validation_idx) in enumerate(
        splitter.split(rows, groups=rows["trip_id"]), start=1,
    ):
        train = rows.iloc[fit_idx]
        validation = rows.iloc[validation_idx]
        model = fit_baseline(estimator, train, max_fit_rows=max_fit_rows)
        scores = model.predict_proba(validation[list(FEATURE_COLUMNS)])[:, 1]
        ranked = validation[OBSERVATION_KEY + ["candidate_rank", "label"]].assign(score=scores)
        best = ranked.sort_values(["score", "candidate_rank"], ascending=[False, True])
        best = best.drop_duplicates(OBSERVATION_KEY)
        yield {
            "fold": fold,
            "fit_rows": len(train) if max_fit_rows is None else min(len(train), max_fit_rows),
            "validation_trips": validation["trip_id"].nunique(),
            "validation_observations": len(best),
            "average_precision": float(average_precision_score(validation["label"], scores)),
            "observation_top1": float(np.mean(best["label"] == 1)),
        }
