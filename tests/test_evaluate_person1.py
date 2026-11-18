import pandas as pd
import pytest

from scripts.evaluate_person1 import candidate_metrics


def test_candidate_metrics_pool_labeled_observations_and_resolve_ties():
    rows = pd.DataFrame({
        "trip_id": ["00000001"] * 5,
        "observation_id": [0, 0, 1, 1, 2],
        "arc_id": [10, 20, 10, 20, 10],
        "candidate_rank": [2, 1, 1, 2, 1],
        "candidate_score": [0.8, 0.8, 0.7, 0.2, 0.99],
        "label": [0, 1, 0, 1, None],
    })
    report = candidate_metrics(rows)
    assert report["label_coverage"] == pytest.approx(2 / 3)
    assert report["top1"] == 0.5
    assert report["candidate_f1_at_0_5"] == pytest.approx(0.4)


def test_candidate_metrics_rejects_partially_labeled_observation():
    rows = pd.DataFrame({
        "trip_id": ["00000001", "00000001"],
        "observation_id": [0, 0], "label": [1, None],
    })
    with pytest.raises(ValueError, match="all candidate labels"):
        candidate_metrics(rows)
