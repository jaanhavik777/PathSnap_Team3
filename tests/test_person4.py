from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import pytest

from frontend.api_client import APIError, PathSnapClient
from frontend.map_layers import build_prediction_map
from src.data.build_dataset import build_trip_dataset
from src.data.features import FEATURE_COLUMNS
from src.data.person4_maps import build_map
from src.models.train_mlp import make_model


def test_mlp_rows_use_shared_features_and_nullable_weak_labels(directed_trip):
    rows = build_trip_dataset(
        directed_trip, radius_m=10_000, max_candidates=8, include_labels=True,
    )
    assert set(FEATURE_COLUMNS).issubset(rows.columns)
    assert rows.label.notna().any()
    assert set(rows.loc[rows.label.notna(), "label"].astype(int)).issubset({0, 1})
    assert rows.loc[rows.label == 1, "arc_id"].eq(8).all()


def test_shared_trip_geometry_map_has_prediction_layer(directed_trip):
    html = build_map(directed_trip, predicted_arc_ids=[8]).get_root().render()
    assert "Predicted route" in html
    assert "GPS observations" in html
    assert "Road network" in html


def test_api_trip_response(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"trip_ids": ["a", "b"]}

    monkeypatch.setattr("frontend.api_client.requests.request", lambda *a, **k: Response())
    assert PathSnapClient("http://fake").trips() == ["a", "b"]


def test_prediction_response_validation(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"trip_id": "x", "gps_points": [], "predicted_route": [], "metrics": {}}

    monkeypatch.setattr("frontend.api_client.requests.request", lambda *a, **k: Response())
    assert PathSnapClient("http://fake").predict("x")["trip_id"] == "x"


def test_missing_frontend_geometry_is_reported():
    with pytest.raises(ValueError):
        build_prediction_map({"gps_points": [], "road_segments": [], "predicted_route": []})


def test_mlp_fit_probability_shape_and_reload(tmp_path):
    X = pd.DataFrame({column: np.linspace(0, 1, 24) for column in FEATURE_COLUMNS})
    y = np.array([0, 1] * 12)
    model = make_model(hidden=(4,), max_iter=5, seed=3).fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (24, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    reloaded = joblib.load(path)
    np.testing.assert_array_equal(model.predict(X), reloaded.predict(X))


def test_api_timeout_is_clear(monkeypatch):
    import requests

    def timeout(*args, **kwargs):
        raise requests.Timeout("slow")

    monkeypatch.setattr("frontend.api_client.requests.request", timeout)
    with pytest.raises(APIError, match="timed out"):
        PathSnapClient("http://fake", timeout=0.01).health()


def test_prediction_trip_id_mismatch_is_rejected(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"trip_id": "different", "metrics": {}, "gps_points": [], "predicted_route": []}

    monkeypatch.setattr("frontend.api_client.requests.request", lambda *a, **k: Response())
    with pytest.raises(APIError, match="mismatch"):
        PathSnapClient("http://fake").predict("expected")
