"""
Unit and integration tests for PathSnap FastAPI backend.

Review 2 deliverable for Person 3 (Hitha).
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import DATA_ROOT

client = TestClient(app)

_has_student_trip_0 = (
    (DATA_ROOT / "00000000" / "00000000.track").is_file()
    or (DATA_ROOT / "00000000.track").is_file()
)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "model_name" in data
    assert data["decoder_ready"] is True
    assert "version" in data


def test_api_health_alias():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")


def test_trips_endpoint():
    response = client.get("/trips")
    assert response.status_code == 200
    data = response.json()
    assert "trips" in data
    assert "trip_ids" in data
    assert isinstance(data["trips"], list)
    if _has_student_trip_0:
        assert len(data["trips"]) > 0
    # Trip IDs must be strings
    assert all(isinstance(t, str) for t in data["trips"])


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "feature_names" in data
    assert isinstance(data["feature_names"], list)
    assert len(data["feature_names"]) > 0
    assert "hmm_config" in data


def test_predict_endpoint_contract(monkeypatch):
    """Verify /predict response schema and pipeline contracts in all environments."""
    mock_prediction = {
        "trip_id": "mock_trip_test",
        "status": "SUCCESS",
        "observation_count": 5,
        "predicted_arc_ids": [10, 11, 12, 13, 14],
        "reconstructed_arc_sequence": [10, 11, 12, 13, 14],
        "route_geometry": [[10.0, 50.0], [10.01, 50.01]],
        "gps_geometry": [[10.0, 50.0], [10.002, 50.002]],
        "gps_points": [{"lat": 50.0, "lon": 10.0}],
        "predicted_route": [{"arc_id": 10, "coordinates": [[10.0, 50.0], [10.01, 50.01]]}],
        "road_segments": [{"arc_id": 10, "coordinates": [[10.0, 50.0], [10.01, 50.01]]}],
        "warnings": [],
        "route_continuity": 1.0,
        "invalid_transition_rate": 0.0,
        "disconnected_jumps": 0,
        "viterbi_log_score": -1.5,
        "model_name": "HistGradientBoostingClassifier",
        "timing": {
            "candidate_gen_ms": 1.0,
            "model_infer_ms": 2.0,
            "viterbi_ms": 3.0,
            "reconstruct_ms": 1.0,
            "total_ms": 7.0,
        },
    }
    mock_evaluation = {
        "metrics_available": True,
        "top1_accuracy": 1.0,
        "matched_observations": 5,
        "candidate_miss_count": 0,
        "candidate_miss_rate": 0.0,
        "evaluation_reason": "Evaluated against ground truth",
    }
    monkeypatch.setattr("api.routes.predict.execute_prediction", lambda **kwargs: mock_prediction)
    monkeypatch.setattr("api.routes.predict.execute_server_evaluation", lambda **kwargs: mock_evaluation)

    response = client.post("/predict", json={"trip_id": "mock_trip_test"})
    assert response.status_code == 200
    data = response.json()
    assert data["trip_id"] == "mock_trip_test"
    assert data["observation_count"] == 5
    assert len(data["predicted_arc_ids"]) == 5
    assert data["metrics"]["route_continuity"] == 1.0
    assert data["metrics"]["top1_accuracy"] == 1.0
    assert len(data["gps_points"]) == 1
    assert len(data["predicted_route"]) == 1


@pytest.mark.skipif(
    not _has_student_trip_0,
    reason="Student test trip dataset supplied locally, not tracked in git",
)
def test_predict_real_student_trip():
    # Test on trip 00000000
    response = client.post("/predict", json={"trip_id": "00000000"})
    assert response.status_code == 200
    data = response.json()

    # Core response contract
    assert data["trip_id"] == "00000000"
    assert data["status"] in ("SUCCESS", "WARNING")
    assert data["observation_count"] == 2503
    assert len(data["predicted_arc_ids"]) == 2503
    assert len(data["reconstructed_arc_sequence"]) > 0
    assert len(data["route_geometry"]) > 0
    assert len(data["gps_geometry"]) == 2503
    assert len(data["gps_points"]) == 2503
    assert len(data["predicted_route"]) > 0

    # Metrics validation
    metrics = data["metrics"]
    assert isinstance(metrics, dict)
    assert "route_continuity" in metrics
    assert metrics["route_continuity"] > 0.90  # Must be > 90% continuity
    assert "inference_time_ms" in data
    assert data["inference_time_ms"] > 0.0


@pytest.mark.skipif(
    not _has_student_trip_0,
    reason="Student test trip dataset supplied locally, not tracked in git",
)
def test_api_predict_alias_compatible_with_frontend():
    # Test POST /api/predict expected by Person 4's PathSnapClient
    response = client.post("/api/predict", json={"trip_id": "00000000"})
    assert response.status_code == 200
    data = response.json()
    assert str(data.get("trip_id")) == "00000000"
    assert isinstance(data.get("metrics"), dict)
    assert isinstance(data.get("gps_points"), list)
    assert isinstance(data.get("predicted_route"), list)


def test_predict_invalid_trip_404():
    response = client.post("/predict", json={"trip_id": "nonexistent_trip_99999"})
    assert response.status_code == 404
