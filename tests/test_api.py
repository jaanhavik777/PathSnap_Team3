"""
Unit and integration tests for PathSnap FastAPI backend.

Review 2 deliverable for Person 3 (Hitha).
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


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
