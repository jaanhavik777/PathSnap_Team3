"""Frontend HTTP and map checks; API responses are mocked test fixtures only."""

import pytest
import requests

from frontend.api_client import APIError, PathSnapClient
from frontend.map_layers import build_prediction_map


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


def test_trip_ids_are_loaded_dynamically_and_preserved(monkeypatch):
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url))
        return FakeResponse({"trip_ids": [{"trip_id": "0001"}, "trip-2"]})

    monkeypatch.setattr("frontend.api_client.requests.request", request)
    client = PathSnapClient("http://laptop1:8000")
    assert client.trips() == ["0001", "trip-2"]
    assert calls == [("GET", "http://laptop1:8000/api/trips")]


def test_prediction_posts_selected_id_and_validates_response(monkeypatch):
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return FakeResponse({
            "trip_id": "0001", "gps_points": [[12.0, 77.0]],
            "road_segments": [], "predicted_route": [], "metrics": {},
            "inference_time_ms": 12.5,
        })

    monkeypatch.setattr("frontend.api_client.requests.request", request)
    data = PathSnapClient("http://laptop1:8000").predict("0001")
    assert data["trip_id"] == "0001"
    assert calls[0][2]["json"] == {"trip_id": "0001"}


@pytest.mark.parametrize("exception, message", [
    (requests.Timeout("late"), "timed out"),
    (requests.ConnectionError("offline"), "Backend unavailable"),
])
def test_backend_failures_become_clear_api_errors(monkeypatch, exception, message):
    def request(*args, **kwargs):
        raise exception

    monkeypatch.setattr("frontend.api_client.requests.request", request)
    with pytest.raises(APIError, match=message):
        PathSnapClient("http://offline", timeout=0.2).health()


def test_malformed_json_and_http_error_are_reported(monkeypatch):
    class BadJson(FakeResponse):
        def json(self):
            raise ValueError("bad json")

    monkeypatch.setattr("frontend.api_client.requests.request", lambda *a, **k: BadJson({}))
    with pytest.raises(APIError, match="malformed JSON"):
        PathSnapClient("http://bad").health()

    class BadStatus(FakeResponse):
        def raise_for_status(self):
            raise requests.HTTPError("404")

    monkeypatch.setattr("frontend.api_client.requests.request", lambda *a, **k: BadStatus({}))
    with pytest.raises(APIError, match="Backend unavailable"):
        PathSnapClient("http://bad").health()


def test_invalid_prediction_contract_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "frontend.api_client.requests.request",
        lambda *a, **k: FakeResponse({"trip_id": "other", "metrics": {}}),
    )
    with pytest.raises(APIError, match="trip_id mismatch"):
        PathSnapClient("http://test").predict("wanted")

    monkeypatch.setattr(
        "frontend.api_client.requests.request",
        lambda *a, **k: FakeResponse({"trip_id": "wanted", "metrics": [], "gps_points": []}),
    )
    with pytest.raises(APIError, match="metrics must be an object"):
        PathSnapClient("http://test").predict("wanted")


def test_map_uses_geojson_coordinates_and_real_arc_ids():
    data = {
        "gps_points": [{"lat": 12.0, "lon": 77.0}],
        "road_segments": [{"arc_id": 42, "coordinates": [[77.0, 12.0], [77.1, 12.1]]}],
        "predicted_route": [{"arc_id": 42, "geometry": {"type": "LineString", "coordinates": [[77.0, 12.0], [77.1, 12.1]]}}],
    }
    rendered = build_prediction_map(data).get_root().render()
    for expected in (
        "77.0", "12.0", "Road arc 42", "Predicted arc 42",
        "Relevant road segments", "Original GPS observations",
    ):
        assert expected in rendered


def test_map_tolerates_malformed_geometry_but_requires_a_valid_point():
    data = {
        "gps_points": [["bad", "point"], [91, 181], [12, 77]],
        "road_segments": [{"arc_id": 2, "coordinates": [[None, 12], [77, 12]]}],
        "predicted_route": [],
    }
    rendered = build_prediction_map(data).get_root().render()
    assert "GPS observation 1" in rendered
    with pytest.raises(ValueError, match="no usable map geometry"):
        build_prediction_map({"gps_points": [], "road_segments": [], "predicted_route": []})
