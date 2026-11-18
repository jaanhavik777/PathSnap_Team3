"""HTTP client for the PathSnap prediction API."""

from __future__ import annotations

import os
from typing import Any

import requests


class APIError(RuntimeError):
    """A user-displayable backend communication or contract error."""


class PathSnapClient:
    """Call the agreed API endpoints with bounded timeouts and validation."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = (
            base_url or os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000")
        ).rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = requests.request(
                method, self.base_url + path, timeout=self.timeout, **kwargs
            )
            response.raise_for_status()
            result = response.json()
        except requests.Timeout as exc:
            raise APIError("Backend request timed out") from exc
        except requests.RequestException as exc:
            raise APIError(f"Backend unavailable: {exc}") from exc
        except ValueError as exc:
            raise APIError("Backend returned malformed JSON") from exc
        if not isinstance(result, dict):
            raise APIError("Backend response must be a JSON object")
        return result

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/health")

    def trips(self) -> list[str]:
        result = self._request("GET", "/api/trips")
        items = result.get("trip_ids", result.get("trips"))
        if not isinstance(items, list):
            raise APIError("Trip response must contain a trip_ids or trips list")
        trip_ids = []
        for item in items:
            value = item.get("trip_id") if isinstance(item, dict) else item
            if value is None or not str(value).strip():
                raise APIError("Trip response contains an empty trip ID")
            trip_ids.append(str(value))
        return trip_ids

    def predict(self, trip_id: str) -> dict[str, Any]:
        result = self._request("POST", "/api/predict", json={"trip_id": str(trip_id)})
        if str(result.get("trip_id")) != str(trip_id):
            raise APIError("Prediction response trip_id mismatch")
        for key in ("gps_points", "road_segments", "predicted_route"):
            if key in result and not isinstance(result[key], list):
                raise APIError(f"Prediction response {key} must be a list")
        if "metrics" in result and not isinstance(result["metrics"], dict):
            raise APIError("Prediction response metrics must be an object")
        duration = result.get("inference_time_ms")
        if duration is not None:
            try:
                if float(duration) < 0:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise APIError("inference_time_ms must be a non-negative number or null") from exc
        return result
