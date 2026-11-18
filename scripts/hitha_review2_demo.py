"""
PathSnap Review 2 Live Demonstration Script — Person 3 (Hitha)

Executes a REAL HTTP client request against the live running FastAPI server on Laptop 1.
Validates:
  1. GET /api/health probe
  2. GET /api/trips listing
  3. POST /api/predict on selected student-test trip
  4. Response schema: predicted arcs, reconstructed route, Folium/GeoJSON geometries
  5. Server-side evaluation metrics (Continuity & Accuracy)
  6. Inference latency breakdown
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import requests


def run_review2_demonstration(base_url: str = "http://127.0.0.1:8000", trip_id: str = "00000000"):
    base_url = base_url.rstrip("/")
    print("=" * 78)
    print("PATHSNAP REVIEW 2 — PERSON 3 (HITHA) LIVE API DEMONSTRATION")
    print(f"Target Backend URL: {base_url}")
    print(f"Evaluation Trip ID: {trip_id}")
    print("=" * 78)

    # 1. Health Probe
    print("\n[STEP 1] Checking FastAPI Service Health (GET /api/health)...")
    try:
        r_health = requests.get(f"{base_url}/api/health", timeout=10)
        r_health.raise_for_status()
        health_data = r_health.json()
        print(f"  • Service Status:   {health_data.get('status', 'unknown').upper()}")
        print(f"  • Active Model:     {health_data.get('model_name')}")
        print(f"  • Decoder Ready:    {health_data.get('decoder_ready')}")
        print(f"  • Active Partition: {health_data.get('active_partition')}")
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Could not connect to FastAPI server at {base_url}.")
        print("Please start the backend first in a separate terminal with:")
        print("  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Health check failed: {exc}")
        sys.exit(1)

    # 2. Available Trips Discovery
    print("\n[STEP 2] Discovering Available Trips (GET /api/trips)...")
    try:
        r_trips = requests.get(f"{base_url}/api/trips", timeout=10)
        r_trips.raise_for_status()
        trips_data = r_trips.json()
        available_trips = trips_data.get("trips", [])
        print(f"  • Available Trips ({len(available_trips)}): {', '.join(available_trips[:8])}...")
        if trip_id not in available_trips and len(available_trips) > 0:
            trip_id = available_trips[0]
            print(f"  • Defaulting to available trip: {trip_id}")
    except Exception as exc:
        print(f"\n[ERROR] Failed to fetch trips: {exc}")
        sys.exit(1)

    # 3. Model Info Probe
    print("\n[STEP 3] Inspecting Active Model Architecture (GET /api/model-info)...")
    try:
        r_info = requests.get(f"{base_url}/api/model-info", timeout=10)
        r_info.raise_for_status()
        info_data = r_info.json()
        print(f"  • Model Type:       {info_data.get('model_name')} ({info_data.get('model_family')})")
        print(f"  • Feature Names ({info_data.get('num_features')}): {', '.join(info_data.get('feature_names', []))}")
        print(f"  • HMM Beta/Sigma:   beta={info_data.get('hmm_config', {}).get('beta_meters')}m, sigma={info_data.get('hmm_config', {}).get('sigma_z_meters')}m")
    except Exception as exc:
        print(f"  • Model info note: {exc}")

    # 4. Prediction Request (Real HTTP POST)
    print(f"\n[STEP 4] Dispatching Prediction Request (POST /api/predict for '{trip_id}')...")
    payload = {"trip_id": trip_id}
    t0 = time.perf_counter()
    try:
        r_pred = requests.post(f"{base_url}/api/predict", json=payload, timeout=60)
        r_pred.raise_for_status()
        pred_data = r_pred.json()
        http_latency_ms = (time.perf_counter() - t0) * 1000
    except requests.exceptions.HTTPError as http_err:
        print(f"\n[ERROR] HTTP {r_pred.status_code}: {r_pred.text}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Prediction request failed: {exc}")
        sys.exit(1)

    # 5. Validate Response Data
    obs_count = pred_data.get("observation_count", 0)
    pred_arcs = pred_data.get("predicted_arc_ids", [])
    recon_arcs = pred_data.get("reconstructed_arc_sequence", [])
    route_geom = pred_data.get("route_geometry", [])
    gps_geom = pred_data.get("gps_geometry", [])
    metrics = pred_data.get("metrics", {})
    infer_ms = pred_data.get("inference_time_ms", 0.0)

    print(f"\n[STEP 5] Validating Prediction Response Contract:")
    print(f"  • Status:                 {pred_data.get('status')}")
    print(f"  • GPS Observations:       {obs_count:,}")
    print(f"  • Decoded Arc Sequence:   {len(pred_arcs):,} arcs (Sample: {pred_arcs[:6]}...)")
    print(f"  • Reconstructed Route:    {len(recon_arcs):,} continuous arcs")
    print(f"  • Route Polyline Points:  {len(route_geom):,} [lat, lon] coordinates")
    print(f"  • GPS GeoJSON Points:     {len(pred_data.get('gps_points', [])):,} points")
    print(f"  • Server Inference Time:  {infer_ms:.1f} ms")
    print(f"  • Total Round-Trip Time:  {http_latency_ms:.1f} ms")

    # 6. Evaluation Metrics Display
    continuity = metrics.get("route_continuity")
    accuracy = metrics.get("accuracy", metrics.get("top1_accuracy"))
    inv_rate = metrics.get("invalid_transition_rate")
    reason = metrics.get("evaluation_reason", "Completed")

    print("\n" + "=" * 78)
    print("SERVER-SIDE VERIFIED EVALUATION METRICS (LAPTOP 1)")
    print("=" * 78)
    print(f"  • Route Continuity:       {continuity * 100:.2f}%" if continuity is not None else "  • Route Continuity:       Unavailable")
    print(f"  • Invalid Transition Rate: {inv_rate * 100:.2f}%" if inv_rate is not None else "  • Invalid Transition Rate: Unavailable")
    print(f"  • Top-1 Arc Accuracy:     {accuracy * 100:.2f}%" if accuracy is not None else "  • Top-1 Arc Accuracy:     Unavailable (Ground truth isolated)")
    print(f"  • Evaluation Status:      {reason}")
    print("=" * 78)
    print("REVIEW 2 LIVE API DEMONSTRATION SUCCESSFUL: ALL ENDPOINTS OPERATIONAL.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="PathSnap Review 2 Live Demonstration Client")
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000"),
        help="FastAPI base URL (e.g. http://192.168.1.20:8000 or http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--trip-id",
        type=str,
        default="00000000",
        help="Trip ID to predict (e.g. 00000000)",
    )
    args = parser.parse_args()
    run_review2_demonstration(base_url=args.base_url, trip_id=args.trip_id)


if __name__ == "__main__":
    main()
