"""
Prediction and map matching endpoint for PathSnap API with visible terminal logging.
"""

from __future__ import annotations
import logging
import sys
import time
from fastapi import APIRouter, HTTPException, status
from api.dependencies import DATA_ROOT
from api.schemas import PredictRequest, PredictResponse, PredictionMetrics
from api.services.prediction_service import execute_prediction
from api.services.evaluation_service import execute_server_evaluation

router = APIRouter(tags=["Prediction"])

logger = logging.getLogger("pathsnap.api")


@router.post("/predict", response_model=PredictResponse)
@router.post("/api/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    """
    Execute map-matching and route reconstruction for selected trip.
    Followed by isolated server-side evaluation (Laptop 1 ground truth).
    """
    trip_id = request.trip_id.strip()
    t_start = time.perf_counter()
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # Visible terminal log header
    print("\n" + "=" * 78, flush=True)
    print(f"[{timestamp_str}] [FASTAPI REQUEST] POST /predict for Trip '{trip_id}'", flush=True)
    print("=" * 78, flush=True)

    # 1. Run Pure ML + HMM Inference Pipeline (zero ground-truth leakage)
    try:
        pred_res = execute_prediction(trip_id=trip_id, data_dir=DATA_ROOT)
    except FileNotFoundError as fnf:
        print(f"[{timestamp_str}] [ERROR] Trip '{trip_id}' not found in {DATA_ROOT}", flush=True)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip '{trip_id}' not found in active data partition.",
        )
    except Exception as exc:
        print(f"[{timestamp_str}] [ERROR] Prediction pipeline failure: {exc}", flush=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Map-matching inference failed: {str(exc)}",
        )

    obs_count = pred_res["observation_count"]
    recon_count = len(pred_res["reconstructed_arc_sequence"])
    continuity = pred_res["route_continuity"]
    timing = pred_res["timing"]

    # Terminal logs for pipeline stages
    print(f"  • GPS Observations:       {obs_count:,}", flush=True)
    print(f"  • Candidate Generation:   {timing['candidate_gen_ms']:.1f} ms", flush=True)
    print(f"  • Model Inference ({pred_res['model_name']}): {timing['model_infer_ms']:.1f} ms", flush=True)
    print(f"  • HMM / Viterbi Decoding: {timing['viterbi_ms']:.1f} ms (Score: {pred_res['viterbi_log_score']:.2f})", flush=True)
    print(f"  • Route Reconstruction:   {timing['reconstruct_ms']:.1f} ms ({recon_count:,} continuous arcs)", flush=True)
    print(f"  • Route Continuity:       {continuity * 100:.2f}% (Disconnected hops: {pred_res['disconnected_jumps']})", flush=True)

    # 2. Server-side Evaluation (Isolated ground truth on Laptop 1 only)
    print(f"  • Executing Server-Side Ground-Truth Evaluation...", flush=True)
    eval_res = execute_server_evaluation(
        trip_id=trip_id,
        predicted_arc_ids=pred_res["predicted_arc_ids"],
        data_dir=DATA_ROOT,
    )

    acc_val = eval_res["top1_accuracy"]
    if eval_res["metrics_available"] and acc_val is not None:
        print(f"  • Server Evaluation:      Top-1 Accuracy = {acc_val * 100:.2f}% (Status: {eval_res['evaluation_reason']})", flush=True)
    else:
        print(f"  • Server Evaluation:      Accuracy Unavailable ({eval_res['evaluation_reason']})", flush=True)

    total_latency_ms = (time.perf_counter() - t_start) * 1000
    print(f"  • Total Pipeline Latency: {total_latency_ms:.1f} ms", flush=True)
    print("=" * 78 + "\n", flush=True)

    # 3. Assemble Response
    metrics = PredictionMetrics(
        accuracy=acc_val,
        top1_accuracy=acc_val,
        route_continuity=round(continuity, 4),
        invalid_transition_rate=round(pred_res["invalid_transition_rate"], 4),
        inference_time_ms=round(total_latency_ms, 2),
        gps_observations=obs_count,
        predicted_segments=recon_count,
        metrics_available=eval_res["metrics_available"],
        evaluation_reason=eval_res["evaluation_reason"],
    )

    return PredictResponse(
        trip_id=trip_id,
        status=pred_res["status"],
        model_name=pred_res["model_name"],
        observation_count=obs_count,
        predicted_arc_ids=pred_res["predicted_arc_ids"],
        reconstructed_arc_sequence=pred_res["reconstructed_arc_sequence"],
        route_geometry=pred_res["route_geometry"],
        gps_geometry=pred_res["gps_geometry"],
        gps_points=pred_res["gps_points"],
        predicted_route=pred_res["predicted_route"],
        road_segments=pred_res["road_segments"],
        metrics=metrics,
        inference_time_ms=round(total_latency_ms, 2),
        warnings=pred_res["warnings"],
    )
