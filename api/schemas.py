"""
Pydantic schemas for PathSnap FastAPI backend.

Compatible with both the official requirements and Person 4's Streamlit frontend contract.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="API operational status (healthy/degraded)")
    model_loaded: bool = Field(..., description="Whether classifier model is active")
    model_name: str = Field(..., description="Name of active classifier")
    decoder_ready: bool = Field(..., description="Whether HMM/Viterbi decoder is ready")
    active_partition: str = Field(..., description="Currently active dataset partition")
    data_root: str = Field(..., description="Configured dataset directory path")
    version: str = Field(default="1.0.0", description="PathSnap API version")


class TripSummary(BaseModel):
    trip_id: str
    num_observations: Optional[int] = None
    has_reference: bool = False


class TripListResponse(BaseModel):
    partition: str
    total_trips: int
    trips: List[str]
    trip_ids: List[str]  # Alias for frontend compatibility
    trip_details: Optional[List[TripSummary]] = None


class PredictRequest(BaseModel):
    trip_id: str = Field(..., example="00000000", description="Selected trip identifier")


class PredictionMetrics(BaseModel):
    accuracy: Optional[float] = Field(None, description="Observation Top-1 accuracy (alias for frontend)")
    top1_accuracy: Optional[float] = Field(None, description="Observation-level Top-1 accuracy")
    route_continuity: Optional[float] = Field(None, description="Route continuity ratio across road network")
    invalid_transition_rate: Optional[float] = Field(None, description="Fraction of disconnected hops")
    inference_time_ms: Optional[float] = Field(None, description="Total pipeline latency in milliseconds")
    gps_observations: Optional[int] = Field(None, description="Number of GPS observations evaluated")
    predicted_segments: Optional[int] = Field(None, description="Number of reconstructed road segments")
    metrics_available: bool = Field(False, description="Whether server-side reference evaluation succeeded")
    evaluation_reason: Optional[str] = Field(None, description="Explanation of evaluation status")


class GeoJSONSegment(BaseModel):
    arc_id: int
    coordinates: List[List[float]] = Field(..., description="LineString coordinates [[lon, lat], ...]")


class LatLonPoint(BaseModel):
    lat: float
    lon: float


class PredictResponse(BaseModel):
    trip_id: str
    status: str = Field(..., description="Execution status: SUCCESS, WARNING, or ERROR")
    model_name: str = Field(..., description="Name of model used for prediction")
    observation_count: int = Field(..., description="Number of GPS observations")
    predicted_arc_ids: List[int] = Field(..., description="Observation-level predicted road arc IDs")
    reconstructed_arc_sequence: List[int] = Field(..., description="Continuous road network arc sequence")
    route_geometry: List[List[float]] = Field(..., description="Polyline coordinates [[lat, lon], ...] for Folium")
    gps_geometry: List[List[float]] = Field(..., description="GPS coordinates [[lat, lon], ...] for Folium")
    gps_points: List[LatLonPoint] = Field(..., description="GPS points [{lat, lon}] for frontend client")
    predicted_route: List[GeoJSONSegment] = Field(..., description="Reconstructed route segments in GeoJSON [lon, lat]")
    road_segments: List[GeoJSONSegment] = Field(..., description="Road network segments in GeoJSON [lon, lat]")
    metrics: PredictionMetrics
    inference_time_ms: float = Field(..., description="Total pipeline latency in milliseconds")
    warnings: List[str] = Field(default_factory=list, description="Diagnostic warnings during prediction")


class ModelInfoResponse(BaseModel):
    model_name: str
    model_family: str
    model_type: str
    feature_names: List[str]
    num_features: int
    hmm_config: Dict[str, Any]
    model_metadata: Dict[str, Any]
