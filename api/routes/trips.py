"""
Trip discovery endpoints for PathSnap API.
"""

from fastapi import APIRouter, HTTPException, status
from api.dependencies import DATA_ROOT, EVALUATION_PARTITION
from api.schemas import TripListResponse
from api.services.trip_service import list_available_trips

router = APIRouter(tags=["Trips"])


@router.get("/trips", response_model=TripListResponse)
@router.get("/api/trips", response_model=TripListResponse)
def get_trips():
    """Dynamically list available trip IDs in active partition without hardcoding."""
    trip_ids, details = list_available_trips(DATA_ROOT)
    return TripListResponse(
        partition=EVALUATION_PARTITION,
        total_trips=len(trip_ids),
        trips=trip_ids,
        trip_ids=trip_ids,  # Alias for Streamlit frontend compatibility
        trip_details=details,
    )
