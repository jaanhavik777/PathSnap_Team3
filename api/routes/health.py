"""
Health check endpoints for PathSnap API.
"""

from fastapi import APIRouter
from api.dependencies import DATA_ROOT, EVALUATION_PARTITION, get_model_adapter
from api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
@router.get("/api/health", response_model=HealthResponse)
def health_check():
    """Return API health status, loaded model name, and decoder readiness."""
    model_loaded = False
    model_name = "Not Loaded"
    try:
        adapter = get_model_adapter()
        model_loaded = True
        model_name = adapter.model_name
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=model_name,
        decoder_ready=True,
        active_partition=EVALUATION_PARTITION,
        data_root=str(DATA_ROOT.resolve()),
        version="1.0.0",
    )
