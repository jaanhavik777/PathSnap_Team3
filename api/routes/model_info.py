"""
Model information and configuration metadata endpoints for PathSnap API.
"""

from fastapi import APIRouter
from api.dependencies import HMM_DEFAULT_CONFIG, get_model_adapter
from api.schemas import ModelInfoResponse

router = APIRouter(tags=["Model Info"])


@router.get("/model-info", response_model=ModelInfoResponse)
@router.get("/api/model-info", response_model=ModelInfoResponse)
def model_info():
    """Return loaded model architecture, feature names, and HMM settings."""
    adapter = get_model_adapter()
    info = adapter.get_info()
    return ModelInfoResponse(
        model_name=adapter.model_name,
        model_family=adapter.model_family,
        model_type=info.get("model_type", "Classifier"),
        feature_names=adapter.feature_names,
        num_features=len(adapter.feature_names),
        hmm_config=HMM_DEFAULT_CONFIG,
        model_metadata=adapter.metadata,
    )
