"""
PathSnap FastAPI Application (Laptop 1 Deployment).

Runs map-matching and route reconstruction backend with isolated server evaluation.
"""

from __future__ import annotations
import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import (
    ALLOWED_ORIGINS,
    DATA_ROOT,
    EVALUATION_PARTITION,
    MODEL_PATH,
    get_model_adapter,
)
from api.routes import (
    health_router,
    trips_router,
    predict_router,
    model_info_router,
)

# Configure logging for Laptop 1 terminal display
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap-Server] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pathsnap.api")

app = FastAPI(
    title="PathSnap API",
    description="Distributed Map-Matching and Route Reconstruction Backend (Laptop 1)",
    version="1.0.0",
)

# CORS middleware for Laptop 2 frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(health_router)
app.include_router(trips_router)
app.include_router(predict_router)
app.include_router(model_info_router)


@app.on_event("startup")
def on_startup():
    print("=" * 78, flush=True)
    print("        PATHSNAP FASTAPI BACKEND READY — LAPTOP 1 DEPLOYMENT          ", flush=True)
    print("=" * 78, flush=True)
    print(f"  • Active Partition:     {EVALUATION_PARTITION}", flush=True)
    print(f"  • Data Root Directory:  {DATA_ROOT.resolve()}", flush=True)
    print(f"  • Model Artifact Path:  {MODEL_PATH.resolve()}", flush=True)
    try:
        adapter = get_model_adapter()
        print(f"  • Loaded Model:         {adapter.model_name} ({adapter.model_family})", flush=True)
        print(f"  • Features ({len(adapter.feature_names)}):       {', '.join(adapter.feature_names)}", flush=True)
        print(f"  • Viterbi Decoder:      INITIALIZED AND READY", flush=True)
    except Exception as exc:
        print(f"  • Model Loading Note:   {exc}", flush=True)
    print(f"  • Server Listening On:  http://0.0.0.0:8000 (LAN Accessible)", flush=True)
    print(f"  • Swagger Docs:         http://127.0.0.1:8000/docs", flush=True)
    print("=" * 78 + "\n", flush=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
