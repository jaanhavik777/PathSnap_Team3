# PathSnap — Person 3 to Person 4 Backend API Handoff Contract

**Milestone**: Review 2 (3:00 PM)  
**Provider**: Person 3 — Hitha (FastAPI Backend, Map-Matching & Evaluation Engineer)  
**Consumer**: Person 4 — Thanvitha (Streamlit Frontend & Visualization Engineer)  
**Host Architecture**: Two-Laptop Network Deployment (Laptop 1: Backend / Laptop 2: Frontend)  

---

## 1. Quickstart: Starting the Backend on Laptop 1

On **Laptop 1**, navigate to the repository root and start the production FastAPI server:

```powershell
# Set PYTHONPATH to project root
$env:PYTHONPATH="."

# Run FastAPI via Uvicorn bound to all interfaces (0.0.0.0)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Expected Startup Terminal Banner on Laptop 1:
```
==============================================================================
        PATHSNAP FASTAPI BACKEND READY — LAPTOP 1 DEPLOYMENT          
==============================================================================
  • Active Partition:     student_test
  • Data Root Directory:  C:\Users\hitha\Videos\ML LAB\PathSnap_Team3_repo\data\raw\student_test
  • Model Artifact Path:  C:\Users\hitha\Videos\ML LAB\PathSnap_Team3_repo\artifacts\models\hist_gradient_boosting.joblib
  • Loaded Model:         HistGradientBoostingClassifier (Ensemble/Classifier)
  • Features (8):         distance_meters, heading_diff_deg, projection_fraction, segment_length_m, gps_speed_mps, bearing_cos, bearing_sin, proximity_score
  • Viterbi Decoder:      INITIALIZED AND READY
  • Server Listening On:  http://0.0.0.0:8000 (LAN Accessible)
  • Swagger Docs:         http://127.0.0.1:8000/docs
==============================================================================
```

---

## 2. Two-Laptop Network Connection (Laptop 2 $\rightarrow$ Laptop 1)

### 2.1 Find Laptop 1's LAN IP Address
On Laptop 1, run in PowerShell:
```powershell
ipconfig | Select-String "IPv4"
# Example output: IPv4 Address. . . . . . . . . . . : 192.168.1.20
```

### 2.2 Laptop 1 Windows Firewall Inbound Rule
To allow incoming HTTP connections from Laptop 2 to port 8000, run once as Administrator on Laptop 1:
```powershell
New-NetFirewallRule -DisplayName "PathSnap-FastAPI-Port8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

### 2.3 Configure Laptop 2 to Connect to Laptop 1
On **Laptop 2**, set the environment variable pointing to Laptop 1's IP before launching Streamlit:

```bash
# On Laptop 2 (replace with Laptop 1's actual LAN IP):
export PATHSNAP_API_URL="http://192.168.1.20:8000"

# Verify connectivity from Laptop 2 using curl:
curl http://192.168.1.20:8000/api/health

# Launch Streamlit on Laptop 2:
streamlit run frontend/app.py
```

*(If running locally on a single laptop for testing, leave default `http://127.0.0.1:8000`).*

---

## 3. Endpoints & API Contract Reference

The backend provides dual route aliases for each endpoint (`/api/...` and `/...`) so that Person 4's `PathSnapClient` and standard REST clients work without modifications.

### 3.1 `GET /api/health` (or `/health`)
Health check and model readiness probe.

**Response Schema (`200 OK`)**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "HistGradientBoostingClassifier",
  "decoder_ready": true,
  "active_partition": "student_test",
  "data_root": "C:\\Users\\hitha\\Videos\\ML LAB\\PathSnap_Team3_repo\\data\\raw\\student_test",
  "version": "1.0.0"
}
```

---

### 3.2 `GET /api/trips` (or `/trips`)
Lists all available evaluation trips dynamically from the active data partition (no hardcoded IDs).

**Response Schema (`200 OK`)**:
```json
{
  "partition": "student_test",
  "total_trips": 10,
  "trips": [
    "00000000",
    "00000007",
    "00000009",
    "00000040",
    "00000057",
    "00000065",
    "00000071",
    "00000083",
    "00000091",
    "00000094"
  ],
  "trip_ids": [
    "00000000",
    "00000007",
    "00000009",
    "00000040",
    "00000057",
    "00000065",
    "00000071",
    "00000083",
    "00000091",
    "00000094"
  ],
  "trip_details": [
    {"trip_id": "00000000", "has_reference": true},
    {"trip_id": "00000007", "has_reference": true}
  ]
}
```

---

### 3.3 `POST /api/predict` (or `/predict`)
Runs map matching, Viterbi decoding, route reconstruction, and isolated server-side ground-truth evaluation.

**Request JSON**:
```json
{
  "trip_id": "00000000"
}
```

**Actual Response JSON (`200 OK`)**:
```json
{
  "trip_id": "00000000",
  "status": "SUCCESS",
  "model_name": "HistGradientBoostingClassifier",
  "observation_count": 2503,
  "predicted_arc_ids": [12, 12, 45, 88],
  "reconstructed_arc_sequence": [12, 13, 14, 45, 88],
  "route_geometry": [
    [50.001234, 8.501234],
    [50.001567, 8.501890]
  ],
  "gps_geometry": [
    [50.001220, 8.501240],
    [50.001550, 8.501910]
  ],
  "gps_points": [
    {"lat": 50.001220, "lon": 8.501240},
    {"lat": 50.001550, "lon": 8.501910}
  ],
  "predicted_route": [
    {
      "arc_id": 12,
      "coordinates": [[8.501234, 50.001234], [8.501890, 50.001567]]
    }
  ],
  "road_segments": [
    {
      "arc_id": 12,
      "coordinates": [[8.501234, 50.001234], [8.501890, 50.001567]]
    }
  ],
  "metrics": {
    "accuracy": 0.8102,
    "top1_accuracy": 0.8102,
    "route_continuity": 0.9500,
    "invalid_transition_rate": 0.0500,
    "inference_time_ms": 3840.5,
    "gps_observations": 2503,
    "predicted_segments": 815,
    "metrics_available": true,
    "evaluation_reason": "Evaluated against verified server-side ground-truth route."
  },
  "inference_time_ms": 3840.5,
  "warnings": []
}
```

---

### 3.4 `GET /api/model-info` (or `/model-info`)
Displays active model architecture, loaded feature set, and HMM hyperparameters.

**Response Schema (`200 OK`)**:
```json
{
  "model_name": "HistGradientBoostingClassifier",
  "model_family": "Ensemble/Classifier",
  "model_type": "HistGradientBoostingClassifier",
  "feature_names": [
    "distance_meters",
    "heading_diff_deg",
    "projection_fraction",
    "segment_length_m",
    "gps_speed_mps",
    "bearing_cos",
    "bearing_sin",
    "proximity_score"
  ],
  "num_features": 8,
  "hmm_config": {
    "beta_meters": 15.0,
    "sigma_z_meters": 12.0,
    "max_speed_mps": 45.0,
    "gap_threshold_sec": 10.0
  },
  "model_metadata": {
    "learning_rate": 0.1,
    "max_iter": 150,
    "max_leaf_nodes": 31
  }
}
```

---

## 4. Coordinate System & Geometry Conventions

To eliminate rendering bugs between GeoJSON standards and Leaflet/Folium:

1. **`predicted_route` & `road_segments` (`coordinates`)**:
   - Order: `[longitude, latitude]` (Standard GeoJSON LineString standard).
   - Consumed directly by `map_layers.py`'s `_segment()` function.
2. **`gps_points`**:
   - Format: `[{"lat": <float>, "lon": <float>}, ...]`.
   - Consumed directly by `map_layers.py`'s `_point()` function.
3. **`route_geometry` & `gps_geometry`**:
   - Order: `[[latitude, longitude], ...]`.
   - Native Folium / Leaflet PolyLine representation.

---

## 5. Security & Ground-Truth Isolation Guarantees

1. **Prediction Isolation**: The prediction function `predict_trip` operates strictly with `include_route=False`. It never opens, reads, parses, or accesses any `.route` ground-truth file.
2. **Server-Side Evaluation**: The evaluation function `evaluate_prediction` executes strictly on Laptop 1 *after* prediction has finalized.
3. **No Leakage**: Ground-truth arc IDs and reference geometries are never returned in the API response or transmitted across the local network to Laptop 2.
4. **Evaluator Mode**: If evaluating a partition where `.route` is omitted (e.g. `evaluator_test`), `metrics.accuracy` automatically returns `null` with `metrics_available: false` and `evaluation_reason: "Reference route not present on server"`. The UI displays `Accuracy: Unavailable` without crashing.

---

## 6. Error Responses

- **Trip Not Found (`404 Not Found`)**:
  ```json
  {
    "detail": "Trip 'nonexistent_trip' not found in active data partition."
  }
  ```
- **Model Not Found (`503 Service Unavailable`)**:
  ```json
  {
    "detail": "Trained model artifact not found. Train the model before serving."
  }
  ```
- **Internal Inference Error (`500 Internal Server Error`)**:
  ```json
  {
    "detail": "Map-matching inference failed: <error message>"
  }
  ```
