# Streamlit to FastAPI interface

## Implementation status

The Streamlit client is implemented and has automated tests with mocked HTTP responses. The current `main` / Person 4 branch has empty API modules under `api/`. Person 3's newer FastAPI implementation is available on remote branch `feature/hitha-hmm-api` (commit `f25601b`) and has been run locally against the permitted `student_test` partition. Its health, trip, prediction, metric, and geometry schemas match this client. That backend implementation is not part of the current Person 4 branch and should arrive through Person 3's branch/PR. The contract below records the verified branch schema.

The UI intentionally has no fake prediction fallback. A mock may be used by isolated tests only.

## Health

`GET /api/health` (also `/health`)

Returns `status`, `model_loaded`, `model_name`, `decoder_ready`, `active_partition`, `data_root`, and `version`. The UI requires a successful HTTP response with a JSON object. If an implementation returns `ready: false`, it is displayed as a warning.

## Available permitted trips

`GET /api/trips` (also `/trips`)

Returns `partition`, `total_trips`, `trips`, `trip_ids`, and optional `trip_details`. The current client reads `trip_ids`. The backend must return only trips allowed in the current evaluation context; IDs are never hardcoded in Streamlit.

## Prediction

`POST /api/predict` (also `/predict`)

Request:

```json
{"trip_id":"00000000"}
```

Response shape (example values are illustrative, not measured results):

```json
{
  "trip_id":"00000000",
  "gps_points":[{"lat":12.0,"lon":77.0}],
  "road_segments":[{"arc_id":42,"coordinates":[[77.0,12.0],[77.1,12.1]]}],
  "predicted_route":[{"arc_id":42,"coordinates":[[77.0,12.0],[77.1,12.1]]}],
  "metrics":{"accuracy":null,"top1_accuracy":null,"route_continuity":0.95,"invalid_transition_rate":0.05,"gps_observations":1,"predicted_segments":1,"metrics_available":false,"evaluation_reason":"Evaluation unavailable"},
  "inference_time_ms":1.0,
  "status":"SUCCESS",
  "model_name":"...",
  "observation_count":1,
  "predicted_arc_ids":[],
  "reconstructed_arc_sequence":[],
  "route_geometry":[],
  "gps_geometry":[],
  "warnings":[]
}
```

Coordinates for GPS points are `{lat, lon}` or two-element `[latitude, longitude]` arrays. Road geometries are GeoJSON LineString or MultiLineString coordinates in `[longitude, latitude]` order. Each segment should preserve its `arc_id`. Include only prediction output and server-computed metrics permitted by the evaluator; never send evaluator reference routes or labels. `accuracy` and other metrics must be omitted or null when they cannot be measured under the event rules.

The client checks status codes, timeouts, JSON object shape, trip ID equality, geometry list types, metric object type, and inference time. The map skips invalid coordinates, preserves arc IDs in tooltips, shows original GPS / relevant roads / predicted road route as toggleable layers, and identifies GPS endpoints.

## Local end-to-end check

The Person 3 branch was run locally with its HGB artifact and the 10-trip `student_test` partition. The actual client fetched health and trip IDs, requested two trips, parsed the returned metrics and geometry, and rendered both maps. Trip `00000000` returned 2,503 GPS points, route continuity 0.9488, observation accuracy 0.6256, and 6,434.93 ms inference time. Trip `00000007` returned 1,354 GPS points, route continuity 0.9682, observation accuracy 0.7334, and 1,323.2 ms inference time. These are development-trip integration checks, not evaluator results. The API terminal logged the requests and pipeline stages. Both systems ran on this same computer; LAN/firewall and separate-laptop behavior remain to be verified.

## Laptop setup

Laptop 1 (backend), from the repository root, after Person 3's backend commit is merged/present and the dataset/model paths are configured:

```powershell
python -m pip install -r requirements.txt
$env:DATA_ROOT = "C:\path\to\Student folder\student_test"
$env:MODEL_PATH = "artifacts/models/hist_gradient_boosting.joblib"
uvicorn api.main:app --host 0.0.0.0 --port 8000
ipconfig
```

Keep the backend terminal visible for real request/inference logs. Allow inbound TCP port 8000 on the private network if required.

Laptop 2 (frontend):

```powershell
python -m pip install -r requirements.txt
$env:PATHSNAP_API_URL = "http://<LAPTOP_1_PRIVATE_IPV4>:8000"
Invoke-RestMethod "$env:PATHSNAP_API_URL/api/health"
python -m streamlit run frontend/app.py
```

The Streamlit UI defaults to local `127.0.0.1:8000` only for same-laptop development. Set the environment variable for the two-laptop demo. Select a backend-provided trip, predict, and repeat for another ID. A same-computer end-to-end check with two student-test trips passed; a cross-device connection has not been performed from this environment.
