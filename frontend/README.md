# PathSnap evaluator UI

Run from the repository root after installing `python -m pip install -r requirements.txt`:

```powershell
$env:PATHSNAP_API_URL = "http://<LAPTOP_1_PRIVATE_IPV4>:8000"
python -m streamlit run frontend/app.py
```

For local development without the environment variable, the API client uses `http://127.0.0.1:8000`. The UI requests health, permitted trip IDs, and a fresh prediction from FastAPI. It does not load project data, reference labels, saved predictions, or model artifacts.

The client matches Person 3's current backend branch contract: `GET /api/health`, `GET /api/trips`, and `POST /api/predict`. Current `main` does not yet include that backend revision, but local end-to-end tests against Person 3's remote branch passed for two student-test trips. See [the API contract and results](../docs/api_contract.md); physical two-laptop connectivity still needs review-day verification.

The map displays GPS observations (blue), relevant road arcs (grey), predicted route arcs (green), start/end markers, layer controls, and arc IDs. Geometry comes from the API response, with GeoJSON road coordinates ordered `[longitude, latitude]`.
