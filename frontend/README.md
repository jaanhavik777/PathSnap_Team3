# Streamlit evaluator UI

The UI gets both the allowed trip IDs and prediction data from FastAPI. It has no data-folder access and never reads reference labels.

```powershell
$env:PATHSNAP_API_URL = "http://<LAPTOP_1_LAN_IP>:8000"
python -m streamlit run frontend/app.py --server.address 0.0.0.0
```

The backend should expose `GET /api/health`, `GET /api/trips` (JSON `trip_ids` or `trips` list), and `POST /api/predict` accepting `{"trip_id":"..."}`. Prediction geometry uses GPS points as `[lat, lon]` or objects with `lat`/`lon`; road and predicted segment geometries use GeoJSON coordinate order `[lon, lat]` (either arrays or objects with `coordinates`). Metrics are shown only when returned. No mock fallback is enabled.

For Laptop 1, start the team's FastAPI app bound to `0.0.0.0:8000`; find its LAN address with `ipconfig`. On Laptop 2, set `PATHSNAP_API_URL` to `http://<LAN-IP>:8000`. Verify `http://<LAN-IP>:8000/api/health` opens before starting Streamlit. Allow TCP port 8000 through the host firewall on the private network. Check both devices use the same reachable network, and verify the backend's `/docs` page for its actual schema. API schema is pending Person 3 because the repository contains no backend yet.
