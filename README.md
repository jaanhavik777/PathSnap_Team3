# PathSnap Team 3

Distributed GPS map matching using road candidates, supervised scoring, HMM/Viterbi route reconstruction, FastAPI, and Streamlit/Folium.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the supplied dataset under `data/raw/` or point commands to an extracted data root. `.track`, `.nodes`, `.arcs`, and `.route` parsing plus candidate features are implemented under `src/data/`. `.route` is a chronological arc sequence, **not** GPS-aligned labels. Evaluator data and labels must not enter training, model selection, or the prediction UI.

## Person 4 — baseline MLP

The MLP uses the shared `build_trip_dataset` pipeline and `FEATURE_COLUMNS`/`label_candidates` implementation. The 80 training trips are processed for fitting and trip-grouped validation; 78 trips contributed labeled rows in this run. The 10 student-test trips are used only for development evaluation. Evaluator-test folders are not opened by this command.

```powershell
python -m scripts.train_mlp_person4 `
  --data-root "C:\path\to\Student folder" `
  --output artifacts/models/mlp
```

Outputs are written under `artifacts/models/mlp/`, `artifacts/metrics/`, and `artifacts/plots/`. Current measured results use the shared pipeline and are documented in `docs/mlp_person4_results.md` and `docs/review_1130.md`.

## Person 4 — frontend

The frontend requests available trips and fresh predictions from the FastAPI service; it does not perform map matching or access reference routes. Set the backend address on Laptop 2:

```powershell
$env:PATHSNAP_API_URL = "http://<LAPTOP_1_LAN_IP>:8000"
Invoke-RestMethod "$env:PATHSNAP_API_URL/api/health"
python -m streamlit run frontend/app.py
```

See `docs/api_contract.md`, `frontend/README.md`, and `docs/demo_checklist.md`. The contract must match the backend implementation before live/two-laptop integration can be claimed.

## Data and feature tools

```powershell
python -m scripts.inspect_data --help
python -m scripts.build_features --help
python -m pytest
```

Raw data is gitignored. Do not commit evaluator labels or private credentials.
