# PathSnap — Person 4 contribution

This branch contains the MLP comparison model, Streamlit evaluator UI, Folium renderers, actual student-test experiment results, and demo instructions. It follows the team's scaffold directories. The API and shared candidate-feature contract files are still placeholders on the fetched branches, so the API shape and provisional label pipeline need teammate confirmation. Changes are local and have not been pushed.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train and evaluate the MLP

The supplied data root has `train/` and `student_test/`. The script does not open an evaluator-test partition.

```powershell
python -m scripts.train_mlp_person4 `
  --data-root "C:\path\to\Student folder" `
  --output artifacts/models/mlp
```

The fitted model is saved under `artifacts/models/mlp/`; measured metrics and plots are copied into `artifacts/metrics/` and `artifacts/plots/`. See `docs/mlp_person4_results.md` for the actual experiment and label-coverage caveat. Agree the provisional route-to-GPS labels and candidate features with Person 1 before comparing model scores.

Generate a development map from a permitted train or student-test trip:

```powershell
python -m src.models.export_mlp_dev_map `
  --trip-folder "C:\path\to\Student folder\student_test\00000000" `
  --include-reference
```

The exporter refuses reference-route mapping for partitions other than `train` and `student_test`.

## Two-laptop UI

On Laptop 1, Person 3 should start FastAPI bound to `0.0.0.0:8000`, check `/api/health`, and leave the server terminal visible for genuine inference logs. Find Laptop 1's LAN IP using `ipconfig`.

On Laptop 2:

```powershell
$env:PATHSNAP_API_URL = "http://<LAPTOP_1_LAN_IP>:8000"
Invoke-RestMethod "$env:PATHSNAP_API_URL/api/health"
python -m streamlit run frontend/app.py
```

See `frontend/README.md`, `docs/api_contract.md`, and `docs/demo_checklist.md` for expected response geometry and network troubleshooting. The UI uses backend-returned trip IDs and predictions; it has no fake fallback. Real API and cross-laptop tests await Person 3's backend.

## Verification

Run `python -m pytest tests/test_person4.py -q` for Person 4 unit tests. The current result is 8 passed; they cover local model save/reload, feature labels, map layers, and mocked API behavior, not live backend integration.
