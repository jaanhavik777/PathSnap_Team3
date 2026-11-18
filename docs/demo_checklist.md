# Review 2 demo checklist — Person 4

## Confirmed in this checkout/work session

- Reproduced MLP training/evaluation on supplied train and student-test partitions only. Exact metrics and the 40-iteration convergence warning are in `docs/mlp_person4_results.md`.
- Saved pipeline, JSON/CSV metrics, loss curve, confusion matrix, and performance plot.
- Full repository tests: **36 passed**. Two warnings come from the deliberately tiny five-iteration MLP unit-test fixture.
- Streamlit started locally and its health endpoint returned `ok`.
- Ran Person 3's current FastAPI branch (`feature/hitha-hmm-api`, commit `f25601b`) locally with its HGB artifact and the permitted student-test partition. The real frontend client fetched health and all 10 dynamic trip IDs, posted two selected IDs, parsed genuine backend metrics/geometry, and rendered both Folium maps. API request and inference logs appeared in the backend terminal. Details are in `docs/api_contract.md`.

## Laptop 1 — backend

Use the repository version that contains Person 3's FastAPI commit. Install requirements, set the data root to the permitted demonstration partition, point to the deployed classifier, and start the actual API:

```powershell
python -m pip install -r requirements.txt
$env:DATA_ROOT = "C:\\path\\to\\Student folder\\student_test"
$env:MODEL_PATH = "artifacts/models/hist_gradient_boosting.joblib"
$env:EVALUATION_PARTITION = "student_test"
uvicorn api.main:app --host 0.0.0.0 --port 8000
ipconfig
```

For final evaluation, use the exact reserved partition authorized by the organizers and set `DATA_ROOT` / `EVALUATION_PARTITION` accordingly. Keep the terminal visible for genuine request and inference logs. Allow inbound TCP 8000 on the private network if required.

## Laptop 2 — Streamlit

```powershell
python -m pip install -r requirements.txt
$env:PATHSNAP_API_URL = "http://<LAPTOP_1_PRIVATE_IPV4>:8000"
Invoke-RestMethod "$env:PATHSNAP_API_URL/api/health"
python -m streamlit run frontend/app.py
```

Select a trip ID returned by the backend, click **Predict route**, show the real server log on Laptop 1, explain the returned metrics and missing values, and repeat with another allowed trip. No mock predictions are used in the evaluator UI.

## Integration boundary

Two actual student-test predictions and map renders succeeded on one computer. A separate Laptop 1/Laptop 2 run, firewall configuration, and route selection from the reserved evaluator partition have not been performed in this environment; do not claim those as passed. On Laptop 2, verify the private IP, same reachable network, API health endpoint, and port 8000 before the review.
