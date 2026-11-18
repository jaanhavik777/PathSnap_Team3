# Review 2 — Person 4 (Thanvitha)

## Completed and verified in this branch

- **Final MLP baseline run:** trained again with seed 42 and the team's shared feature and weak-label pipeline on the supplied `train/` and `student_test/` partitions only. It produced a saved scikit-learn pipeline, metrics JSON/CSV, training-loss curve, confusion matrix, and performance plot.
- **Measured results:** 3-fold trip-grouped CV candidate Macro F1 `0.8924 ± 0.0167`; student-test candidate Macro F1 `0.9280`; student-test candidate accuracy `0.9530`; student-test observation Top-1 `0.8884` over 10,271 labeled observations; student-test log loss `0.1137`. Model fit reached the 40-iteration limit, which remains a convergence limitation. Full metrics and interpretation are in `mlp_person4_results.md` and `../reports/sections/07_api_frontend.md`.
- **Streamlit dashboard:** dynamic trip list, API connection status, predict action, metric display, clear errors, and stale-result clearing on trip changes.
- **Folium map:** API-returned original GPS, relevant road geometries, and predicted route as separate layers; arc IDs in tooltips, endpoints, map bounds, and safe handling for invalid/missing geometry.
- **Reproducible integration checks:** HTTP client validation, mocked health/trip/predict responses, timeout/HTTP/JSON errors, selected-trip payload, response mismatch, coordinate order, road arc IDs, and missing geometry.
- **Execution verification:** `python -m pytest -q` on latest `origin/main` plus this feature work → **40 passed, 2 warnings** (both from the deliberately tiny five-iteration MLP unit-test fixture). Streamlit started locally on port 8767 and `/_stcore/health` returned `ok`.

## MLP score table

| Metric | Value |
| --- | ---: |
| Labeled train candidate rows | 614,508 |
| Train trips with labeled candidates | 78 / 80 |
| Trip-grouped CV Macro F1 | 0.8924 ± 0.0167 |
| Grouped-fold observation Top-1 mean | 0.8390 |
| Student candidate accuracy | 0.9530 |
| Student candidate Macro F1 | 0.9280 |
| Student precision / recall | 0.8801 / 0.8912 |
| Student observation Top-1 | 0.8884 (10,271 labeled observations) |
| Student log loss | 0.1137 |
| Fit / student evaluation time | 49.26 s / 0.216 s |

Student-test scores are development results, not reserved evaluator performance. The reference-route labeler leaves uncertain observations nullable; such rows are excluded from training/evaluation. No evaluator partition was supplied in the archive or opened by the training command.

## Remaining cross-team dependency

The API implementation files under `api/` are empty in the current checkout. Therefore no real FastAPI route/schema, live predictions, server inference logs, or Laptop 1-to-Laptop 2 integration can be claimed. The frontend contract is proposed in `api_contract.md`; Person 3 must supply/confirm the running backend contract before this part can be demonstrated end-to-end. The UI has no artificial fallback. Follow `demo_checklist.md` and use a real backend response for the live demo.

## Commands for review

```powershell
python -m scripts.train_mlp_person4 --data-root "C:\path\to\Student folder" --output artifacts/models/mlp --seed 42 --radius-m 50 --max-candidates 8 --folds 3
python -m pytest -q
python -m streamlit run frontend/app.py
```

For two laptops, set `$env:PATHSNAP_API_URL = "http://<LAPTOP_1_PRIVATE_IPV4>:8000"` on Laptop 2 after Person 3 starts the confirmed backend on Laptop 1. See `api_contract.md` and `demo_checklist.md` for the proposed payload and network steps.
