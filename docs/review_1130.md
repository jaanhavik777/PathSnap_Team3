# 11:30 review — Person 4 (Thanvitha)

## Deliverables prepared

- Baseline MLP training/evaluation on the 80-trip train partition; student-test development metrics are recorded in `docs/mlp_person4_results.md` and `artifacts/metrics/mlp_metrics.json`.
- Trip-grouped three-fold cross-validation, saved scikit-learn preprocessing/model pipeline, training loss plot, confusion matrix, and performance plot.
- Streamlit UI with backend-provided trip selection, Predict button, processing/errors, returned metrics, and Folium map layers.
- Request/response contract proposal in `docs/api_contract.md` for coordination with Person 3.
- Folium layers for GPS, local road geometry, model predictions, and a permitted development reference route. The exporter was run locally on student trip `00000000`; regenerate the large standalone HTML using the command in `README.md` if needed.

## Measured baseline

- Grouped CV candidate Macro F1: 0.8679 ± 0.0210.
- Student-test candidate Macro F1: 0.8903.
- Student-test observation Top-1: 0.8633 on 14,698 labeled observations.
- Student-test log loss: 0.2027.
- All results use the documented provisional route-to-GPS association rule. Ambiguous or unmatched observations are excluded. Align candidate and label generation with Person 1 before direct comparisons.

## Demo status and dependencies

- `python -m pytest tests/test_person4.py -q`: 8 passed (the five-iteration toy MLP test emits an expected convergence warning).
- Streamlit process startup and its health endpoint were verified locally.
- The FastAPI scaffold and schemas are empty; live request handling and cross-laptop communication are pending Person 3's implementation and API confirmation. No end-to-end backend success is claimed.
- The evaluator-test partition was not supplied and was not used.
