# Person 4 MLP reproduction

The MLP consumes `src.data.build_dataset.build_trip_dataset`, `src.data.features.FEATURE_COLUMNS`, and `src.data.labels.label_candidates`. Training and evaluation use the shared 50 m candidate radius, up to 8 candidates, and weak-label thresholds of 30 m and 5 m ambiguity margin. Unresolved observations remain null and are removed only from supervised metrics/fitting.

```powershell
python -m pip install -r requirements.txt
python -m scripts.train_mlp_person4 `
  --data-root "C:\path\to\Student folder" `
  --output artifacts/models/mlp `
  --seed 42 --radius-m 50 --max-candidates 8 --folds 3
```

Expected data root has `train/` and `student_test/` directories. The CLI does not enumerate `evaluator_test/`. Model/config metadata and loss history are saved under `artifacts/models/mlp/`; metrics are copied into `artifacts/metrics/`, plots into `artifacts/plots/`. GroupKFold keeps trips intact, and each fold fits its own median imputer and standard scaler. The baseline uses one 32-unit ReLU layer, Adam, alpha 0.001, batch size 512, 40 maximum iterations, and no internal random-row early stopping. The measured fit reached the iteration limit; see the recorded convergence note in `docs/mlp_person4_results.md`.
