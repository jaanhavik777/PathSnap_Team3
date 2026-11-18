# Person 4 MLP

Train on the official `train/` partition and evaluate development-only on `student_test/`; the CLI never reads an evaluator partition. It derives positive candidate labels only when a route-listed arc is an unambiguous nearest arc within 35 m. This is a transparent provisional association policy; review and agree it with the team's shared data pipeline before comparing models. Ambiguous and unmatched observations are excluded from labeled metrics, and the JSON logs labeled coverage.

```powershell
python -m pip install -r requirements.txt
python -m scripts.train_mlp_person4 --data-root "C:\path\to\Student folder" --output artifacts/models/mlp
```

The expected data root has `train/` and `student_test/` subfolders. The command writes the fitted pipeline under `artifacts/models/mlp/`, metrics under `artifacts/metrics/`, and actual training loss/confusion/performance plots under `artifacts/plots/`. Three-fold GroupKFold is grouped by trip, and imputation/scaling is fitted inside each fold. The architecture is one 32-unit ReLU layer, Adam, alpha .001, seed 42, max 80 iterations and early stopping. Candidate count defaults to four for tractable laptop experiments; set `--candidate-count` to investigate other values and record those runs separately.
