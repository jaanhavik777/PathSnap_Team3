# Person 4 — MLP, frontend, and visualization

## MLP model and training

The MLP is a feed-forward classifier that maps each GPS-to-road candidate's engineered attributes to a match probability. ReLU supplies a nonlinear activation between the input and output layers; the Adam optimizer updates weights through backpropagation. The implementation uses scikit-learn `MLPClassifier` with one 32-unit hidden layer, ReLU, Adam, alpha 0.001, batch size 512, learning rate 0.001, seed 42, and 40 maximum iterations. Median imputation and standardization are part of the fitted pipeline. Within each of three `GroupKFold` folds, the complete pipeline is fit on training trips only, so a trip never appears in both sides of a fold.

The feature inputs follow the team's shared pipeline: distance to arc, projection fraction, candidate rank, arc length, start/end graph degree, step distance, speed, and heading error. Candidate labels come from the shared reference-route association logic. It leaves ambiguous or unresolved GPS observations unlabeled; those rows are excluded from training and metrics instead of being treated as negatives. The `.route` file is a chronological sequence of arc IDs, not a row-aligned GPS label vector.

## Measured results

The baseline was run on the supplied 80 training trips and 10 student-test trips. Of 80 train trips, 78 produced labeled candidate rows. The run trained on 614,508 labeled candidate rows, including 121,337 positive rows. It evaluated 50,337 labeled student candidate rows across 10 trips. No evaluator partition was present in the supplied archive, and the training script only opens `train/` and `student_test/`.

| Measure | Result |
| --- | ---: |
| Trip-grouped CV candidate Macro F1 | 0.8924 ± 0.0167 |
| Grouped-fold observation Top-1 mean | 0.8390 |
| Student-test candidate accuracy | 0.9530 |
| Student-test candidate Macro F1 | 0.9280 |
| Student-test precision / recall | 0.8801 / 0.8912 |
| Student-test observation Top-1 | 0.8884 (10,271 labeled observations) |
| Student-test log loss | 0.1137 |
| Final fit time / student evaluation time | 49.26 s / 0.216 s |

Candidate accuracy is row-level; Top-1 is calculated by ranking candidate probabilities within each labeled GPS observation. It is not the proportion of all GPS points matched. These are development-set results and do not estimate performance on a reserved evaluator set. The three fold models and final model all reached the 40-iteration limit. The decreasing loss curve is saved, but this configuration did not converge according to scikit-learn's stopping criterion; higher iteration limits and additional configurations remain future experiments.

The saved estimator is `artifacts/models/mlp/mlp_pipeline.joblib`. Metric JSON/CSV and actual training-loss, confusion-matrix, and performance plots are in `artifacts/metrics/` and `artifacts/plots/`. Reproduce from the repository root with:

```powershell
python -m scripts.train_mlp_person4 --data-root "C:\path\to\Student folder" --output artifacts/models/mlp --seed 42 --radius-m 50 --max-candidates 8 --folds 3
```

## Streamlit and Folium frontend

The Streamlit dashboard requests health status and permitted trip IDs from FastAPI. On **Predict route**, it posts the chosen ID and renders only the response for that trip. It does not load model artifacts, match roads, read local trip data, or access reference labels. It reports backend failures and malformed responses rather than showing saved or fabricated predictions. Changing the selected trip invalidates the displayed result.

The Folium map consumes actual API-returned geometries. It provides separate GPS observation, relevant road segment, and predicted route layers; uses blue, grey, and green styling; preserves arc IDs in road tooltips; includes start/end markers; and fits map bounds to available geometry. It skips invalid coordinates and reports when the response has no usable geometry. GeoJSON segment coordinates are converted from `[longitude, latitude]` to Folium's `[latitude, longitude]` order.

## API and deployment status

The UI and HTTP client match Person 3's current FastAPI branch schema (`feature/hitha-hmm-api`, commit `f25601b`). Although that backend revision is not yet in the current `main` checkout, it was run in a separate detached worktree with the actual HGB artifact and permitted `student_test` data. The client dynamically retrieved ten trip IDs, then requested and rendered two real responses. Streamlit `AppTest` ran the real app against that API, exercised prediction on both IDs, confirmed selection changes clear stale results, and produced zero app exceptions:

| Trip | GPS points | Route geometry lines | Server Top-1 | Continuity | Inference |
| --- | ---: | ---: | ---: | ---: | ---: |
| `00000000` | 2,503 | 440 | 0.6256 | 0.9488 | 6,434.93 ms |
| `00000007` | 1,354 | 187 | 0.7334 | 0.9682 | 1,323.2 ms |

These development-trip measurements confirm client/API/map compatibility only; they are not evaluator results or the MLP's scores. Server-side evaluation runs on Laptop 1; the frontend displays only metrics from the API and receives no reference route. The server terminal showed request/inference logs. A physical Laptop 1/Laptop 2 connection and the reserved evaluator partition were not available for verification.

Local verification: full `pytest` suite on latest `main` plus this feature work, 40 passed; Streamlit started and its local health endpoint returned `ok`; two real API requests and Folium renders succeeded. The full team's final consolidated PDF remains a team deliverable and must include the other members' completed report sections and review-approved screenshots/logs.
