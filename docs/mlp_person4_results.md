# Person 4 MLP experiment record

## Completed experiment

Run on 2026-11-18 with the supplied archive: 80 training trips and all 10 student-test trips. The evaluator partition was not present in the archive and was not used. Six candidate features were used: point-to-segment distance, movement heading alignment, capped speed, segment length, projection progress, and log time interval. Candidate generation used four nearby midpoint-indexed segments followed by exact segment projection. Route-derived labels were accepted only when the nearest candidate arc was in the trip's reference route, within 35 m, and 2 m closer than another route-listed candidate. This is a provisional heuristic; route membership alone is not a temporal GPS association. Unmatched or ambiguous observations are excluded from labeled metrics.

Configuration: `MLPClassifier(hidden_layer_sizes=(32,), activation=relu, solver=adam, alpha=0.001, learning_rate_init=0.001, max_iter=80, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=42)`, with median imputation and standardization in a pipeline. GroupKFold used three folds grouped by trip, with preprocessing refitted inside each fold.

| Measurement | Actual result |
|---|---:|
| Training candidate rows / positive rows | 582,748 / 145,687 |
| Grouped CV candidate Macro F1 (mean ± SD) | 0.8679 ± 0.0210 |
| Mean grouped-fold training Macro F1 | 0.8706 |
| Training time (one full-data fit) | 83.70 s |
| Student candidate rows | 58,792 |
| Student candidate accuracy | 0.9179 |
| Student candidate Macro F1 | 0.8903 |
| Student precision / recall | 0.8374 / 0.8333 |
| Student observation Top-1 accuracy | 0.8633 |
| Labeled student observations | 14,698 (coverage denominator) |
| Student log loss | 0.2027 |
| Student inference/evaluation time | 6.29 s |

Top-1 is calculated only among observations with an unambiguous route-derived label and candidate coverage. The 14,698 labeled observations are not all GPS points; the report must state this restriction. Candidate accuracy is dominated by non-matches and should not be compared directly with Top-1. Training loss decreased across the saved training iterations; it does not itself establish generalization. Some CV fold fits emitted the maximum-iteration convergence warning at 80 iterations; the full-data estimator stopped early after its recorded 43 iterations. Mean fold training Macro F1 (0.8706) was close to validation (0.8679), with no large CV gap under this association heuristic. Student metrics are development results and were not used to tune this MLP. Compare with teammates only after sharing the agreed features, candidate labels, candidate counts, folds, and metric definitions.

The MLP is a comparison model; deployment is a team decision. The full reproducible metrics and graphs are in `artifacts/metrics/`, `artifacts/plots/`, and `artifacts/models/mlp/`.
