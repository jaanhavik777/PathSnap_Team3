# Person 4 MLP experiment record

## Shared-pipeline baseline

Run on 2026-11-18 using the team's shared parser, candidate generator, feature builder, and nullable route labeler. The supplied archive contained 80 train trips and 10 student-test trips; 78 train trips contained labeled candidate rows and contributed to fitting/CV. All 10 student-test trips contributed labeled rows. The evaluator partition was not supplied and was not accessed.

The nine shared inputs were `distance_m`, `projection_fraction`, `candidate_rank`, `arc_length_m`, `from_node_in_degree`, `to_node_out_degree`, `step_distance_m`, `speed_mps`, and `heading_error_deg`. Candidate generation used the shared 50 m radius and up to 8 candidates. The shared labeler used its documented 30 m maximum distance, 5 m ambiguity margin, route uniqueness and chronological consistency checks. Unresolved labels remained null and were excluded, never coerced to non-match.

Configuration: `MLPClassifier(hidden_layer_sizes=(32,), activation=relu, solver=adam, alpha=0.001, batch_size=512, learning_rate_init=0.001, max_iter=40, early_stopping=False, random_state=42)`, with median imputation and standardization inside the sklearn pipeline. GroupKFold used three trip-grouped folds; each fold fitted its own imputer and scaler. No evaluator data was loaded.

| Measurement | Actual result |
|---|---:|
| Labeled train candidate rows / positive rows | 614,508 / 121,337 |
| Train trips with labeled rows | 78 of 80 |
| Grouped CV candidate Macro F1 (mean ± SD) | 0.8924 ± 0.0167 |
| Mean grouped-fold training Macro F1 | 0.9085 |
| Mean grouped-fold observation Top-1 | 0.8390 |
| Full-data training time | 56.40 s |
| Labeled student candidate rows | 50,337 |
| Student candidate accuracy | 0.9530 |
| Student candidate Macro F1 | 0.9280 |
| Student precision / recall | 0.8801 / 0.8912 |
| Student observation Top-1 accuracy | 0.8884 |
| Labeled student observations | 10,271 |
| Student log loss | 0.1137 |
| Student inference/evaluation time | 0.172 s |

Observation Top-1 ranks candidate probabilities within each labeled observation; the denominator is 10,271, not every GPS point. Candidate accuracy is row-level and should not be compared directly with Top-1. The MLP hit the configured 40-iteration limit and scikit-learn emitted convergence warnings for the three CV fits and full-data fit. Training loss decreased, but further iterations/experimentation may improve convergence. Mean fold training F1 was 0.0166 above validation F1; student-test scores are development measurements and should not be used as evaluator-test claims. Use these results for team comparison only if the other models use the same shared features, labels, partitions, and metric definitions.

Artifacts: saved pipeline in `artifacts/models/mlp/`, metrics in `artifacts/metrics/`, and actual training-loss, confusion-matrix, and performance plots in `artifacts/plots/`.
