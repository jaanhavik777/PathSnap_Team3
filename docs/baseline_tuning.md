# Logistic Regression and KNN tuning

Run `python -m scripts.run_tuning --folds 3 --trials 11` from the repository
root after building the labeled training candidate table. This experiment
used scikit-learn 1.9.0 and Optuna 5.1.0. The search uses a seeded
Tree-structured Parzen Estimator (TPE), with the original configuration
queued as trial 0. The sampler uses its first ten completed trials during
startup and generates trial 10 using its fitted distributions. The search
space and runtime settings are in [the recorded configuration](../configs/optuna.yaml).

Each trial maximizes mean observation top-1 over three folds grouped by
training trip. The preprocessor is fitted inside each fold. KNN uses a
reproducible stratified sample of at most 25,000 training rows per fit.
Trial 0 is the baseline for this experiment. The five-fold scores in the
[original baseline report](baseline_results.md) come from a separate run.

The runner refits the best configuration on the permitted training rows.
The evaluation command loads those models and measures their student-test
candidate metrics and decoded routes. Evaluator data is excluded from tuning.

## Training-fold results

| Model | Baseline trial top-1 | Selected trial top-1 | Selected parameters |
| --- | ---: | ---: | --- |
| Logistic Regression | 83.74% | 83.90% | `C=0.00140837` |
| KNN | 83.70% | 84.00% | `n_neighbors=31`, `weights=uniform` |

Both studies completed 11 trials and selected trial 10. The CSVs record
all trials, and the JSON files record the selected configuration and fold
metrics.

- [Logistic Regression trials](../reports/data/tuning/logistic_regression_trials.csv)
- [Logistic Regression selected parameters](../reports/data/tuning/logistic_regression_best.json)
- [KNN trials](../reports/data/tuning/knn_trials.csv)
- [KNN selected parameters](../reports/data/tuning/knn_best.json)

## Tuned student-test results

| Model | Candidate top-1 | Positive-class F1 | Candidate log loss | Mean decoded route arc F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 88.34% | 0.8583 | 0.1662 | 0.5564 |
| KNN | 88.80% | 0.8746 | 0.1757 | 0.5358 |

Candidate metrics cover 10,271 weakly labeled observations. Decoded route
metrics cover all ten student-test trips. Log loss uses the evaluator's
standard candidate metric without the optional Random Forest clipping.
Both tuned models still had zero exact route matches. The tracked model
files are the original baselines, and the tuned files remain in
`artifacts/tuned_baselines/` after reproducing the experiment.

[Full tuned student-test summary](../reports/data/tuning/student_test_summary.csv)
