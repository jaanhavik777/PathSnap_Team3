# Logistic Regression and KNN results

These are measured results from the two original baselines. Run the commands in
the [README](../README.md) to regenerate the detailed tables under
`artifacts/results/`. Tuned model metrics should be reported separately after
running the training-only search.

## Data coverage

| Measure | Count | Rate |
| --- | ---: | ---: |
| Training observations with weak labels | 121,337 / 208,417 | 58.22% |
| Training candidate recall on independent weak reference matches | 121,337 / 121,404 | 99.94% |
| Student-test observations with weak labels | 10,271 / 19,160 | 53.61% |

The candidate generator uses a 50 m search radius and up to eight arcs. Weak
labels use geometric and chronological checks against the reference route,
as documented in the [feature contract](feature_contract.md).

## Training cross-validation

| Model | Mean average precision | Mean observation top-1 |
| --- | ---: | ---: |
| Logistic Regression | 0.8597 | 83.99% |
| KNN | 0.8827 | 83.95% |

Five folds hold out whole trips. Imputation and scaling are fitted on training
rows inside each fold. KNN uses a deterministic, stratified sample of up to
25,000 training candidate rows per fold and for the final fit.

## Student-test comparison

| Model | Candidate top-1 | Positive-class F1 at 0.5 | Candidate log loss | Mean decoded continuity | Mean route arc F1 | Exact routes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 88.26% | 0.8586 | 0.1645 | 99.98% | 0.5517 | 0/10 |
| KNN | 88.60% | 0.8725 | 0.2524 | 99.98% | 0.5137 | 0/10 |

Candidate metrics pool weakly labeled observations. Route metrics cover all
ten trips after Viterbi decoding. Continuity measures graph connectivity
between decoded observations. Route arc F1 counts overlapping arc occurrences,
while exact route match requires an identical ordered arc sequence and zero
disconnections. Use matching F1 definitions and decoding stages when comparing
against the other team models.

Evaluator scores and decoded routes were generated for both baselines without
loading evaluator reference routes. Select any submission model using the
documented development criteria.

For the Random Forest comparison, use the [shared comparison CSV](../reports/data/student_test_baseline_summary.csv).
It includes pre-decoder direct continuity (93.58% for Logistic Regression and
93.26% for KNN) and log loss using the same probability clipping as the Random
Forest evaluation (0.1645 and 0.1756, respectively). The
[model report](../reports/sections/04_models.md) explains the denominators and
includes the comparison figures.

The [candidate confusion matrix figure](../reports/figures/baselines/student_test_confusion_matrices.png)
and its [aggregate counts](../reports/data/student_test_confusion_counts.csv)
use threshold 0.5 and omit blank weak labels. Regenerate them with
`python -m scripts.plot_baseline_confusion`.
