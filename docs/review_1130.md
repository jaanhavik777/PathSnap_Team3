# 11:30 review — Person 4 (Thanvitha)

## Deliverables

- Baseline MLP trained using the shared parser, candidate generator, feature builder, and route labeler.
- Trip-grouped three-fold validation; imputation and scaling fitted inside each fold.
- Saved pipeline, measured metrics, training loss plot, confusion matrix, and comparison plot.
- Streamlit UI with dynamic backend-provided trip selection, Predict action, result metrics, error handling, and Folium geometry layers.
- Proposed frontend request/response contract documented in `docs/api_contract.md`.
- A development map was generated locally from student trip `00000000`; regenerate it with the command in `README.md` if a standalone HTML is needed.

## MLP baseline results

- Shared features: nine candidate/motion/network features from `src.data.features.FEATURE_COLUMNS`.
- Labeled train rows: 614,508 from 78 train trips; 80 train trips were available, with two having no labeled candidate rows under the shared labeler.
- Grouped CV Macro F1: 0.8924 ± 0.0167; fold Top-1 mean: 0.8390.
- Student candidate Macro F1: 0.9280; candidate accuracy: 0.9530.
- Student observation Top-1: 0.8884 on 10,271 labeled observations.
- Student log loss: 0.1137.
- The full-data model and CV fits reached the 40-iteration limit; retain the convergence caveat.

## Verification and dependencies

- `python -m pytest tests/test_person4.py -q`: 8 passed. Two expected warnings are emitted by the five-iteration tiny MLP fit in the unit test.
- Streamlit startup and its local health endpoint were verified. API calls are unit-tested with mocked responses.
- The API modules have not yet been confirmed to implement the documented contract; live request and cross-laptop integration remain pending Person 3's backend confirmation.
- The evaluator-test partition was not supplied and was not accessed.
