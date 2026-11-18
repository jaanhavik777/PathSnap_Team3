# Person 4 implementation notes

## Data and MLP

The supplied archive contains 80 training trips and 10 student-test trips. No evaluator-test partition was supplied. This branch now merges the latest team `main` and uses the common parser (`src.data.parsers.load_trip`), candidate/feature/label pipeline (`src.data.build_dataset.build_trip_dataset`), and feature schema (`src.data.features.FEATURE_COLUMNS`). `.route` is a chronological sequence of arc IDs, not GPS-aligned labels. The shared labeler returns nullable labels after its distance, ambiguity, route-uniqueness, and order checks; unresolved rows are excluded from model fitting.

The baseline uses a 32-unit ReLU `MLPClassifier`, Adam, alpha 0.001, batch size 512, seed 42, and max 40 iterations. A median imputer and standard scaler are fitted within each fold. Three-fold GroupKFold assigns whole trips to each validation fold. The full-data and fold models hit the 40-iteration cap, so this experiment records a convergence limitation rather than claiming full convergence. Actual scores and sample coverage are in `docs/mlp_person4_results.md` and `artifacts/metrics/mlp_metrics.json`.

## Frontend and maps

The Streamlit client dynamically requests trip IDs and fresh predictions from FastAPI. It displays returned metrics and geometry only; it does not calculate matches or access labels. Its API client handles timeouts, HTTP/JSON errors, mismatched trip IDs, and malformed geometry. The Folium frontend has GPS, road network, and predicted-route layers; a separate development exporter can add reference routes only for `train` or `student_test` trips.

## Integration status

The `api/` files on the merged branch are still empty; `docs/api_contract.md` is therefore a proposed contract awaiting Person 3's confirmation. The UI works with mocked API responses in tests, but live API communication and the two-laptop demo remain unverified. Evaluator data is not included in the MLP training command.

AI assistance was used to draft and implement code. Review, understand, reproduce, and disclose assistance under the event policy.
