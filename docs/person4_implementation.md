# Person 4 implementation notes

## Current status

The first remote fetch initially had no repository history. The teammate's scaffold has since been fetched, and this work is on local branch `feature/thanvitha-mlp-frontend`, based on `origin/main`. The official student archive contains 80 training trips and 10 student-testing trips, each with `.track`, `.nodes`, `.arcs`, and `.route`. No evaluator-testing partition or populated shared Person 1 feature pipeline was present in the supplied archive/scaffold. `.track` rows are longitude, latitude, seconds; `.nodes` rows are longitude, latitude; `.arcs` rows index nodes and their row index is arc ID; `.route` is a sequence of arc IDs and is not a row-wise GPS label array.

The MLP code uses a provisional association: for each GPS point it spatially retrieves nearby road segments and labels only a route-listed segment that is closest, within 35 m, and at least 2 m closer than another route-listed candidate. All other observations are unlabeled for candidate classification. This avoids asserting a false row alignment, but it is an independent provisional feature/label implementation and must be aligned with Person 1's agreed shared pipeline before team model comparisons. Projection uses local equirectangular metres centered at each trip latitude; short-trip use only. Candidate indexing uses a segment midpoint KD tree to keep networks with large arc counts tractable.

## MLP

The MLP is a supervised candidate-match classifier. Median imputation and standardization are fitted as part of the sklearn pipeline, including independently within each GroupKFold training fold. The hidden layer has 32 ReLU units. Adam optimizes the classifier's log-loss with L2 regularization (`alpha=0.001`), seed 42, and early stopping. Candidate-level metrics are separate from observation Top-1, which ranks candidate probabilities within each GPS observation and scores only labeled observations. The student test output is development evidence and its coverage limitation must accompany any reported score.

## Frontend and map

The Streamlit client gets the trip list and each newly generated result from FastAPI. It displays only returned metrics and geometry, has no route-label or local dataset access, and drops results when requesting another prediction. The Folium renderer accepts GPS `[lat,lon]` and road-segment GeoJSON `[lon,lat]`, draws separately toggled layers, and omits malformed coordinates safely. The API contract must be confirmed with Person 3; no backend was in the repository at inspection time. No cross-laptop integration is claimed.

AI assistance was used to draft and implement code. The team member must review, understand, reproduce and disclose assistance under the event policy.
