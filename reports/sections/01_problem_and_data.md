# Problem and data

PathSnap matches a sequence of GPS observations to a directed road graph and reconstructs the route a vehicle drove. The pipeline generates candidate arcs for each observation, scores them with a classifier, and uses Viterbi decoding and the road graph to connect successive observations.

| File | Contents |
| --- | --- |
| `.track` | Longitude, latitude, and timestamp in seconds per GPS observation |
| `.nodes` | Longitude and latitude per road-network vertex |
| `.arcs` | Source and destination node IDs for each directed road segment |
| `.route` | Chronological reference arc sequence |

Node and arc IDs are zero-based file row positions and belong to a single trip. Reference arc rows have no per-observation timestamps. The training partition has 80 trips, and the student-test development partition has 10. Training and parameter selection use grouped trip folds. Student-test provides the development comparison, and evaluator inference uses unlabeled inputs only.

See the [dataset format](../../docs/dataset_format.md) and [EDA and preprocessing](02_eda_and_preprocessing.md).
