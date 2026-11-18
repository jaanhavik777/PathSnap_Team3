# Candidate feature table

Call `build_trip_dataset(load_trip(folder, include_route=False))` for inference.
The CLI `python -m scripts.build_features --help` lists CSV export options.

Each row represents one directed road candidate. The row key is
`(trip_id, observation_id, arc_id)` within a partition. Read `trip_id` as a
string to preserve leading zeros. Observation IDs are zero-based track row
positions. Candidate ranks start at one. Use `FEATURE_COLUMNS` from
`src.data.features` to select model inputs.

| Feature | Meaning |
| --- | --- |
| `distance_m` | Distance to the closest point on the road segment in a trip-local metric plane |
| `projection_fraction` | Closest point along the directed arc, from zero at its start to one at its end |
| `candidate_rank` | Distance rank for the observation, with ties resolved by arc ID |
| `arc_length_m` | Directed segment length in metres |
| `from_node_in_degree` | Number of network arcs entering the candidate's start node |
| `to_node_out_degree` | Number of network arcs leaving the candidate's end node |
| `step_distance_m` | GPS displacement from the preceding observation |
| `speed_mps` | Displacement divided by positive elapsed time |
| `heading_error_deg` | Absolute angle between incoming GPS motion and the arc direction |

The first observation has blank motion features. Speed is also blank when
elapsed time is nonpositive. Heading is blank for zero displacement or
a zero-length arc.

The default search returns up to eight arcs within 50 metres. A radius miss
falls back to the nearest arc when the network has at least one arc.

Use `--labels` with `--partition train` for training or `--partition student_test` for development evaluation. Evaluator-test exports always omit labels.
A positive requires an unambiguous reference arc within 30 metres, with the
second-closest reference arc more than 5 metres farther away. The selected arc
must occur once on the reference route, agree with the ordering of other
spatial matches, and appear among the generated candidates. Other candidates
at that observation receive zero. Unresolved observations have blank labels.
For supervised training, filter blank labels and keep trips together when
splitting rows for validation.
