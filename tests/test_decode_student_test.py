import pandas as pd

from scripts.decode_student_test import decode_scores, route_metrics


def test_score_export_can_feed_team_decoder():
    trip = {
        "trip_id": "test",
        "track": pd.DataFrame({
            "longitude": [0.00025, 0.00275], "latitude": [0, 0], "timestamp": [0, 10],
        }),
        "nodes": pd.DataFrame({
            "longitude": [0, 0.001, 0.002, 0.003], "latitude": [0, 0, 0, 0],
        }),
        "arcs": pd.DataFrame({
            "arc_id": [0, 1, 2], "from_node_id": [0, 1, 2], "to_node_id": [1, 2, 3],
        }),
    }
    scores = pd.DataFrame({
        "observation_id": [1, 0],
        "arc_id": [2, 0],
        "candidate_rank": [1, 1],
        "distance_m": [0.0, 0.0],
        "projection_fraction": [0.75, 0.25],
        "candidate_score": [0.9, 0.9],
    })
    route = decode_scores(trip, scores)
    assert route.reconstructed_arc_sequence == [0, 1, 2]
    assert route.disconnected_jumps == 0


def test_arc_overlap_counts_occurrences_and_exact_match_checks_order():
    metrics = route_metrics([2, 1, 1], [1, 2], route_breaks=0)
    assert metrics["matched_arc_occurrences"] == 2
    assert metrics["arc_f1"] == 0.8
    assert not metrics["exact_route_match"]
    assert not route_metrics([1, 2], [1, 2], route_breaks=1)["exact_route_match"]
