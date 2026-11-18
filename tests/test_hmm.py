"""
Unit tests for PathSnap HMM map matching, RoadGraph, and Viterbi decoder.

Review 1 test suite for Person 3 (Hitha).
"""

from types import SimpleNamespace
import numpy as np
import pytest

from src.data.road_graph import RoadGraph
from src.data.spatial import haversine_distance, calculate_bearing, LocalProjector
from src.hmm.states import CandidateState
from src.hmm.emissions import calculate_emission_log_probs
from src.hmm.transitions import calculate_transition_log_prob
from src.hmm.viterbi import ViterbiDecoder, viterbi_decode
from src.hmm.decode_route import reconstruct_continuous_route
from src.hmm.route_metrics import (
    calculate_route_continuity,
    calculate_invalid_transition_rate,
    calculate_top1_accuracy,
    calculate_candidate_miss_rate,
)
from src.models.train_hist_gradient_boosting import HistGradientBoostingModel


@pytest.fixture
def straight_graph():
    # 4 nodes in a straight line: 0 -> 1 -> 2 -> 3
    nodes = np.array([
        [8.500, 50.000],
        [8.505, 50.000],
        [8.510, 50.000],
        [8.515, 50.000],
    ])
    arcs = np.array([
        [0, 1],  # arc 0
        [1, 2],  # arc 1
        [2, 3],  # arc 2
    ])
    return RoadGraph(nodes, arcs)


# ---------------------------------------------------------
# 1. RoadGraph Tests
# ---------------------------------------------------------
def test_road_graph_construction(straight_graph):
    assert straight_graph.num_nodes == 4
    assert straight_graph.num_arcs == 3
    assert straight_graph.get_arc_nodes(0) == (0, 1)
    assert straight_graph.get_arc_nodes(1) == (1, 2)
    assert straight_graph.get_arc_nodes(2) == (2, 3)

    # Arc lengths should be positive geodesic distances
    for a in range(3):
        assert straight_graph.get_arc_length(a) > 0.0

    # Shortest path from node 0 to node 2 should be nodes [0, 1, 2] and arcs [0, 1]
    assert straight_graph.get_shortest_path_nodes(0, 2) == [0, 1, 2]
    assert straight_graph.get_shortest_path_arcs(0, 2) == [0, 1]


def test_road_graph_directed_connectivity(straight_graph):
    # Directed graph: path 0 -> 2 exists, but 2 -> 0 should be inf
    forward_dist = straight_graph.shortest_path_distance(0, 2)
    reverse_dist = straight_graph.shortest_path_distance(2, 0)
    assert forward_dist < float("inf")
    assert np.isinf(reverse_dist)


def test_road_graph_parallel_arcs():
    # Two nodes with 2 parallel arcs of different lengths
    nodes = np.array([[8.50, 50.00], [8.51, 50.00]])
    arcs = np.array([[0, 1], [0, 1]])  # arc 0 and arc 1 between same nodes
    rg = RoadGraph(nodes, arcs)
    assert rg.num_arcs == 2
    best_arcs = rg.get_shortest_path_arcs(0, 1)
    assert len(best_arcs) == 1


# ---------------------------------------------------------
# 2. HMM Emission Tests
# ---------------------------------------------------------
def test_emission_log_probabilities():
    c0 = SimpleNamespace(arc_id=0, distance_meters=5.0, projection_fraction=0.5)
    c1 = SimpleNamespace(arc_id=1, distance_meters=35.0, projection_fraction=0.5)

    scores = np.array([0.9, 0.15])
    log_em = calculate_emission_log_probs([c0, c1], scores, sigma_z_meters=12.0)

    assert len(log_em) == 2
    # c0 is closer and has higher classifier confidence
    assert log_em[0] > log_em[1]
    # Sum of probabilities must normalize to ~1.0
    probs = np.exp(log_em)
    assert pytest.approx(np.sum(probs), 1e-4) == 1.0


# ---------------------------------------------------------
# 3. HMM Transition Tests
# ---------------------------------------------------------
def test_transition_log_probabilities(straight_graph):
    c_prev = SimpleNamespace(arc_id=0, projection_fraction=0.1)
    c_curr_same = SimpleNamespace(arc_id=0, projection_fraction=0.5)

    arc0_len = straight_graph.get_arc_length(0)
    gps_dist = 0.4 * arc0_len
    # When net_dist matches gps_dist with normal speed, transition penalty is minimal
    log_p = calculate_transition_log_prob(
        straight_graph, c_prev, c_curr_same, gps_dist=gps_dist, dt_sec=10.0, beta_meters=15.0
    )
    assert pytest.approx(log_p, 1e-2) == 0.0

    # Disconnected reverse transition must hit min_log_prob floor
    c_rev = SimpleNamespace(arc_id=1, projection_fraction=0.5)
    log_p_rev = calculate_transition_log_prob(
        straight_graph, c_rev, c_prev, gps_dist=gps_dist, dt_sec=10.0, beta_meters=15.0
    )
    assert log_p_rev <= -30.0


def test_transition_large_gps_gap(straight_graph):
    c_prev = SimpleNamespace(arc_id=0, projection_fraction=0.1)
    c_curr = SimpleNamespace(arc_id=1, projection_fraction=0.5)

    # 60s dropout should be handled gracefully via gap threshold
    log_p = calculate_transition_log_prob(
        straight_graph, c_prev, c_curr, gps_dist=50.0, dt_sec=60.0, gap_threshold_sec=10.0
    )
    assert not np.isnan(log_p)
    assert not np.isinf(log_p)


# ---------------------------------------------------------
# 4. Viterbi Decoder Tests
# ---------------------------------------------------------
def test_viterbi_straight_route(straight_graph):
    decoder = ViterbiDecoder(straight_graph)

    lons = np.array([8.502, 8.507, 8.512])
    lats = np.array([50.000, 50.000, 50.000])
    timestamps = np.array([1.0, 2.0, 3.0])

    cands = [
        [SimpleNamespace(arc_id=0, distance_meters=2.0, projection_fraction=0.4)],
        [SimpleNamespace(arc_id=1, distance_meters=2.0, projection_fraction=0.4)],
        [SimpleNamespace(arc_id=2, distance_meters=2.0, projection_fraction=0.4)],
    ]
    scores = [np.array([0.9]), np.array([0.9]), np.array([0.9])]

    selected, obs_arcs, score, warnings = decoder.decode(
        cands, scores, lons, lats, timestamps
    )
    assert obs_arcs == [0, 1, 2]
    assert len(selected) == 3


def test_viterbi_parallel_road_disambiguation():
    # Two parallel roads: Road A (0->1->2, arcs 0, 1) and Road B (3->4, arc 2)
    nodes = np.array([
        [8.5000, 50.0000],  # 0
        [8.5002, 50.0000],  # 1 (~14m east)
        [8.5004, 50.0000],  # 2 (~14m east)
        [8.5000, 50.0002],  # 3 (~22m north offset)
        [8.5002, 50.0002],  # 4
    ])
    arcs = np.array([
        [0, 1],  # arc 0
        [1, 2],  # arc 1
        [3, 4],  # arc 2 (parallel dead-end)
    ])
    rg = RoadGraph(nodes, arcs)
    decoder = ViterbiDecoder(rg, beta_meters=15.0, sigma_z_meters=15.0)

    lons = np.array([8.5001, 8.5003])
    lats = np.array([50.0000, 50.0000])
    timestamps = np.array([1.0, 2.0])

    # Obs 0: candidate on arc 0 or closer false candidate on arc 2
    c0_A = SimpleNamespace(arc_id=0, distance_meters=5.0, projection_fraction=0.5)
    c0_B = SimpleNamespace(arc_id=2, distance_meters=4.0, projection_fraction=0.5)

    # Obs 1: candidate on arc 1
    c1_A = SimpleNamespace(arc_id=1, distance_meters=5.0, projection_fraction=0.5)

    cands = [[c0_A, c0_B], [c1_A]]
    scores = [np.array([0.5, 0.55]), np.array([0.9])]

    _, obs_arcs, _, _ = decoder.decode(cands, scores, lons, lats, timestamps)
    # Viterbi must select Road A [0, 1] because arc 2 -> arc 1 is physically disconnected
    assert obs_arcs == [0, 1]


# ---------------------------------------------------------
# 5. Route Reconstruction Tests
# ---------------------------------------------------------
def test_route_reconstruction_interpolates_gaps(straight_graph):
    # Observations jump from arc 0 to arc 2, skipping arc 1
    obs_arcs = [0, 2]
    lons = np.array([8.505, 8.515])
    lats = np.array([50.00, 50.00])

    res = reconstruct_continuous_route(obs_arcs, straight_graph, lons, lats)
    # Sequence should interpolate arc 1
    assert res.reconstructed_arc_sequence == [0, 1, 2]
    assert res.continuity_ratio == 1.0
    assert res.disconnected_jumps == 0
    assert len(res.route_geometry) >= 4


# ---------------------------------------------------------
# 6. Route Metrics Tests
# ---------------------------------------------------------
def test_route_metrics(straight_graph):
    # Perfect continuity
    c1 = calculate_route_continuity([0, 1, 2], straight_graph)
    assert c1 == 1.0
    assert calculate_invalid_transition_rate([0, 1, 2], straight_graph) == 0.0

    # Disconnected transition
    c2 = calculate_route_continuity([2, 0], straight_graph)
    assert c2 == 0.0
    assert calculate_invalid_transition_rate([2, 0], straight_graph) == 1.0

    # Top-1 Accuracy
    acc = calculate_top1_accuracy([0, 1, 2], [0, 1, 3])
    assert pytest.approx(acc, 1e-4) == 2.0 / 3.0

    # Candidate Miss Rate
    cands = [
        [SimpleNamespace(arc_id=0)],
        [SimpleNamespace(arc_id=1)],
        [SimpleNamespace(arc_id=4)],  # missed arc 2
    ]
    miss = calculate_candidate_miss_rate(cands, [0, 1, 2])
    assert pytest.approx(miss, 1e-4) == 1.0 / 3.0


# ---------------------------------------------------------
# 7. Model Artifact Load Test
# ---------------------------------------------------------
def test_hist_gradient_boosting_load():
    model_path = "artifacts/models/hist_gradient_boosting.joblib"
    model = HistGradientBoostingModel.load(model_path)
    assert model.is_fitted
    assert len(model.feature_names) == 8

    # Run dummy prediction through model
    dummy_x = np.zeros((2, 8), dtype=np.float32)
    scores = model.predict_candidate_scores(dummy_x)
    assert len(scores) == 2
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
