import pandas as pd

from scripts.analyze_data import analyze_trip
from scripts.export_results import direct_continuity


def test_components_keep_directed_and_weak_connectivity_separate(directed_trip):
    # The parser's graph contract uses zero-based node positions.
    nodes = {10: 0, 20: 1, 30: 2}
    directed_trip["nodes"]["node_id"] = directed_trip["nodes"]["node_id"].map(nodes)
    for column in ("from_node_id", "to_node_id"):
        directed_trip["arcs"][column] = directed_trip["arcs"][column].map(nodes)
    report = analyze_trip(directed_trip, "train")
    assert report["weak_components"] == 2
    assert report["strong_components"] == 2
    assert report["largest_weak_component_fraction"] == 2 / 3


def test_direct_continuity_pools_pairs_and_respects_trip_boundaries():
    top = pd.DataFrame({
        "trip_id": ["a", "b", "a", "a", "a", "b"],
        "observation_id": [3, 0, 1, 2, 0, 1],
        "arc_id": [8, 4, 6, 6, 4, 8],
    })
    arcs = pd.DataFrame({"from_node_id": [0, 1, 4], "to_node_id": [1, 2, 5]}, index=[4, 6, 8])
    assert direct_continuity(top, {"a": arcs, "b": arcs}) == (2, 4)
