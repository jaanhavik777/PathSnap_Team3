"""Inference-safe features on candidate rows."""

import numpy as np

from src.data.candidates import SegmentIndex, trip_coordinates

FEATURE_COLUMNS = (
    "distance_m",
    "projection_fraction",
    "candidate_rank",
    "arc_length_m",
    "from_node_in_degree",
    "to_node_out_degree",
    "step_distance_m",
    "speed_mps",
    "heading_error_deg",
)


def build_features(trip, candidates):
    """Add arc geometry and incoming GPS motion, preserving candidate row order.

    Motion features without valid history are NaN. Headings are directed.
    """
    result = candidates.copy()
    track_xy, node_xy = trip_coordinates(trip)
    nodes = trip["nodes"]
    network = trip["arcs"]
    unique_arcs = network.set_index("arc_id", drop=False).loc[result["arc_id"].unique()]
    arc_positions = unique_arcs.index.get_indexer(result["arc_id"])
    arcs = unique_arcs.iloc[arc_positions]
    result["from_node_in_degree"] = (
        arcs["from_node_id"].map(network["to_node_id"].value_counts()).fillna(0).to_numpy(dtype=np.int64)
    )
    result["to_node_out_degree"] = (
        arcs["to_node_id"].map(network["from_node_id"].value_counts()).fillna(0).to_numpy(dtype=np.int64)
    )
    geometry = SegmentIndex(nodes, node_xy, unique_arcs)
    lengths = geometry.lengths[arc_positions]
    result["arc_length_m"] = lengths

    observations = result["observation_id"].to_numpy(dtype=np.intp)
    steps = np.full_like(track_xy, np.nan)
    steps[1:] = np.diff(track_xy, axis=0)
    steps = steps[observations]
    distance = np.linalg.norm(steps, axis=1)
    result["step_distance_m"] = distance
    timestamps = trip["track"]["timestamp"].to_numpy(dtype=float)
    elapsed = np.full(len(timestamps), np.nan)
    elapsed[1:] = np.diff(timestamps)
    elapsed = elapsed[observations]
    result["speed_mps"] = np.divide(
        distance, elapsed, out=np.full(len(result), np.nan),
        where=np.isfinite(elapsed) & (elapsed > 0),
    )
    step_angle = np.arctan2(steps[:, 1], steps[:, 0])
    vectors = geometry.vectors[arc_positions]
    arc_angle = np.arctan2(vectors[:, 1], vectors[:, 0])
    error = np.abs((np.degrees(step_angle - arc_angle) + 180) % 360 - 180)
    error[(distance == 0) | (lengths == 0)] = np.nan
    result["heading_error_deg"] = error
    return result
