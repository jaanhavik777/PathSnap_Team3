"""Conservative weak labels from the chronological reference route."""

import numpy as np
import pandas as pd

from src.data.candidates import SegmentIndex, trip_coordinates


def label_candidates(trip, candidates, *, max_distance_m=30.0, ambiguity_margin_m=5.0):
    """Add nullable binary labels, leaving unresolved observations unlabeled.

    Match against all reference arcs before checking candidate coverage. Reject
    ambiguous geometry, repeated route arcs, and anchors that conflict in order.
    """
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError("max_distance_m must be positive and finite")
    if not np.isfinite(ambiguity_margin_m) or ambiguity_margin_m < 0:
        raise ValueError("ambiguity_margin_m must be nonnegative and finite")
    result = candidates.copy()
    result["label"] = pd.array([pd.NA] * len(result), dtype="Int8")
    route = trip["route"]
    if route is None or route.empty or result.empty:
        return result

    track_xy, node_xy = trip_coordinates(trip)
    route_ids = pd.Index(route["arc_id"])
    unique_ids = route_ids.unique()
    route_arcs = trip["arcs"].set_index("arc_id", drop=False).loc[unique_ids]
    index = SegmentIndex(trip["nodes"], node_xy, route_arcs)
    repeated = route_ids.duplicated(keep=False)
    positions = {
        arc_id: position
        for position, arc_id in enumerate(route_ids)
        if not repeated[position]
    }

    anchors = []
    for observation, point in enumerate(track_xy):
        arc_ids, distances, _ = index.nearby(point, max_distance_m + ambiguity_margin_m)
        if not len(arc_ids) or distances[0] > max_distance_m:
            continue
        if len(arc_ids) > 1 and distances[1] - distances[0] <= ambiguity_margin_m:
            continue
        arc_id = arc_ids[0]
        if arc_id in positions:
            anchors.append((observation, arc_id, positions[arc_id]))

    if not anchors:
        return result
    route_positions = np.array([position for _, _, position in anchors])
    prefix_max = np.maximum.accumulate(route_positions)
    suffix_min = np.minimum.accumulate(route_positions[::-1])[::-1]
    aligned = (route_positions == prefix_max) & (route_positions == suffix_min)
    winners = {
        observation: arc_id
        for (observation, arc_id, _), keep in zip(anchors, aligned)
        if keep
    }
    chosen = result["observation_id"].map(winners)
    is_positive = result["arc_id"].eq(chosen)
    covered = result["observation_id"].isin(result.loc[is_positive, "observation_id"])
    result.loc[covered, "label"] = is_positive[covered].astype("int8")
    return result
