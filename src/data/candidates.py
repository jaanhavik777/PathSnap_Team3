"""Directed road candidates for each GPS observation."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

COORDINATES = ["longitude", "latitude"]
CANDIDATE_DTYPES = {
    "observation_id": "int64",
    "arc_id": "int64",
    "candidate_rank": "int64",
    "distance_m": "float64",
    "projection_fraction": "float64",
}


def trip_coordinates(trip):
    """Return GPS and node coordinates in a common local metric plane."""
    track = trip["track"][COORDINATES].to_numpy(dtype=float)
    nodes = trip["nodes"][COORDINATES].to_numpy(dtype=float)
    reference = track if len(track) else nodes
    origin = reference.mean(axis=0) if len(reference) else np.zeros(2)
    scale = np.array([np.cos(np.radians(origin[1])), 1]) * np.pi / 180 * 6_371_008.8
    return (track - origin) * scale, (nodes - origin) * scale


class SegmentIndex:
    def __init__(self, nodes, node_xy, arcs):
        xy = pd.DataFrame(node_xy, index=nodes["node_id"])
        self.arc_ids = arcs["arc_id"].to_numpy(dtype=np.int64)
        starts = xy.loc[arcs["from_node_id"]].to_numpy()
        ends = xy.loc[arcs["to_node_id"]].to_numpy()
        self.vectors = ends - starts
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        # Project reverse arcs with identical arithmetic so distance ties stay exact.
        self.reversed = (arcs["from_node_id"] > arcs["to_node_id"]).to_numpy()
        self.starts = np.where(self.reversed[:, None], ends, starts)
        self.projection_vectors = np.where(self.reversed[:, None], -self.vectors, self.vectors)
        self.tree = cKDTree(self.starts + self.projection_vectors / 2)
        self.max_half_length = self.lengths.max(initial=0) / 2

    def project(self, point, indices):
        offsets = point - self.starts[indices]
        vectors = self.projection_vectors[indices]
        squared_lengths = self.lengths[indices] ** 2
        fractions = np.divide(
            np.einsum("ij,ij->i", offsets, vectors),
            squared_lengths,
            out=np.zeros(len(indices)),
            where=squared_lengths > 0,
        ).clip(0, 1)
        distances = np.linalg.norm(offsets - fractions[:, None] * vectors, axis=1)
        fractions = np.where(self.reversed[indices] & (squared_lengths > 0), 1 - fractions, fractions)
        return distances, fractions

    def nearby(self, point, radius_m):
        # Any segment within radius has a midpoint within radius + half its length.
        bound = np.nextafter(radius_m + self.max_half_length, np.inf)
        indices = np.asarray(self.tree.query_ball_point(point, bound), dtype=np.intp)
        distances, fractions = self.project(point, indices)
        order = np.lexsort((self.arc_ids[indices], distances))
        order = order[distances[order] <= radius_m]
        return self.arc_ids[indices[order]], distances[order], fractions[order]

    def nearest(self, point):
        _, seed = self.tree.query(point)
        distances, _ = self.project(point, np.array([seed]))
        return self.nearby(point, np.nextafter(distances[0], np.inf))


def generate_candidates(trip, *, radius_m=50.0, max_candidates=8, fallback=True):
    """Return up to max_candidates directed arcs per observation, nearest first.

    Observation IDs are zero-based track row positions. Ranks are one-based.
    If enabled, fallback returns the globally nearest segment on a radius miss.
    """
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be positive and finite")
    if not isinstance(max_candidates, (int, np.integer)) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    track_xy, node_xy = trip_coordinates(trip)
    rows = []
    if len(trip["arcs"]):
        index = SegmentIndex(trip["nodes"], node_xy, trip["arcs"])
        for observation_id, point in enumerate(track_xy):
            arc_ids, distances, fractions = index.nearby(point, radius_m)
            limit = max_candidates
            if not len(arc_ids) and fallback:
                arc_ids, distances, fractions = index.nearest(point)
                limit = 1
            matches = zip(arc_ids[:limit], distances[:limit], fractions[:limit])
            for rank, (arc_id, distance, fraction) in enumerate(matches, start=1):
                rows.append((observation_id, arc_id, rank, distance, fraction))
    return pd.DataFrame(rows, columns=list(CANDIDATE_DTYPES)).astype(CANDIDATE_DTYPES)
