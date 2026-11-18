"""Build a candidate feature table from a parsed trip."""

from src.data.candidates import generate_candidates
from src.data.features import build_features
from src.data.labels import label_candidates


def build_trip_dataset(trip, *, radius_m=50.0, max_candidates=8, include_labels=False):
    candidates = generate_candidates(trip, radius_m=radius_m, max_candidates=max_candidates)
    table = build_features(trip, candidates)
    if include_labels:
        table = label_candidates(trip, table)
    table.insert(0, "trip_id", trip["trip_id"])
    return table
