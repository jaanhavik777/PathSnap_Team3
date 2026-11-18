import pandas as pd

from src.data.candidates import generate_candidates
from src.data.labels import label_candidates


def test_labels_are_nullable_and_require_positive_coverage(directed_trip):
    candidates = generate_candidates(directed_trip)
    labels = label_candidates(directed_trip, candidates)
    assert labels.loc[labels.arc_id == 8, "label"].eq(1).all()
    assert labels.loc[labels.arc_id != 8, "label"].eq(0).all()
    assert label_candidates(directed_trip, candidates[candidates.arc_id != 8]).label.isna().all()
    assert "label" not in candidates.columns


def test_ambiguous_repeated_and_missing_routes_abstain(directed_trip):
    candidates = generate_candidates(directed_trip)
    for route in ([8, 4], [8, 8], None):
        directed_trip["route"] = None if route is None else pd.DataFrame({"arc_id": route})
        assert label_candidates(directed_trip, candidates).label.isna().all()


def test_chronological_conflict_abstains(directed_trip):
    directed_trip["track"].loc[19, ["longitude", "latitude"]] = [0, 0.0002]
    directed_trip["route"] = pd.DataFrame({"arc_id": [6, 8]})
    candidates = generate_candidates(directed_trip)
    assert label_candidates(directed_trip, candidates).label.isna().all()
