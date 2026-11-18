import pytest

from src.data.candidates import generate_candidates


def test_segment_interiors_direction_and_ranking(directed_trip):
    candidates = generate_candidates(directed_trip, radius_m=30)
    first = candidates[candidates.observation_id == 0]
    assert first.arc_id.tolist() == [4, 8, 6]
    assert first.candidate_rank.tolist() == [1, 2, 3]
    assert first.distance_m.iloc[:2].tolist() == pytest.approx([0, 0], abs=1e-9)
    assert first.projection_fraction.tolist() == pytest.approx([1 / 6, 5 / 6, 0])
    assert set(candidates.observation_id) == {0, 1}


def test_exact_nearest_fallback_can_find_long_segment(directed_trip):
    directed_trip["track"]["latitude"] = 0.00001
    assert generate_candidates(directed_trip, radius_m=0.5, fallback=False).empty
    result = generate_candidates(directed_trip, radius_m=0.5)
    assert result.groupby("observation_id").size().tolist() == [1, 1]
    assert result.arc_id.isin([4, 8]).all()
    assert result.distance_m.tolist() == pytest.approx([1.112, 1.112], abs=0.001)


def test_empty_network_and_candidate_cap(directed_trip):
    capped = generate_candidates(directed_trip, max_candidates=1)
    assert capped.groupby("observation_id").size().tolist() == [1, 1]
    directed_trip["arcs"] = directed_trip["arcs"].iloc[:0]
    empty = generate_candidates(directed_trip)
    assert empty.empty
    assert empty.dtypes.equals(capped.dtypes)
