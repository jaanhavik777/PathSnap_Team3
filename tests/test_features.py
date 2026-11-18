import pytest

from src.data.build_dataset import build_trip_dataset
from src.data.features import FEATURE_COLUMNS


def test_motion_and_directed_heading(directed_trip):
    result = build_trip_dataset(directed_trip)
    first = result[result.observation_id == 0]
    second = result[result.observation_id == 1].set_index("arc_id")
    assert set(FEATURE_COLUMNS).issubset(result.columns)
    assert first.speed_mps.isna().all()
    assert second.loc[8, "speed_mps"] == pytest.approx(27.799, abs=0.002)
    assert second.loc[8, "heading_error_deg"] == pytest.approx(0)
    assert second.loc[4, "heading_error_deg"] == pytest.approx(180)
    assert second.loc[8, "from_node_in_degree"] == 1
    assert second.loc[8, "to_node_out_degree"] == 1
    assert "label" not in result.columns


def test_empty_features(directed_trip):
    directed_trip["track"] = directed_trip["track"].iloc[:0]
    result = build_trip_dataset(directed_trip)
    assert result.empty
    assert set(FEATURE_COLUMNS).issubset(result.columns)
