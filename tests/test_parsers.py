from pathlib import Path

import pytest

from src.data.parsers import load_trip, read_nodes, read_tracks


@pytest.mark.parametrize(
    ("reader", "row", "message"),
    [(read_nodes, "0 91", "Coordinates out of range"),
     (read_tracks, "0 0 inf", "Non-finite values")],
)
def test_invalid_numeric_values_are_rejected(tmp_path, reader, row, message):
    source = tmp_path / "invalid"
    source.write_text(row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        reader(source)


def test_tiny_trip(tmp_path):
    folder = tmp_path / "00000001"
    folder.mkdir()
    files = {
        "track": "8.5 49.8 0\n",
        "nodes": "8.5 49.8\n8.6 49.8\n",
        "arcs": "0 1\n",
        "route": "0\n",
    }
    for suffix, content in files.items():
        (folder / f"{folder.name}.{suffix}").write_text(content, encoding="utf-8")
    trip = load_trip(folder)
    assert trip["track"]["timestamp"].tolist() == [0]
    assert trip["nodes"]["node_id"].tolist() == [0, 1]
    assert trip["arcs"]["arc_id"].tolist() == [0]
    assert trip["route"]["arc_id"].tolist() == [0]
    assert load_trip(folder, include_route=False)["route"] is None


@pytest.mark.skipif(
    not Path("data/raw/train/00000001/00000001.track").is_file(),
    reason="Private dataset supplied locally",
)
def test_real_trip():
    trip=load_trip('data/raw/train/00000001')
    assert len(trip['track'])>0
    assert trip['nodes'].node_id.iloc[0]==0
    assert trip['arcs'].arc_id.iloc[0]==0
    assert trip['route'].arc_id.max()<len(trip['arcs'])
