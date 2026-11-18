from src.data.parsers import load_trip
def test_real_trip():
    trip=load_trip('data/raw/train/00000001')
    assert len(trip['track'])>0
    assert trip['nodes'].node_id.iloc[0]==0
    assert trip['arcs'].arc_id.iloc[0]==0
    assert trip['route'].arc_id.max()<len(trip['arcs'])
