import pandas as pd
import pytest


@pytest.fixture
def directed_trip():
    return {
        "trip_id": "synthetic",
        "track": pd.DataFrame({
            "longitude": [0.0, 0.0005], "latitude": [0.0, 0.0], "timestamp": [1, 3],
        }, index=[11, 19]),
        "nodes": pd.DataFrame({
            "node_id": [10, 20, 30],
            "longitude": [-0.05, 0.01, 0.0],
            "latitude": [0.0, 0.0, 0.0002],
        }),
        "arcs": pd.DataFrame({
            "arc_id": [8, 4, 6], "from_node_id": [10, 20, 30], "to_node_id": [20, 10, 30],
        }),
        "route": pd.DataFrame({"arc_id": [8]}),
    }
