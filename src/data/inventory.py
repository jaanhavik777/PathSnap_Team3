from pathlib import Path
import numpy as np
import pandas as pd
from .parsers import load_trip

def inventory(data_root):
    records=[]
    for partition in ("train","student_test"):
        for folder in sorted((Path(data_root)/partition).glob("*")):
            if not folder.is_dir(): continue
            t=load_trip(folder); gps=t["track"]; nodes=t["nodes"]; arcs=t["arcs"]; route=t["route"]
            delta=gps.timestamp.diff().dropna()
            records.append({"partition":partition,"trip_id":t["trip_id"],
               "gps_points":len(gps),"nodes":len(nodes),"arcs":len(arcs),
               "route_arcs":len(route) if route is not None else 0,
               "missing_gps":int(gps.isna().any(axis=1).sum()),
               "invalid_coordinates":int(((gps.longitude.abs()>180)|(gps.latitude.abs()>90)).sum()),
               "nonpositive_time_steps":int((delta<=0).sum()),
               "gaps_over_5s":int((delta>5).sum()),
               "max_gap_s":float(delta.max()) if len(delta) else 0,
               "min_lon":float(gps.longitude.min()),"max_lon":float(gps.longitude.max()),
               "min_lat":float(gps.latitude.min()),"max_lat":float(gps.latitude.max()),
               "isolated_nodes":int(len(nodes)-len(np.unique(arcs[["from_node_id","to_node_id"]].to_numpy())))})
    return pd.DataFrame(records)
