"""PathSnap raw data readers. Row indices are node and arc identifiers."""
from pathlib import Path
import numpy as np
import pandas as pd

def _read(path, names, dtype=float):
    df=pd.read_csv(path,sep=r"\s+",header=None,names=names,dtype=dtype)
    if df.isna().any().any():
        raise ValueError(f"Missing values in {path}")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f"Non-finite values in {path}")
    return df


def _read_coordinates(path, names):
    df = _read(path, names)
    if not (df.longitude.abs().le(180) & df.latitude.abs().le(90)).all():
        raise ValueError(f"Coordinates out of range in {path}")
    return df


def read_tracks(path):
    return _read_coordinates(path, ["longitude", "latitude", "timestamp"])

def read_nodes(path):
    df = _read_coordinates(path, ["longitude", "latitude"])
    df.insert(0,"node_id",np.arange(len(df),dtype=np.int64))
    return df

def read_arcs(path):
    df=_read(path,["from_node_id","to_node_id"],dtype=np.int64)
    df.insert(0,"arc_id",np.arange(len(df),dtype=np.int64))
    return df

def read_routes(path):
    return _read(path,["arc_id"],dtype=np.int64)

def load_trip(folder, include_route=True):
    folder=Path(folder); stem=folder.name
    track=read_tracks(folder/f"{stem}.track")
    nodes=read_nodes(folder/f"{stem}.nodes")
    arcs=read_arcs(folder/f"{stem}.arcs")
    if len(arcs) and (arcs[["from_node_id","to_node_id"]].to_numpy().min()<0 or arcs[["from_node_id","to_node_id"]].to_numpy().max()>=len(nodes)):
        raise ValueError(f"Invalid node references in {folder}")
    route=read_routes(folder/f"{stem}.route") if include_route and (folder/f"{stem}.route").exists() else None
    if route is not None and len(route) and (route.arc_id.min()<0 or route.arc_id.max()>=len(arcs)):
        raise ValueError(f"Invalid arc references in {folder}")
    return {"trip_id":stem,"track":track,"nodes":nodes,"arcs":arcs,"route":route}
