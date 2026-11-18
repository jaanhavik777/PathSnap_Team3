"""Trip reader and conservative route-derived candidate feature builder.

Route IDs are sequence labels, not GPS-row labels. We associate a route arc to
its nearest GPS observation by projecting the observation onto the segment,
then accept only an unambiguous nearest arc within a configurable radius.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FEATURE_COLUMNS = ["distance_m", "heading_alignment", "speed_mps", "segment_length_m", "progress", "log_dt"]


@dataclass
class Trip:
    trip_id: str
    gps: pd.DataFrame
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray


def read_trip(folder: str | Path, include_route: bool = True) -> Trip:
    folder = Path(folder)
    trip_id = folder.name
    def load(name: str, columns: int) -> np.ndarray:
        path = folder / f"{trip_id}.{name}"
        if not path.exists():
            raise FileNotFoundError(path)
        arr = np.loadtxt(path, dtype=float, ndmin=2)
        if arr.shape[1] != columns:
            raise ValueError(f"{path} has {arr.shape[1]} columns, expected {columns}")
        return arr
    gps = pd.DataFrame(load("track", 3), columns=["lon", "lat", "timestamp"])
    gps = gps.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    gps = gps[(gps.lon.between(-180, 180)) & (gps.lat.between(-90, 90))].reset_index(drop=True)
    nodes, arcs = load("nodes", 2), load("arcs", 2).astype(int)
    route_path = folder / f"{trip_id}.route"
    route = np.loadtxt(route_path, dtype=int, ndmin=1) if include_route and route_path.exists() else np.array([], dtype=int)
    return Trip(trip_id, gps, nodes, arcs, route)


def _xy(lon: np.ndarray, lat: np.ndarray, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    """Local equirectangular metres; suitable for short urban trips."""
    return lon * (111320.0 * math.cos(math.radians(lat0))), lat * 110574.0


def candidate_features(trip: Trip, candidate_count: int = 8, label_radius_m: float = 35.0,
                       ambiguity_margin_m: float = 2.0) -> pd.DataFrame:
    if candidate_count < 1 or label_radius_m <= 0:
        raise ValueError("candidate_count and label_radius_m must be positive")
    gps, arcs, nodes = trip.gps, trip.arcs, trip.nodes
    if len(gps) < 1 or len(arcs) < 1 or len(nodes) < 1:
        return pd.DataFrame()
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError("arc endpoint refers to a missing node")
    lat0 = float(gps.lat.mean())
    gx, gy = _xy(gps.lon.to_numpy(), gps.lat.to_numpy(), lat0)
    nx, ny = _xy(nodes[:, 0], nodes[:, 1], lat0)
    starts, ends = arcs[:, 0], arcs[:, 1]
    ax, ay = nx[starts], ny[starts]
    bx, by = nx[ends], ny[ends]
    vx, vy = bx-ax, by-ay
    lengths = np.hypot(vx, vy)
    safe_len2 = np.maximum(lengths**2, 1e-9)
    k = min(candidate_count, len(arcs))
    midpoint_tree = cKDTree(np.column_stack(((ax+bx)/2, (ay+by)/2)))
    # Restrict full-road-network quadratic workspace per observation.
    prev_heading = np.zeros(len(gps))
    speed = np.zeros(len(gps))
    dt = np.ones(len(gps))
    if len(gps) > 1:
        ddx, ddy = np.diff(gx), np.diff(gy)
        dts = np.diff(gps.timestamp.to_numpy())
        dt[1:] = np.maximum(dts, 1e-3)
        prev_heading[1:] = np.arctan2(ddx, ddy)
        speed[1:] = np.hypot(ddx, ddy) / dt[1:]
    rows = []
    ref = set(map(int, trip.route.tolist()))
    for i in range(len(gps)):
        # Spatially shortlist segment midpoints, then rank exact segment distance.
        pool_n = min(len(arcs), max(k*16, 64))
        _, pool = midpoint_tree.query([gx[i], gy[i]], k=pool_n)
        pool = np.atleast_1d(pool).astype(int)
        dx, dy = gx[i]-ax[pool], gy[i]-ay[pool]
        tp = np.clip((dx*vx[pool]+dy*vy[pool])/safe_len2[pool], 0, 1)
        dp = np.hypot(gx[i]-(ax[pool]+tp*vx[pool]), gy[i]-(ay[pool]+tp*vy[pool]))
        order = np.argsort(dp)[:k]
        ids = pool[order]
        distances = {int(a): float(d) for a,d in zip(pool,dp)}
        progress = {int(a): float(t0) for a,t0 in zip(pool,tp)}
        # A route label is assigned only when that route arc is itself in the
        # candidate list and nearest route arc is separated from alternatives.
        label = -1
        if ref:
            route_ids = np.fromiter((a for a in ids if int(a) in ref), dtype=int)
            if len(route_ids):
                ranked = sorted(map(int,route_ids),key=lambda a:distances[a])
                d0 = distances[ranked[0]]
                d1 = distances[ranked[1]] if len(ranked) > 1 else float("inf")
                if d0 <= label_radius_m and d1-d0 >= ambiguity_margin_m:
                    label = int(ranked[0])
        for arc_id in ids:
            align = math.cos(prev_heading[i] - math.atan2(vx[arc_id], vy[arc_id])) if i else 0.0
            rows.append({"trip_id": trip.trip_id, "observation_id": i, "candidate_arc_id": int(arc_id),
                         "distance_m": distances[int(arc_id)], "heading_alignment": align,
                         "speed_mps": float(min(speed[i], 80)), "segment_length_m": float(lengths[arc_id]),
                         "progress": progress[int(arc_id)], "log_dt": float(math.log1p(dt[i])),
                         "label": -1 if label < 0 else int(arc_id == label), "reference_arc_id": label})
    return pd.DataFrame(rows)


def training_frame(data_root: str | Path, candidate_count: int = 8) -> pd.DataFrame:
    root = Path(data_root)
    frames = [candidate_features(read_trip(p), candidate_count=candidate_count)
              for p in sorted((root / "train").iterdir()) if p.is_dir()]
    frames = [f[f.reference_arc_id >= 0] for f in frames if not f.empty]
    frames = [f for f in frames if not f.empty and f.label.eq(1).any()]
    if not frames:
        raise ValueError("No unambiguous route-derived candidate labels; review association policy")
    return pd.concat(frames, ignore_index=True)
