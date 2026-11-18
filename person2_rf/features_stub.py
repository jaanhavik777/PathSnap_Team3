import os, sys, glob, pickle
import numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

ROOT = sys.argv[1] if len(sys.argv) > 1 else "Student_folder"
K = 10
os.makedirs("artifacts", exist_ok=True)

def partition_of(rel):
    r = rel.lower()
    if "evaluator" in r or "eval" in r: return "evaluator_test"
    if "student" in r: return "student_test"
    if "train" in r: return "train"
    return "unknown"

def sibling(track, ext):
    d = os.path.dirname(track); stem = os.path.splitext(os.path.basename(track))[0]
    f = os.path.join(d, stem + ext)
    if os.path.exists(f): return f
    c = glob.glob(os.path.join(d, "*" + ext))
    return c[0] if len(c) == 1 else None

def rd(f): return pd.read_csv(f, sep=r"\s+", header=None).values

def seg_dist(P, A, B):
    AB = B - A; L2 = np.maximum((AB ** 2).sum(-1), 1e-9)
    t = np.clip(((P - A) * AB).sum(-1) / L2, 0, 1)
    C = A + t[..., None] * AB
    return np.linalg.norm(P - C, axis=-1), t

def label_obs(P, A, B, route):
    """Monotonic alignment of GPS points to the ordered route arcs (DP)."""
    D, _ = seg_dist(P[:, None, :], A[route][None], B[route][None])
    n, m = D.shape; idx = np.arange(m)
    dp = np.empty((n, m)); bp = np.zeros((n, m), int); dp[0] = D[0]
    for i in range(1, n):
        prev = dp[i - 1]; cm = np.minimum.accumulate(prev)
        bp[i] = np.maximum.accumulate(np.where(prev == cm, idx, 0))
        dp[i] = D[i] + cm
    lab = np.empty(n, int); j = int(dp[-1].argmin())
    for i in range(n - 1, -1, -1):
        lab[i] = j; j = bp[i][j]
    return route[lab], D[np.arange(n), lab]

def process(track_f, trip_id, part):
    tr = rd(track_f); tr = tr[np.isfinite(tr).all(1)]
    nodes = rd(sibling(track_f, ".nodes")); arcs = rd(sibling(track_f, ".arcs")).astype(int)
    rf = sibling(track_f, ".route")
    route = rd(rf).astype(int).ravel() if rf else None
    zone = int((nodes[:, 0].mean() + 180) // 6) + 1
    epsg = (32600 if nodes[:, 1].mean() >= 0 else 32700) + zone
    tf = Transformer.from_crs(4326, epsg, always_xy=True)   # lon, lat order
    P = np.c_[tf.transform(tr[:, 0], tr[:, 1])]; N = np.c_[tf.transform(nodes[:, 0], nodes[:, 1])]
    A, B = N[arcs[:, 0]], N[arcs[:, 1]]
    n = len(P); k = min(K, len(arcs)); kq = min(60, len(arcs))
    _, nb = cKDTree((A + B) / 2).query(P, k=kq)
    nb = nb.reshape(n, kq)
    D, T = seg_dist(P[:, None, :], A[nb], B[nb])
    o = np.argsort(D, 1)[:, :k]
    nb, D, T = [np.take_along_axis(x, o, 1) for x in (nb, D, T)]
    d = np.zeros_like(P); d[1:-1] = P[2:] - P[:-2]; d[0] = P[1] - P[0]; d[-1] = P[-1] - P[-2]
    dt = np.r_[np.diff(tr[:, 2]), 1.0]; dt = np.clip(dt, 1e-3, None)
    spd = np.r_[np.linalg.norm(np.diff(P, axis=0), axis=1), 0] / dt
    av = B[nb] - A[nb]; alen = np.linalg.norm(av, axis=-1)
    cos = (d[:, None, :] * av).sum(-1) / (np.linalg.norm(d, axis=1)[:, None] * alen + 1e-9)
    rep = lambda v: np.repeat(v, k)
    df = pd.DataFrame({
        "trip_id": trip_id, "partition": part, "obs_idx": np.repeat(np.arange(n), k),
        "arc_id": nb.ravel(), "dist": D.ravel(), "dist_rank": np.tile(np.arange(k), n),
        "dist_gap": (D - D[:, [0]]).ravel(), "dist_ratio": (D / (D[:, [0]] + 1)).ravel(),
        "t_along": T.ravel(), "arc_len": alen.ravel(), "heading_cos": cos.ravel(),
        "speed": rep(spd), "dt": rep(dt)})
    if route is not None:
        lab, ld = label_obs(P, A, B, route)
        df["y"] = (df.arc_id.values == np.repeat(lab, k)).astype(int)
    else:
        df["y"] = -1
    return df, arcs

if __name__ == "__main__":
    tracks = sorted(glob.glob(os.path.join(ROOT, "**", "*.track"), recursive=True))
    print("tracks found:", len(tracks))
    dfs, arcs_by = [], {}
    for t in tracks:
        rel = os.path.relpath(t, ROOT); tid = rel.replace(os.sep, "/")
        df, arcs = process(t, tid, partition_of(rel)); dfs.append(df); arcs_by[tid] = arcs
    out = pd.concat(dfs, ignore_index=True)
    print(out.groupby("partition").trip_id.nunique())     # CHECK: 80 / 10 (/10)
    out.to_parquet("artifacts/candidates.parquet")
    pickle.dump(arcs_by, open("artifacts/arcs_by_trip.pkl", "wb"))
    print("rows:", len(out), "| candidate miss rate:",
          float((out.groupby(["trip_id", "obs_idx"]).y.max() == 0).mean()))