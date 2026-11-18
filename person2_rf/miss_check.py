import pandas as pd
d = pd.read_parquet("artifacts/candidates.parquet")
o = d.groupby(["partition", "trip_id", "obs_idx"]).y.max().eq(0).rename("miss").reset_index()
t = o.groupby(["partition", "trip_id"]).miss.agg(["mean", "size"]).sort_values("mean", ascending=False)
print(t.head(15))
print("\nmiss rate by partition:\n", o.groupby("partition").miss.mean())
print("\ntrips with miss > 20%:", (t["mean"] > 0.2).sum(), "of", len(t))
keep = t[t["mean"] <= 0.2].index.get_level_values(1)
print("miss rate excluding those trips:", o[o.trip_id.isin(keep)].miss.mean())