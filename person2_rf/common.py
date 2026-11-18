import os, json, time, pickle
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (f1_score, log_loss, precision_score, recall_score,
                             accuracy_score, confusion_matrix)
SEED = 42
ART, RES = "artifacts", "results"
os.makedirs(ART, exist_ok=True); os.makedirs(RES, exist_ok=True)
META = ["trip_id", "partition", "obs_idx", "arc_id", "y"]
ARCS = pickle.load(open(f"{ART}/arcs_by_trip.pkl", "rb"))

def load_data(exclude_bad=False, miss_thresh=0.2):
    """Train + student_test only. Evaluator data is never loaded.
    Default: use ALL train trips. exclude_bad=True drops train trips whose
    candidate miss rate > miss_thresh (team decision pending)."""
    df = pd.read_parquet(f"{ART}/candidates.parquet")
    feats = [c for c in df.columns if c not in META]
    tr = df[df.partition == "train"]
    if exclude_bad:
        miss = tr.groupby(["trip_id", "obs_idx"]).y.max().eq(0).groupby("trip_id").mean()
        bad = miss[miss > miss_thresh]
        bad.rename("miss_rate").to_csv(f"{RES}/excluded_train_trips.csv")
        print("Excluded train trips:", dict(bad.round(3)))
        tr = tr[~tr.trip_id.isin(bad.index)]
    tr = tr.reset_index(drop=True)
    te = df[df.partition == "student_test"].reset_index(drop=True)
    return tr, te, feats

def get_scores(model, X):
    if hasattr(model, "predict_proba"): return model.predict_proba(X)[:, 1], True
    return model.decision_function(X), False

def top1_pred(df, s):
    d = pd.DataFrame({"trip_id": df.trip_id.values, "obs_idx": df.obs_idx.values,
                      "arc_id": df.arc_id.values, "y": df.y.values, "s": s})
    return d.loc[d.groupby(["trip_id", "obs_idx"])["s"].idxmax()].sort_values(["trip_id", "obs_idx"])

def miss_rate(df):
    return float((df.groupby(["trip_id", "obs_idx"]).y.max() == 0).mean())

def route_continuity(pred):
    """Share of consecutive predicted arcs that are identical or directly connected (to_node == from_node)."""
    ok = tot = 0
    for tid, g in pred.groupby("trip_id"):
        a = g.arc_id.values; arcs = ARCS[tid]; p, n = a[:-1], a[1:]
        ok += ((p == n) | (arcs[p, 1] == arcs[n, 0])).sum(); tot += len(p)
    return float(ok / max(tot, 1))

def metrics(df, s, is_proba):
    y = df.y.values; yhat = (s >= (0.5 if is_proba else 0.0)).astype(int)
    pred = top1_pred(df, s)
    m = dict(f1=f1_score(y, yhat, zero_division=0), f1_macro=f1_score(y, yhat, average="macro", zero_division=0),
             precision=precision_score(y, yhat, zero_division=0), recall=recall_score(y, yhat, zero_division=0),
             accuracy=accuracy_score(y, yhat), top1=float(pred.y.mean()),
             miss_rate=miss_rate(df), continuity=route_continuity(pred),
             logloss=float(log_loss(y, np.clip(s, 1e-7, 1 - 1e-7))) if is_proba else None,
             cm=confusion_matrix(y, yhat, labels=[0, 1]).tolist())
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in m.items()}

def evaluate_model(name, family, config, make_model, train, test, feats,
                   cv_splits=5, outdir=RES, save=True):
    """make_model: zero-arg function returning a NEW unfitted estimator (use a Pipeline for scaling)."""
    Xtr, ytr = train[feats], train.y.values
    folds = []
    for f, (a, b) in enumerate(GroupKFold(cv_splits).split(Xtr, ytr, groups=train.trip_id)):
        m = make_model().fit(Xtr.iloc[a], ytr[a])
        s, p = get_scores(m, Xtr.iloc[b]); folds.append(metrics(train.iloc[b], s, p))
    t0 = time.time(); model = make_model().fit(Xtr, ytr); train_time = time.time() - t0
    s_tr, p = get_scores(model, Xtr); tr_m = metrics(train, s_tr, p)
    t0 = time.time(); s_te, p = get_scores(model, test[feats]); inf = time.time() - t0
    te_m = metrics(test, s_te, p)
    ll = [x["logloss"] for x in folds]
    res = dict(name=name, family=family, config=config, seed=SEED, n_features=len(feats),
               cv_f1_mean=float(np.mean([x["f1"] for x in folds])), cv_f1_std=float(np.std([x["f1"] for x in folds])),
               cv_top1_mean=float(np.mean([x["top1"] for x in folds])),
               cv_logloss_mean=None if ll[0] is None else float(np.mean(ll)),
               train=tr_m, test=te_m, train_time_s=train_time,
               infer_ms_per_1k_candidates=1000 * inf / len(test) * 1000, folds=folds)
    if save:
        os.makedirs(outdir, exist_ok=True)
        json.dump(res, open(f"{outdir}/{name}.json", "w"), indent=2, default=str)
    print(f"[{name}] CV F1 {res['cv_f1_mean']:.4f}±{res['cv_f1_std']:.4f} | test F1 {te_m['f1']:.4f} "
          f"| test top1 {te_m['top1']:.4f} | logloss {te_m['logloss']} | cont {te_m['continuity']:.4f} "
          f"| train {train_time:.1f}s")
    return res, model




#USE THIS IF CONFIRMED WITH THAVITHA

# def load_data(miss_thresh=0.2):
#     """Train + student_test only. Evaluator data is never loaded.
#     Rule: drop TRAIN trips whose reference route misses > miss_thresh of GPS points."""
#     df = pd.read_parquet(f"{ART}/candidates.parquet")
#     feats = [c for c in df.columns if c not in META]
#     tr = df[df.partition == "train"]
#     miss = (tr.groupby(["trip_id", "obs_idx"]).y.max().eq(0)
#               .groupby("trip_id").mean())
#     bad = miss[miss > miss_thresh]
#     if len(bad):
#         bad.rename("miss_rate").to_csv(f"{RES}/excluded_train_trips.csv")
#         print("Excluded train trips (miss > %.0f%%):" % (miss_thresh * 100), dict(bad.round(3)))
#     tr = tr[~tr.trip_id.isin(bad.index)].reset_index(drop=True)
#     te = df[df.partition == "student_test"].reset_index(drop=True)
#     return tr, te, feats