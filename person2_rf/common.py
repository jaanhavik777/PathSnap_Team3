import os, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (f1_score, log_loss, precision_score, recall_score,
                             accuracy_score, confusion_matrix)
SEED = 42
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ART, RES = str(HERE / "artifacts"), str(HERE / "results")
os.makedirs(ART, exist_ok=True); os.makedirs(RES, exist_ok=True)
META = ["trip_id", "partition", "obs_idx", "arc_id", "y"]
ARCS = {}

def _read(part):
    df = pd.read_csv(ROOT / "data" / "processed" / f"{part}.csv", dtype={"trip_id": str})
    df = df.rename(columns={"observation_id": "obs_idx", "label": "y"})
    df["y"] = df["y"].fillna(-1).astype(int)          # -1 = unlabeled
    df["partition"] = part
    stems = df["trip_id"].copy()
    df["trip_id"] = part + "/" + stems
    for stem in stems.unique():
        f = ROOT / "Student_folder" / part / stem / f"{stem}.arcs"
        ARCS[f"{part}/{stem}"] = pd.read_csv(f, sep=r"\s+", header=None).values.astype(int)
    return df

def load_data():
    """Train + student_test only. Evaluator data is never loaded.
    Train keeps labeled observations only. Test keeps every row, because route
    continuity needs all points; metrics() scores only the labeled ones."""
    tr, te = _read("train"), _read("student_test")
    feats = [c for c in tr.columns if c not in META]
    tr = tr[tr.y >= 0].reset_index(drop=True)
    print("features:", feats)
    print("train labeled observations:", tr.groupby("trip_id").obs_idx.nunique().sum())
    print("test labeled observations:", te[te.y >= 0].groupby("trip_id").obs_idx.nunique().sum(),
          "of", te.groupby("trip_id").obs_idx.nunique().sum())
    return tr, te.reset_index(drop=True), feats

def get_scores(model, X):
    if hasattr(model, "predict_proba"): return model.predict_proba(X)[:, 1], True
    return model.decision_function(X), False

def top1_pred(df, s):
    d = pd.DataFrame({"trip_id": df.trip_id.values, "obs_idx": df.obs_idx.values,
                      "arc_id": df.arc_id.values, "y": df.y.values, "s": s})
    return d.loc[d.groupby(["trip_id", "obs_idx"])["s"].idxmax()].sort_values(["trip_id", "obs_idx"])

def route_continuity(pred):
    """Share of consecutive predicted arcs that are identical or directly connected."""
    ok = tot = 0
    for tid, g in pred.groupby("trip_id"):
        a = g.arc_id.values; arcs = ARCS[tid]; p, n = a[:-1], a[1:]
        ok += ((p == n) | (arcs[p, 1] == arcs[n, 0])).sum(); tot += len(p)
    return float(ok / max(tot, 1))

def metrics(df, s, is_proba):
    pred_all = top1_pred(df, s)                       # every observation, for continuity
    lab = df.y.values >= 0
    d, s = df[lab], s[lab]
    y = d.y.values; yhat = (s >= (0.5 if is_proba else 0.0)).astype(int)
    pred = top1_pred(d, s)
    m = dict(f1=f1_score(y, yhat, zero_division=0), f1_macro=f1_score(y, yhat, average="macro", zero_division=0),
             precision=precision_score(y, yhat, zero_division=0), recall=recall_score(y, yhat, zero_division=0),
             accuracy=accuracy_score(y, yhat), top1=float(pred.y.mean()),
             labeled_obs_frac=float(len(pred) / len(pred_all)), miss_rate=None,
             continuity=route_continuity(pred_all),
             logloss=float(log_loss(y, np.clip(s, 1e-7, 1 - 1e-7))) if is_proba else None,
             cm=confusion_matrix(y, yhat, labels=[0, 1]).tolist())
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in m.items()}

def evaluate_model(name, family, config, make_model, train, test, feats,
                   cv_splits=5, outdir=RES, save=True):
    """make_model: zero-arg function returning a NEW unfitted estimator."""
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
               cv_f1_macro_mean=float(np.mean([x["f1_macro"] for x in folds])),
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