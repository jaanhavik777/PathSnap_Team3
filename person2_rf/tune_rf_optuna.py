import json, time, joblib, numpy as np, pandas as pd, optuna
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from common import *

N_TRIALS, STEP = 40, 3            # STEP: tune on every 3rd GPS point for speed (final refit uses ALL data)
tr, te, feats = load_data()
sub = tr[tr.obs_idx % STEP == 0].reset_index(drop=True)
X, y, g = sub[feats], sub.y.values, sub.trip_id

SPACE = {"n_estimators": "int 100-500", "max_depth": "int 4-30", "min_samples_leaf": "int 1-30 (log)",
         "min_samples_split": "int 2-20", "max_features": "float 0.2-1.0",
         "max_samples": "float 0.2-1.0", "criterion": "gini|entropy", "class_weight": "balanced_subsample|None"}
json.dump({"space": SPACE, "objective": "mean GroupKFold(5) F1 on train trips", "sampler": "TPE (Bayesian)",
           "n_trials": N_TRIALS, "seed": SEED, "tuning_row_step": STEP}, open(f"{RES}/optuna_search_space.json", "w"), indent=2)

def params_from(t):
    return dict(n_estimators=t.suggest_int("n_estimators", 100, 500), max_depth=t.suggest_int("max_depth", 4, 30),
                min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 30, log=True),
                min_samples_split=t.suggest_int("min_samples_split", 2, 20),
                max_features=t.suggest_float("max_features", 0.2, 1.0),
                max_samples=t.suggest_float("max_samples", 0.2, 1.0),
                criterion=t.suggest_categorical("criterion", ["gini", "entropy"]),
                class_weight=t.suggest_categorical("class_weight", ["balanced_subsample", None]),
                n_jobs=-1, random_state=SEED)

def objective(t):
    p = params_from(t); sc = []
    for i, (a, b) in enumerate(GroupKFold(5).split(X, y, groups=g)):
        m = RandomForestClassifier(**p).fit(X.iloc[a], y[a])
        sc.append(f1_score(y[b], m.predict(X.iloc[b]), zero_division=0))
        t.report(float(np.mean(sc)), i)
        if t.should_prune(): raise optuna.TrialPruned()
    return float(np.mean(sc))

t0 = time.time()
study = optuna.create_study(direction="maximize", study_name="rf", storage=f"sqlite:///{ART}/optuna.db",
                            load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=8))
study.optimize(objective, n_trials=N_TRIALS)
opt_time = time.time() - t0
study.trials_dataframe().to_csv(f"{RES}/optuna_trials.csv", index=False)
best = {**study.best_params, "n_jobs": -1, "random_state": SEED}
json.dump({"best_params": best, "best_cv_f1_tuning_subset": study.best_value, "optimisation_time_s": opt_time},
          open(f"{RES}/optuna_best.json", "w"), indent=2)
vals = [t.value for t in study.trials if t.value is not None]
plt.plot(vals, "o", alpha=.5, label="trial"); plt.plot(np.maximum.accumulate(vals), label="best so far")
plt.xlabel("trial"); plt.ylabel("CV F1"); plt.legend(); plt.title("Optuna convergence"); plt.savefig(f"{RES}/optuna_convergence.png", dpi=150)
try:
    optuna.visualization.matplotlib.plot_param_importances(study); plt.tight_layout(); plt.savefig(f"{RES}/optuna_importances.png", dpi=150)
except Exception as e: print("importance plot skipped:", e)

# Retrain on FULL train with best params, same metrics/partitions as the untuned model
res, model = evaluate_model("RandomForest_tuned", "Ensemble (bagging)", best,
                            lambda: RandomForestClassifier(**best), tr, te, feats, outdir=f"{RES}/tuned")
u = json.load(open(f"{RES}/RandomForest.json"))
cmp = pd.DataFrame([{"model": n, "cv_f1": r["cv_f1_mean"], "test_top1": r["test"]["top1"], "test_f1": r["test"]["f1"],
                     "logloss": r["test"]["logloss"], "infer_ms_per_1k": r["infer_ms_per_1k_candidates"],
                     "continuity": r["test"]["continuity"], "train_s": r["train_time_s"]}
                    for n, r in [("untuned", u), ("tuned", res)]])
cmp.to_csv(f"{RES}/tuned_vs_untuned.csv", index=False); print(cmp.round(4).to_string(index=False))

import os; os.makedirs(f"{ART}/final_model", exist_ok=True)
joblib.dump(model, f"{ART}/final_model/model.joblib")
json.dump({"model": "RandomForest_tuned", "params": best, "features": feats, "threshold": 0.5,
           "candidates_per_obs": 10, "seed": SEED, "optimisation_time_s": opt_time},
          open(f"{ART}/final_model/config.json", "w"), indent=2, default=str)