import sys
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from common import *

tr, te, feats = load_data()
BASE = dict(n_estimators=100, max_depth=20, min_samples_leaf=10, max_features="sqrt",
            max_samples=0.3, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
mk = lambda **kw: (lambda: RandomForestClassifier(**{**BASE, **kw}))

# Baseline: goes into the 5-model comparison
res, model = evaluate_model("RandomForest", "Ensemble (bagging)", BASE, mk(), tr, te, feats)
joblib.dump(model, f"{ART}/rf_untuned.joblib")

# Feature importance
imp = pd.Series(model.feature_importances_, index=feats).sort_values()
imp.to_csv(f"{RES}/rf_feature_importance.csv")
imp.plot.barh(figsize=(6, 4), title="RF feature importance (train)"); plt.tight_layout()
plt.savefig(f"{RES}/rf_feature_importance.png", dpi=150); plt.close()

# Confusion matrix (student-test, candidate level)
cm = np.array(res["test"]["cm"])
ConfusionMatrixDisplay(cm, display_labels=["not matched", "matched"]).plot(cmap="Blues", values_format="d")
plt.title("RF confusion matrix (student-test, candidate level)")
plt.savefig(f"{RES}/rf_confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()

# Hyperparameter experiments: only when you pass --variants (each one runs a full 5-fold CV)
variants = {"depth8": dict(max_depth=8), "depth16": dict(max_depth=16), "leaf1": dict(min_samples_leaf=1),
            "leaf20": dict(min_samples_leaf=20), "no_classweight": dict(class_weight=None),
            "maxfeat0.5": dict(max_features=0.5), "trees200": dict(n_estimators=200)}
if "--variants" in sys.argv:
    rows = []
    for n, kw in variants.items():
        r, _ = evaluate_model(f"RF_{n}", "Ensemble (bagging)", {**BASE, **kw}, mk(**kw), tr, te, feats,
                              outdir=f"{RES}/rf_variants")
        rows.append(dict(variant=n, cv_f1=r["cv_f1_mean"], train_f1=r["train"]["f1"], test_f1=r["test"]["f1"],
                         test_top1=r["test"]["top1"], logloss=r["test"]["logloss"], train_s=r["train_time_s"]))
    pd.DataFrame(rows).to_csv(f"{RES}/rf_variants.csv", index=False)