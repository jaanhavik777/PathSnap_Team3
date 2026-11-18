import glob, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = [json.load(open(f)) for f in sorted(glob.glob("results/*.json"))]
rows = [dict(Model=r["name"], Family=r["family"], CV_F1=r["cv_f1_mean"], CV_F1_std=r["cv_f1_std"],
             Test_F1=r["test"]["f1"], Top1=r["test"]["top1"], Continuity=r["test"]["continuity"],
             LogLoss=r["test"]["logloss"], MissRate=r["test"]["miss_rate"],
             Train_s=r["train_time_s"], Infer_ms_per_1k=r["infer_ms_per_1k_candidates"],
             Train_F1=r["train"]["f1"], Config=json.dumps(r["config"], default=str)[:80]) for r in R]
df = pd.DataFrame(rows).sort_values("Top1", ascending=False)
df.to_csv("results/model_comparison.csv", index=False)
open("results/model_comparison.md", "w").write(df.drop(columns="Config").round(4).fillna("N/A").to_markdown(index=False))
print(df.drop(columns="Config").round(4).to_string(index=False))

def bar(cols, title, fn, err=None):
    ax = df.set_index("Model")[cols].plot.bar(figsize=(8, 4), yerr=err, capsize=3, title=title, rot=30)
    plt.tight_layout(); plt.savefig(f"results/{fn}", dpi=150); plt.close()
bar(["CV_F1"], "Cross-validation F1 (80 train trips, GroupKFold)", "cmp_cv_f1.png", df.set_index("Model")["CV_F1_std"].values)
bar(["Test_F1", "Top1"], "Student-test F1 and Top-1 accuracy", "cmp_test.png")
if df.LogLoss.notna().any(): bar(["LogLoss"], "Student-test Log Loss (where applicable)", "cmp_logloss.png")
bar(["Train_s"], "Training time (s)", "cmp_train_time.png")
bar(["Infer_ms_per_1k"], "Inference time (ms per 1k candidates)", "cmp_infer_time.png")
bar(["Train_F1", "Test_F1"], "Train vs test F1 (overfitting check)", "cmp_overfit.png")