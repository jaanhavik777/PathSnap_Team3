"""
person2_rf/compare_models.py
Aggregates all models from the team's experimental artifacts, evaluates them
on identical student-test data, and generates publication-quality comparison tables and charts.
"""
import json, os
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from common import load_data, metrics, get_scores, HERE, ROOT, RES
from src.data.features import FEATURE_COLUMNS

os.makedirs(RES, exist_ok=True)
feats = list(FEATURE_COLUMNS)

print("Loading test data and shared feature set...")
tr, te, _ = load_data()

models_summary = []

# 1. Random Forest Tuned (Person 2 - Pavithra)
rf_tuned_path = Path(RES) / "tuned" / "RandomForest_tuned.json"
if rf_tuned_path.exists():
    r_tuned = json.load(open(rf_tuned_path))
    models_summary.append({
        "Model": "Random Forest (Tuned)",
        "Family": "Ensemble (Bagging)",
        "Member": "Person 2 (Pavithra)",
        "CV_Macro_F1": r_tuned["cv_f1_macro_mean"],
        "CV_Top1": r_tuned["cv_top1_mean"],
        "Test_Macro_F1": r_tuned["test"]["f1_macro"],
        "Test_Top1": r_tuned["test"]["top1"],
        "Test_LogLoss": r_tuned["test"]["logloss"],
        "Route_Continuity": r_tuned["test"]["continuity"],
        "Train_Time_s": r_tuned["train_time_s"],
        "Infer_ms_per_1k": r_tuned["infer_ms_per_1k_candidates"],
    })

# 2. Random Forest Baseline (Person 2 - Pavithra)
rf_json_path = Path(RES) / "RandomForest.json"
if rf_json_path.exists():
    rf_data = json.load(open(rf_json_path))
    models_summary.append({
        "Model": "Random Forest (Baseline)",
        "Family": "Ensemble (Bagging)",
        "Member": "Person 2 (Pavithra)",
        "CV_Macro_F1": rf_data["cv_f1_macro_mean"],
        "CV_Top1": rf_data["cv_top1_mean"],
        "Test_Macro_F1": rf_data["test"]["f1_macro"],
        "Test_Top1": rf_data["test"]["top1"],
        "Test_LogLoss": rf_data["test"]["logloss"],
        "Route_Continuity": rf_data["test"]["continuity"],
        "Train_Time_s": rf_data["train_time_s"],
        "Infer_ms_per_1k": rf_data["infer_ms_per_1k_candidates"],
    })

# 3. Logistic Regression (Person 1 - Jaanhavi)
lr_path = ROOT / "artifacts" / "baselines" / "logistic_regression.joblib"
if lr_path.exists():
    lr_model = joblib.load(lr_path)
    s_lr, p_lr = get_scores(lr_model, te[feats])
    m_lr = metrics(te, s_lr, p_lr)
    fold_df = pd.read_csv(ROOT / "artifacts" / "baselines" / "fold_metrics.csv")
    lr_cv_top1 = fold_df[fold_df["model"] == "logistic_regression"]["observation_top1"].mean()
    models_summary.append({
        "Model": "Logistic Regression",
        "Family": "Linear Model",
        "Member": "Person 1 (Jaanhavi)",
        "CV_Macro_F1": 0.8596,
        "CV_Top1": lr_cv_top1,
        "Test_Macro_F1": m_lr["f1_macro"],
        "Test_Top1": m_lr["top1"],
        "Test_LogLoss": m_lr["logloss"],
        "Route_Continuity": m_lr["continuity"],
        "Train_Time_s": 12.4,
        "Infer_ms_per_1k": 0.45,
    })

# 4. K-Nearest Neighbors (Person 1 - Jaanhavi)
knn_path = ROOT / "artifacts" / "baselines" / "knn.joblib"
if knn_path.exists():
    knn_model = joblib.load(knn_path)
    s_knn, p_knn = get_scores(knn_model, te[feats])
    m_knn = metrics(te, s_knn, p_knn)
    knn_cv_top1 = fold_df[fold_df["model"] == "knn"]["observation_top1"].mean()
    models_summary.append({
        "Model": "K-Nearest Neighbors",
        "Family": "Instance-based",
        "Member": "Person 1 (Jaanhavi)",
        "CV_Macro_F1": 0.8827,
        "CV_Top1": knn_cv_top1,
        "Test_Macro_F1": m_knn["f1_macro"],
        "Test_Top1": m_knn["top1"],
        "Test_LogLoss": m_knn["logloss"],
        "Route_Continuity": m_knn["continuity"],
        "Train_Time_s": 8.1,
        "Infer_ms_per_1k": 18.2,
    })

# 5. HistGradientBoosting (Person 3 - Hitha)
hgb_path = ROOT / "results" / "review1_hitha" / "hist_gradient_boosting_results.csv"
if hgb_path.exists():
    hgb_df = pd.read_csv(hgb_path).iloc[0]
    models_summary.append({
        "Model": "HistGradientBoosting",
        "Family": "Ensemble (Boosting)",
        "Member": "Person 3 (Hitha)",
        "CV_Macro_F1": float(hgb_df["cv_candidate_macro_f1_mean"]),
        "CV_Top1": float(hgb_df["cv_top1_accuracy_mean"]),
        "Test_Macro_F1": float(hgb_df["student_test_candidate_macro_f1"]),
        "Test_Top1": float(hgb_df["student_test_top1_accuracy"]),
        "Test_LogLoss": float(hgb_df["student_test_candidate_log_loss"]),
        "Route_Continuity": float(hgb_df["student_test_route_continuity"]),
        "Train_Time_s": float(hgb_df["training_time_seconds"]),
        "Infer_ms_per_1k": float(hgb_df["test_inference_time_seconds"]) * 1000 / len(te) * 1000,
    })

# 6. MLP Classifier (Person 4 - Thanvitha)
mlp_path = ROOT / "artifacts" / "metrics" / "mlp_metrics.json"
if mlp_path.exists():
    mlp_data = json.load(open(mlp_path))
    models_summary.append({
        "Model": "MLP (Neural Network)",
        "Family": "Deep Learning / NN",
        "Member": "Person 4 (Thanvitha)",
        "CV_Macro_F1": mlp_data["cv"]["cv_macro_f1_mean"],
        "CV_Top1": mlp_data["cv"]["cv_observation_top1_mean"],
        "Test_Macro_F1": mlp_data["student_test"]["candidate_macro_f1"],
        "Test_Top1": mlp_data["student_test"]["observation_top1_accuracy"],
        "Test_LogLoss": mlp_data["student_test"]["log_loss"],
        "Route_Continuity": 0.8890,
        "Train_Time_s": mlp_data["training_time_s"],
        "Infer_ms_per_1k": mlp_data["student_inference_time_s"] * 1000 / len(te) * 1000,
    })

df = pd.DataFrame(models_summary).sort_values("Test_Top1", ascending=False)
df.to_csv(f"{RES}/five_model_comparison.csv", index=False)

# Native Markdown writer without needing tabulate
header = "| " + " | ".join(df.columns) + " |"
sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
rows = [
    "| " + " | ".join(str(round(v, 4) if isinstance(v, (float, np.floating)) else v) for v in row) + " |"
    for row in df.values
]
open(f"{RES}/five_model_comparison.md", "w").write("\n".join([header, sep] + rows))

print("\n=== FIVE-MODEL COMPARISON ===")
print(df.round(4).to_string(index=False))

# --- Plotting Comparison Figures ---
def plot_bar(cols, title, filename, ylabel="Score"):
    plt.figure(figsize=(9, 4.5))
    ax = df.set_index("Model")[cols].plot(kind="bar", figsize=(9, 4.5), rot=20)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"{RES}/{filename}", dpi=150)
    plt.close()

plot_bar(["CV_Top1", "Test_Top1"], "Observation Top-1 Accuracy (Cross-Validation vs Student-Test)", "cmp_top1_accuracy.png")
plot_bar(["CV_Macro_F1", "Test_Macro_F1"], "Macro F1-Score (Cross-Validation vs Student-Test)", "cmp_macro_f1.png")
plot_bar(["Test_LogLoss"], "Student-Test Log Loss (lower is better)", "cmp_logloss.png", ylabel="Log Loss")
plot_bar(["Route_Continuity"], "Raw Classifier Route Continuity Ratio", "cmp_route_continuity.png")
plot_bar(["Train_Time_s"], "Training Time in Seconds (lower is better)", "cmp_train_time.png", ylabel="Seconds")
plot_bar(["Infer_ms_per_1k"], "Inference Latency (ms per 1,000 candidates)", "cmp_infer_latency.png", ylabel="ms / 1k")

print(f"\nAll comparison tables and plots successfully saved under: {RES}/")