"""
Audit script for PathSnap accuracy and metric discrepancies.
Analyzes the verified student-testing predictions and computes:
  1. Micro-averaged Top-1 accuracy across all student-test observations
  2. Per-trip Top-1 accuracy for each of the 10 student-test trips
  3. Macro-averaged Top-1 accuracy (unweighted mean of per-trip accuracies)
  4. Candidate miss rate (per-trip and micro/macro)
  5. Accuracy conditional on reference arc being present in candidate set
  6. Route continuity (Classifier-Only vs HMM)
  7. Exact mathematical explanation of discrepancies between 72.77%, 75.68%, and 76.32%
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

def main():
    csv_path = REPO_ROOT / "results" / "review2_hitha" / "hmm_comparison.csv"
    if not csv_path.exists():
        csv_path = REPO_ROOT.parent / "hack" / "results" / "evaluation" / "hmm_comparison.csv"
    
    df = pd.read_csv(csv_path)
    
    # Calculate counts
    df["correct_clf"] = (df["observations"] * df["clf_top1_accuracy"]).round().astype(int)
    df["correct_hmm"] = (df["observations"] * df["hmm_top1_accuracy"]).round().astype(int)
    df["n_missed"] = (df["observations"] * df["candidate_miss_rate"]).round().astype(int)
    df["n_present"] = df["observations"] - df["n_missed"]
    
    df["clf_top1_conditional"] = (df["correct_clf"] / df["n_present"]).round(4)
    df["hmm_top1_conditional"] = (df["correct_hmm"] / df["n_present"]).round(4)

    total_obs = int(df["observations"].sum())
    total_correct_clf = int(df["correct_clf"].sum())
    total_correct_hmm = int(df["correct_hmm"].sum())
    total_missed = int(df["n_missed"].sum())
    total_present = int(df["n_present"].sum())

    # Micro metrics
    micro_clf_acc = total_correct_clf / total_obs
    micro_hmm_acc = total_correct_hmm / total_obs
    micro_miss_rate = total_missed / total_obs
    micro_clf_cond = total_correct_clf / total_present
    micro_hmm_cond = total_correct_hmm / total_present

    # Macro metrics
    macro_clf_acc = df["clf_top1_accuracy"].mean()
    macro_hmm_acc = df["hmm_top1_accuracy"].mean()
    macro_miss_rate = df["candidate_miss_rate"].mean()
    macro_clf_cond = df["clf_top1_conditional"].mean()
    macro_hmm_cond = df["hmm_top1_conditional"].mean()
    macro_clf_cont = df["clf_continuity"].mean()
    macro_hmm_cont = df["hmm_continuity"].mean()

    print("=" * 85)
    print("PATHSNAP STEP 1 — ACCURACY AND METRIC AUDIT REPORT")
    print("=" * 85)
    print(f"Total Student-Test Trips:        {len(df)}")
    print(f"Total Student-Test Observations: {total_obs:,}")
    print("\n--- PER-TRIP BREAKDOWN ---")
    display_cols = [
        "trip_id", "observations", "candidate_miss_rate", 
        "clf_top1_accuracy", "clf_top1_conditional", 
        "hmm_top1_accuracy", "hmm_top1_conditional", 
        "clf_continuity", "hmm_continuity"
    ]
    print(df[display_cols].to_string(index=False))

    print("\n" + "=" * 85)
    print("--- COMPARISON: MICRO-AVERAGED vs MACRO-AVERAGED METRICS ---")
    print(f"{'Metric':<38} | {'Micro-Averaged':<18} | {'Macro-Averaged':<18}")
    print("-" * 80)
    print(f"{'Classifier-Only Top-1 Accuracy':<38} | {micro_clf_acc*100:>16.2f}% | {macro_clf_acc*100:>16.2f}%")
    print(f"{'Classifier + HMM Top-1 Accuracy':<38} | {micro_hmm_acc*100:>16.2f}% | {macro_hmm_acc*100:>16.2f}%")
    print(f"{'Candidate Miss Rate':<38} | {micro_miss_rate*100:>16.2f}% | {macro_miss_rate*100:>16.2f}%")
    print(f"{'Classifier Acc (Cond. on Hit)':<38} | {micro_clf_cond*100:>16.2f}% | {macro_clf_cond*100:>16.2f}%")
    print(f"{'HMM Acc (Cond. on Hit)':<38} | {micro_hmm_cond*100:>16.2f}% | {macro_hmm_cond*100:>16.2f}%")
    print(f"{'Classifier Route Continuity':<38} | {'--':>18} | {macro_clf_cont*100:>16.2f}%")
    print(f"{'Classifier + HMM Route Continuity':<38} | {'--':>18} | {macro_hmm_cont*100:>16.2f}%")
    print("=" * 85)

    print("\n--- EXPLANATION OF REPORTED NUMBERS ---")
    print(f"1. 72.77%: Micro-averaged Top-1 accuracy across all {total_obs:,} observations.")
    print("   Trip 00000077 is exceptionally long (3,911 obs, 20.4% of total) and has only")
    print(f"   50.98% accuracy, pulling down the unweighted observation pool to {micro_clf_acc*100:.2f}%.")
    print(f"2. 75.68%: Macro-averaged Top-1 accuracy for Classifier-Only.")
    print("   Unweighted arithmetic mean of the 10 individual trip accuracies.")
    print(f"3. 76.32%: Macro-averaged Top-1 accuracy for Classifier + HMM.")
    print("   HMM sequential decoding improves mean trip accuracy from 75.68% to 76.32%,")
    print("   while boosting route continuity from 92.69% to 97.84% (a +5.15% gain).")

    audit_out = REPO_ROOT / "results" / "audit_metrics.json"
    audit_data = {
        "total_observations": total_obs,
        "micro": {
            "clf_top1_accuracy": round(float(micro_clf_acc), 4),
            "hmm_top1_accuracy": round(float(micro_hmm_acc), 4),
            "candidate_miss_rate": round(float(micro_miss_rate), 4),
            "clf_conditional_accuracy": round(float(micro_clf_cond), 4),
            "hmm_conditional_accuracy": round(float(micro_hmm_cond), 4),
        },
        "macro": {
            "clf_top1_accuracy": round(float(macro_clf_acc), 4),
            "hmm_top1_accuracy": round(float(macro_hmm_acc), 4),
            "candidate_miss_rate": round(float(macro_miss_rate), 4),
            "clf_conditional_accuracy": round(float(macro_clf_cond), 4),
            "hmm_conditional_accuracy": round(float(macro_hmm_cond), 4),
            "clf_continuity": round(float(macro_clf_cont), 4),
            "hmm_continuity": round(float(macro_hmm_cont), 4),
        },
        "per_trip": df.to_dict(orient="records"),
    }
    with open(audit_out, "w") as fp:
        json.dump(audit_data, fp, indent=2)
    print(f"\nAudit results successfully saved to {audit_out}")

if __name__ == "__main__":
    main()
