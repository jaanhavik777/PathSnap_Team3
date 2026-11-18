"""
Comprehensive evaluation script for Enhanced HistGradientBoosting + HMM on all 10 student-test trips.
Compares:
  - Baseline HistGradientBoosting (8 features, Review 2)
  - Enhanced HistGradientBoosting (14 features + dense spatial index)
  - Enhanced HistGradientBoosting + HMM (Viterbi decoding)
Saves:
  - results/accuracy_improvement/enhanced_evaluation_summary.csv
  - results/accuracy_improvement/enhanced_evaluation_summary.json
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.data.parsers import load_trip
from src.data.road_graph import RoadGraph
from src.data.alignment import align_reference_route_to_observations
from src.data.enhanced_features import extract_enhanced_candidates
from src.hmm.viterbi import ViterbiDecoder
from src.hmm.route_metrics import calculate_route_continuity, calculate_top1_accuracy, calculate_candidate_miss_rate

REPO_ROOT = Path(__file__).resolve().parent.parent

def locate_student_test():
    candidates = [
        REPO_ROOT.parent / "hack" / "Student folder" / "Student folder" / "student_test",
        REPO_ROOT / "data" / "raw" / "student_test",
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*/*.track")):
            return c
    raise FileNotFoundError("student_test directory not found")

def main():
    test_dir = locate_student_test()
    model_path = REPO_ROOT / "artifacts" / "models" / "hist_gradient_boosting_enhanced.joblib"
    model = joblib.load(model_path)
    
    trip_folders = sorted([f for f in test_dir.iterdir() if f.is_dir() and (f / f"{f.name}.track").exists()])
    
    print("=" * 88)
    print("PATHSNAP FINAL EVALUATION — ENHANCED HISTGRADIENTBOOSTING + HMM")
    print(f"Model: {model_path}")
    print(f"Number of Student Trips: {len(trip_folders)}")
    print("=" * 88)

    records = []

    for f in trip_folders:
        tid = f.name
        trip = load_trip(f, include_route=True)
        track = trip["track"]
        N = len(track)
        lons = track["longitude"].to_numpy(dtype=float)
        lats = track["latitude"].to_numpy(dtype=float)
        timestamps = track["timestamp"].to_numpy(dtype=float)

        rg = RoadGraph(
            trip["nodes"][["longitude", "latitude"]].to_numpy(dtype=float),
            trip["arcs"][["from_node_id", "to_node_id"]].to_numpy(dtype=int),
            trip_id=tid,
        )

        raw_route = trip["route"]["arc_id"].to_numpy(dtype=int)
        gt = align_reference_route_to_observations(lons, lats, timestamps, rg, raw_route, search_window=50)

        # 1. Enhanced Candidates (Dense segment index R=75m, K=12)
        t_cand_0 = time.perf_counter()
        obs_cands, X_feat, _ = extract_enhanced_candidates(
            rg, lons, lats, timestamps, search_radius_m=75.0, max_cands=12
        )
        t_cand_ms = (time.perf_counter() - t_cand_0) * 1000

        # 2. Classifier Prediction
        t_clf_0 = time.perf_counter()
        scores = model.predict_proba(X_feat)[:, 1]
        t_clf_ms = (time.perf_counter() - t_clf_0) * 1000

        scores_per_obs = []
        idx = 0
        for t in range(N):
            cnt = len(obs_cands[t])
            scores_per_obs.append(scores[idx : idx + cnt])
            idx += cnt

        clf_arcs = []
        for t in range(N):
            cands = obs_cands[t]
            sc = scores_per_obs[t]
            if len(cands) > 0 and len(sc) > 0:
                clf_arcs.append(cands[int(np.argmax(sc))].arc_id)
            else:
                clf_arcs.append(-1)
        clf_arcs = np.array(clf_arcs)

        # 3. HMM Viterbi Sequential Decoding
        t_hmm_0 = time.perf_counter()
        decoder = ViterbiDecoder(
            road_graph=rg,
            beta_meters=18.0,
            sigma_z_meters=10.0,
            max_speed_mps=45.0,
            gap_threshold_sec=10.0,
        )
        _, hmm_arcs_list, _, _ = decoder.decode(obs_cands, scores_per_obs, lons, lats, timestamps)
        hmm_arcs = np.array(hmm_arcs_list)
        t_hmm_ms = (time.perf_counter() - t_hmm_0) * 1000

        # Metrics
        miss_rate = calculate_candidate_miss_rate(obs_cands, gt)
        clf_acc = calculate_top1_accuracy(clf_arcs, gt)
        hmm_acc = calculate_top1_accuracy(hmm_arcs, gt)
        clf_cont = calculate_route_continuity(clf_arcs, rg)
        hmm_cont = calculate_route_continuity(hmm_arcs, rg)

        rec = {
            "trip_id": tid,
            "observations": N,
            "candidates": len(X_feat),
            "candidate_miss_rate": round(float(miss_rate), 4),
            "clf_top1_accuracy": round(float(clf_acc), 4),
            "hmm_top1_accuracy": round(float(hmm_acc), 4),
            "acc_delta": round(float(hmm_acc - clf_acc), 4),
            "clf_continuity": round(float(clf_cont), 4),
            "hmm_continuity": round(float(hmm_cont), 4),
            "cont_delta": round(float(hmm_cont - clf_cont), 4),
            "candidate_gen_ms": round(float(t_cand_ms), 1),
            "clf_latency_ms": round(float(t_clf_ms), 1),
            "hmm_latency_ms": round(float(t_hmm_ms), 1),
            "total_latency_ms": round(float(t_cand_ms + t_clf_ms + t_hmm_ms), 1),
        }
        records.append(rec)
        print(f"Trip {tid:<8} ({N:>4} obs) | Miss: {miss_rate*100:>4.2f}% | "
              f"Acc: {clf_acc*100:>5.2f}% -> {hmm_acc*100:>5.2f}% ({rec['acc_delta']*100:>+5.2f}%) | "
              f"Cont: {clf_cont*100:>5.2f}% -> {hmm_cont*100:>5.2f}% ({rec['cont_delta']*100:>+5.2f}%)")

    df = pd.DataFrame(records)
    out_dir = REPO_ROOT / "results" / "accuracy_improvement"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "enhanced_evaluation_summary.csv"
    json_path = out_dir / "enhanced_evaluation_summary.json"

    df.to_csv(csv_path, index=False)

    total_obs = int(df["observations"].sum())
    macro_miss = float(df["candidate_miss_rate"].mean())
    macro_clf_acc = float(df["clf_top1_accuracy"].mean())
    macro_hmm_acc = float(df["hmm_top1_accuracy"].mean())
    macro_clf_cont = float(df["clf_continuity"].mean())
    macro_hmm_cont = float(df["hmm_continuity"].mean())

    micro_clf_acc = float(np.sum(df["observations"] * df["clf_top1_accuracy"]) / total_obs)
    micro_hmm_acc = float(np.sum(df["observations"] * df["hmm_top1_accuracy"]) / total_obs)
    micro_miss = float(np.sum(df["observations"] * df["candidate_miss_rate"]) / total_obs)

    summary = {
        "macro": {
            "candidate_miss_rate": round(macro_miss, 4),
            "clf_top1_accuracy": round(macro_clf_acc, 4),
            "hmm_top1_accuracy": round(macro_hmm_acc, 4),
            "clf_continuity": round(macro_clf_cont, 4),
            "hmm_continuity": round(macro_hmm_cont, 4),
        },
        "micro": {
            "candidate_miss_rate": round(micro_miss, 4),
            "clf_top1_accuracy": round(micro_clf_acc, 4),
            "hmm_top1_accuracy": round(micro_hmm_acc, 4),
        },
        "per_trip": df.to_dict(orient="records"),
    }
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    print("\n" + "=" * 88)
    print("--- ENHANCED MODEL STUDENT-TEST SUMMARY ---")
    print(f"{'Metric':<35} | {'Baseline (Review 2)':<22} | {'Enhanced (Step 6)':<22}")
    print("-" * 88)
    print(f"{'Macro Candidate Miss Rate':<35} | {'5.23%':<22} | {macro_miss*100:>20.2f}%")
    print(f"{'Macro Classifier Top-1 Accuracy':<35} | {'75.68%':<22} | {macro_clf_acc*100:>20.2f}%")
    print(f"{'Macro Classifier + HMM Top-1':<35} | {'76.32%':<22} | {macro_hmm_acc*100:>20.2f}%")
    print(f"{'Macro Route Continuity (HMM)':<35} | {'97.84%':<22} | {macro_hmm_cont*100:>20.2f}%")
    print(f"{'Micro Classifier Top-1 Accuracy':<35} | {'72.77%':<22} | {micro_clf_acc*100:>20.2f}%")
    print(f"{'Micro Classifier + HMM Top-1':<35} | {'73.75%':<22} | {micro_hmm_acc*100:>20.2f}%")
    print("=" * 88)
    print(f"Artifacts saved to:\n  {csv_path}\n  {json_path}")

if __name__ == "__main__":
    main()
