"""Plot measured baseline and EDA results for the report."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

MODEL_NAMES = {"logistic_regression": "Logistic Regression", "knn": "KNN"}
COLORS = ["#225C85", "#BC6337"]


def save(fig, folder, name):
    fig.savefig(folder / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_results(results, output):
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(results / "student_test_baseline_summary.csv").set_index("model").loc[list(MODEL_NAMES)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    for ax, column, title, proportion in zip(axes.flat, (
        "top1", "candidate_f1_at_0_5", "candidate_log_loss_rf_clip", "mean_route_arc_f1",
    ), ("Candidate top-1", "Positive-class F1 at 0.5", "Candidate log loss (lower is better)",
        "Mean decoded route arc F1"), (True, False, False, False)):
        bars = ax.bar(MODEL_NAMES.values(), summary[column], color=COLORS, width=0.55)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylim(0, 1.08 if column != "candidate_log_loss_rf_clip" else 1.25 * summary[column].max())
        labels = [f"{value:.2%}" if proportion else f"{value:.3f}" for value in summary[column]]
        ax.bar_label(bars, labels=labels, padding=3)
        if proportion:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Student-test baseline comparison", fontsize=15)
    save(fig, output, "student_test_comparison.png")

    folds = pd.read_csv(results / "baseline_cross_validation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for ax, metric, title in zip(axes, ("observation_top1", "average_precision"),
                                 ("Observation top-1", "Average precision")):
        for (key, name), color in zip(MODEL_NAMES.items(), COLORS):
            values = folds.loc[folds.model.eq(key)].sort_values("fold")
            ax.plot(values["fold"], values[metric], marker="o", color=color, label=name)
        ax.set(title=title, xlabel="Held-out trip fold", xticks=sorted(folds.fold.unique()), ylim=(0, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(loc="lower right")
    fig.suptitle("Training cross-validation by trip")
    save(fig, output, "training_cv.png")

    eda = pd.read_csv(results / "dataset_eda.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for partition, color in zip(("train", "student_test"), COLORS):
        rows = eda.loc[eda.partition.eq(partition)]
        x = rows.max_lon - rows.min_lon
        y = rows.max_lat - rows.min_lat
        axes[0].scatter(x, y, color=color, alpha=0.6, label=partition, s=16)
        gaps = rows.max_gap_s.sort_values().to_numpy()
        axes[1].plot(range(1, len(gaps) + 1), gaps.clip(1, None), color=color, label=partition)
    axes[0].set(xlabel="Longitude span (degrees)", ylabel="Latitude span (degrees)", title="Geographic extent by trip")
    axes[1].set(xlabel="Trip rank within partition", ylabel="Maximum gap (seconds)",
                title="Largest GPS gap per trip", yscale="log")
    for ax in axes:
        ax.legend()
    fig.suptitle("GPS trace diagnostics")
    save(fig, output, "gps_trace_diagnostics.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("artifacts/results"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures/baselines"))
    args = parser.parse_args()
    plot_results(args.results, args.out_dir)
    print(f"Saved comparison and EDA figures to {args.out_dir}")


if __name__ == "__main__":
    main()
