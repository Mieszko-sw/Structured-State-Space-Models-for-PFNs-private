"""Plot elastic looped 3x6 inference depths against the TabPFN baseline."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="result_csvs/tab_parity_reasoning_depth_summary.csv",
    )
    parser.add_argument(
        "--output",
        default="result_csvs/tab_parity_elastic_depths_vs_tabpfn.png",
    )
    parser.add_argument(
        "--uncertainty",
        choices=("std", "sem", "none"),
        default="std",
        help="Shaded uncertainty around each mean curve (default: std).",
    )
    return parser.parse_args()


def uncertainty_column(kind: str) -> str | None:
    if kind == "std":
        return "std_auc_roc"
    if kind == "sem":
        return "sem_auc_roc"
    return None


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.input_csv)
    required = {
        "model",
        "inference_depth",
        "k",
        "task_count",
        "mean_auc_roc",
        "std_auc_roc",
        "sem_auc_roc",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    elastic = summary.loc[summary["model"] == "elastic_looped_3x6"].copy()
    tabpfn = summary.loc[summary["model"] == "tabpfn"].copy()
    if elastic.empty or tabpfn.empty:
        raise ValueError("The input must contain elastic_looped_3x6 and tabpfn rows")

    elastic["inference_depth"] = elastic["inference_depth"].astype(int)
    elastic = elastic.sort_values(["inference_depth", "k"])
    tabpfn = tabpfn.sort_values("k")
    depths = sorted(elastic["inference_depth"].unique())
    ks = sorted(set(elastic["k"]) | set(tabpfn["k"]))
    uncertainty = uncertainty_column(args.uncertainty)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.9, len(depths)))

    for color, depth in zip(colors, depths):
        frame = elastic.loc[elastic["inference_depth"] == depth].sort_values("k")
        x = frame["k"].to_numpy()
        mean = frame["mean_auc_roc"].to_numpy()
        ax.plot(
            x,
            mean,
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=f"Elastic 3x6 — depth {depth} ({3 * depth} layer applications)",
        )
        if uncertainty is not None:
            spread = frame[uncertainty].fillna(0.0).to_numpy()
            ax.fill_between(
                x,
                np.clip(mean - spread, 0.0, 1.0),
                np.clip(mean + spread, 0.0, 1.0),
                color=color,
                alpha=0.08,
                linewidth=0,
            )

    x = tabpfn["k"].to_numpy()
    mean = tabpfn["mean_auc_roc"].to_numpy()
    ax.plot(
        x,
        mean,
        color="black",
        linestyle="--",
        marker="D",
        markersize=5,
        linewidth=2.4,
        label="TabPFN (12 layers)",
        zorder=10,
    )
    if uncertainty is not None:
        spread = tabpfn[uncertainty].fillna(0.0).to_numpy()
        ax.fill_between(
            x,
            np.clip(mean - spread, 0.0, 1.0),
            np.clip(mean + spread, 0.0, 1.0),
            color="black",
            alpha=0.08,
            linewidth=0,
            zorder=9,
        )

    band_label = {"std": "±1 std across tasks", "sem": "±1 SEM", "none": ""}
    title_suffix = f" ({band_label[args.uncertainty]})" if uncertainty else ""
    ax.axhline(0.5, color="0.5", linestyle=":", linewidth=1.3, label="Chance")
    ax.set_title(f"Tab-Parity: Elastic Depth vs. TabPFN{title_suffix}")
    ax.set_xlabel("Parity order k")
    ax.set_ylabel("Query AUC-ROC")
    ax.set_xticks(ks)
    ax.set_ylim(0.4, 1.02)
    ax.grid(axis="both", alpha=0.2)
    ax.legend(loc="best", fontsize=8.5, frameon=True)
    fig.tight_layout()

    parent = os.path.dirname(args.output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved focused elastic-depth plot to {args.output}")


if __name__ == "__main__":
    main()
