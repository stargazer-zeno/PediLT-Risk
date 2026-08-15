from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from significance_common import FIGURES_DIR, RESULTS_DIR, TARGETS, ensure_output_dirs


def plot_forest(df: pd.DataFrame) -> None:
    if df.empty:
        return
    data = df[df["Metric"] == "auroc"].copy()
    if data.empty:
        return
    data["Label"] = data["Target"] + "  " + data["Model_A"] + " - " + data["Model_B"]
    data = data.sort_values(["Target", "Delta_A_minus_B"])
    y = np.arange(len(data))
    fig_height = max(5, 0.36 * len(data) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    x = data["Delta_A_minus_B"].to_numpy()
    low = data["CI_95_Low"].to_numpy()
    high = data["CI_95_High"].to_numpy()
    ax.errorbar(x, y, xerr=[x - low, high - x], fmt="o", color="#1f77b4", ecolor="#4f83b7", capsize=3)
    ax.axvline(0, color="#666666", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(data["Label"])
    ax.set_xlabel("Delta AUROC (Model A - Model B)")
    ax.set_title("Primary paired AUROC differences with patient-cluster bootstrap 95% CI")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "auc_difference_forest.png", dpi=220)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame) -> None:
    if df.empty:
        return
    q_col = "Q_Value_BH_All_AUROC" if "Q_Value_BH_All_AUROC" in df.columns else "P_Value"
    data = df[df["Metric"] == "auroc"].copy() if "Metric" in df.columns else df.copy()
    if data.empty:
        return
    data["Comparison"] = data["Model_A"] + " vs " + data["Model_B"]
    comparisons = list(dict.fromkeys(data["Comparison"].tolist()))
    matrix = np.full((len(comparisons), len(TARGETS)), np.nan)
    for i, comparison in enumerate(comparisons):
        for j, target in enumerate(TARGETS):
            row = data[(data["Comparison"] == comparison) & (data["Target"] == target)]
            if not row.empty:
                matrix[i, j] = float(row.iloc[0][q_col])
    transformed = -np.log10(np.clip(matrix, 1e-300, 1.0))
    fig_height = max(5, 0.32 * len(comparisons) + 1.6)
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    im = ax.imshow(transformed, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(TARGETS)))
    ax.set_xticklabels(TARGETS)
    ax.set_yticks(np.arange(len(comparisons)))
    ax.set_yticklabels(comparisons)
    ax.set_title("Pairwise AUROC significance (-log10 adjusted P/q)")
    for i in range(len(comparisons)):
        for j in range(len(TARGETS)):
            value = matrix[i, j]
            if np.isfinite(value):
                label = "<0.001" if value < 0.001 else f"{value:.3f}"
                ax.text(j, i, label, ha="center", va="center", color="white" if transformed[i, j] > 1.2 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("-log10(q)" if q_col != "P_Value" else "-log10(P)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pairwise_auc_qvalue_heatmap.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    primary_path = RESULTS_DIR / "primary_cluster_bootstrap_auc.csv"
    all_path = RESULTS_DIR / "all_pairwise_auc_fdr.csv"
    primary = pd.read_csv(primary_path) if primary_path.exists() else pd.DataFrame()
    all_auc = pd.read_csv(all_path) if all_path.exists() else primary
    plot_forest(primary)
    plot_heatmap(all_auc)
    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
