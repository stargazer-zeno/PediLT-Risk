from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = PROJECT_ROOT / "evaluation" / "statistical_significance"
RESULTS_DIR = ANALYSIS_ROOT / "results"
FIGURES_DIR = ANALYSIS_ROOT / "figures"
DOCS_DIR = ANALYSIS_ROOT / "docs"

SOURCE_TABLE = RESULTS_DIR / "overall_pairwise_cluster_bootstrap_auc_final_common.csv"

TARGETS = ("1m", "1y", "5y")
TARGET_LABELS = {
    "1m": "1-month",
    "1y": "1-year",
    "5y": "5-year",
}
MODEL_ORDER = (
    "XGBoost",
    "LSTM",
    "RSF",
    "Qwen3-4B SFT",
    "Qwen3-4B baseline",
    "Llama3.1-8B",
    "Huatuo-O1-7B",
)
MODEL_DISPLAY = {
    "XGBoost": "XGBoost",
    "LSTM": "LSTM",
    "RSF": "RSF",
    "Qwen3-4B SFT": "Qwen3-4B\nSFT",
    "Qwen3-4B baseline": "Qwen3-4B\nbase",
    "Llama3.1-8B": "Llama3.1\n8B",
    "Huatuo-O1-7B": "Huatuo-O1\n7B",
}

VMIN = -0.45
VMAX = 0.45
COLORBAR_TICKS = np.arange(VMIN, VMAX + 0.001, 0.15)


@dataclass(frozen=True)
class PairResult:
    column_auc: float
    row_auc: float
    delta_column_minus_row: float
    q_value: float
    p_value: float


def ensure_output_dirs() -> None:
    for path in (RESULTS_DIR, FIGURES_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def significance_stars(q_value: float) -> str:
    if not np.isfinite(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def load_results() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_TABLE)
    required = {
        "Model_A",
        "Model_B",
        "Target",
        "Model_A_Value",
        "Model_B_Value",
        "Delta_A_minus_B",
        "P_Value",
        "Q_Value_BH",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {SOURCE_TABLE}: {sorted(missing)}")
    return df


def get_pair_result(df: pd.DataFrame, target: str, column_model: str, row_model: str) -> PairResult:
    direct = df[
        (df["Target"] == target)
        & (df["Model_A"] == column_model)
        & (df["Model_B"] == row_model)
    ]
    if not direct.empty:
        row = direct.iloc[0]
        return PairResult(
            column_auc=float(row["Model_A_Value"]),
            row_auc=float(row["Model_B_Value"]),
            delta_column_minus_row=float(row["Delta_A_minus_B"]),
            q_value=float(row["Q_Value_BH"]),
            p_value=float(row["P_Value"]),
        )

    reverse = df[
        (df["Target"] == target)
        & (df["Model_A"] == row_model)
        & (df["Model_B"] == column_model)
    ]
    if not reverse.empty:
        row = reverse.iloc[0]
        return PairResult(
            column_auc=float(row["Model_B_Value"]),
            row_auc=float(row["Model_A_Value"]),
            delta_column_minus_row=-float(row["Delta_A_minus_B"]),
            q_value=float(row["Q_Value_BH"]),
            p_value=float(row["P_Value"]),
        )

    raise ValueError(f"Missing pairwise result for {target}: {column_model} vs {row_model}")


def build_step_matrix(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for row_index, row_model in enumerate(MODEL_ORDER[1:]):
        full_row_index = row_index + 1
        for column_index, column_model in enumerate(MODEL_ORDER[:-1]):
            visible = column_index < full_row_index
            if not visible:
                continue
            result = get_pair_result(df, target, column_model, row_model)
            rows.append(
                {
                    "Target": target,
                    "Row_Index": row_index,
                    "Column_Index": column_index,
                    "Column_Model": column_model,
                    "Row_Model": row_model,
                    "Column_AUROC": result.column_auc,
                    "Row_AUROC": result.row_auc,
                    "Delta_AUROC_Column_Minus_Row": result.delta_column_minus_row,
                    "P_Value": result.p_value,
                    "Q_Value_BH": result.q_value,
                    "Significance": significance_stars(result.q_value),
                }
            )
    return pd.DataFrame.from_records(rows)


def draw_significance_legend(fig: plt.Figure) -> None:
    ax = fig.add_axes([0.675, 0.50, 0.19, 0.18])
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#222222", linewidth=0.9))
    ax.plot([0, 1], [0.63, 0.63], color="#222222", linewidth=0.9)
    ax.text(0.5, 0.82, "FDR-adjusted\nq value", ha="center", va="center", fontsize=9, fontweight="bold")
    legend_rows = [("*", "< 0.05"), ("**", "< 0.01"), ("***", "< 0.001")]
    y_positions = [0.46, 0.29, 0.12]
    for (label, threshold), y in zip(legend_rows, y_positions, strict=True):
        ax.text(0.17, y, label, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(0.62, y, threshold, ha="center", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def draw_step_heatmap(matrix: pd.DataFrame, target: str) -> None:
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=VMIN, vcenter=0.0, vmax=VMAX)
    n = len(MODEL_ORDER) - 1

    fig = plt.figure(figsize=(11.2, 8.0))
    fig.text(
        0.07,
        0.94,
        f"Overall AUROC Difference Matrix ({TARGET_LABELS[target]})",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    ax = fig.add_axes([0.07, 0.10, 0.60, 0.78])
    ax.set_aspect("equal")
    ax.set_xlim(-0.75, n - 0.45)
    ax.set_ylim(n - 0.45, -0.95)
    ax.set_axis_off()

    for _, item in matrix.iterrows():
        x = int(item["Column_Index"])
        y = int(item["Row_Index"])
        value = float(item["Delta_AUROC_Column_Minus_Row"])
        stars = item["Significance"]
        facecolor = cmap(norm(value))
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor=facecolor,
                edgecolor="#cfcfcf",
                linewidth=0.8,
            )
        )
        label = f"{value:+.3f}" if not stars else f"{value:+.3f}\n{stars}"
        text_color = "white" if abs(value) >= 0.31 else "black"
        ax.text(x, y, label, ha="center", va="center", fontsize=9, color=text_color, linespacing=1.1)

    for y, model in enumerate(MODEL_ORDER[1:]):
        ax.text(-0.66, y, MODEL_DISPLAY[model], ha="right", va="center", fontsize=10)

    for x, model in enumerate(MODEL_ORDER[:-1]):
        ax.text(x, x - 0.67, MODEL_DISPLAY[model], ha="center", va="bottom", fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.90, 0.20, 0.025, 0.62])
    cbar = fig.colorbar(sm, cax=cax, ticks=COLORBAR_TICKS)
    cbar.ax.set_yticklabels([f"{tick:.2f}".rstrip("0").rstrip(".") for tick in COLORBAR_TICKS])
    cbar.set_label("AUROC difference (column - row)", labelpad=7)

    draw_significance_legend(fig)

    for suffix in ("png", "pdf", "svg"):
        output = FIGURES_DIR / f"overall_auc_delta_step_heatmap_{target}.{suffix}"
        kwargs = {}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(output, **kwargs)
    plt.close(fig)


def write_color_scale_summary(all_matrices: list[pd.DataFrame]) -> None:
    combined = pd.concat(all_matrices, ignore_index=True)
    summary = pd.DataFrame(
        [
            {
                "Color_Map": "RdBu_r",
                "Color_Center": 0.0,
                "Fixed_Vmin": VMIN,
                "Fixed_Vmax": VMAX,
                "Observed_Min_Delta": combined["Delta_AUROC_Column_Minus_Row"].min(),
                "Observed_Max_Delta": combined["Delta_AUROC_Column_Minus_Row"].max(),
                "Difference_Definition": "AUROC(column model) - AUROC(row model)",
            }
        ]
    )
    summary.to_csv(RESULTS_DIR / "color_scale_summary.csv", index=False)


def write_execution_note() -> None:
    note = f"""# Overall Step Heatmap Execution

Input table:

```text
{SOURCE_TABLE}
```

AUROC source:

```text
{PROJECT_ROOT / "evaluation" / "metrics" / "final_overall_auroc_common_cohort.csv"}
```

Generated figures use a staircase lower-triangle layout. The value in each cell is:

```text
AUROC(column model) - AUROC(row model)
```

The three endpoint figures share the same fixed color scale:

```text
{VMIN} to {VMAX}
```

This keeps color interpretation identical across the 1-month, 1-year, and 5-year figures.
"""
    (DOCS_DIR / "execution_note.md").write_text(note, encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    df = load_results()
    matrices: list[pd.DataFrame] = []
    for target in TARGETS:
        matrix = build_step_matrix(df, target)
        matrix.to_csv(RESULTS_DIR / f"overall_step_heatmap_matrix_{target}.csv", index=False)
        draw_step_heatmap(matrix, target)
        matrices.append(matrix)
    write_color_scale_summary(matrices)
    write_execution_note()
    print(f"Wrote overall step heatmaps to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
