from __future__ import annotations

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

SOURCE_TABLE = RESULTS_DIR / "stage_xgboost_vs_qwen_sft_cluster_bootstrap_auc_final_common.csv"

STAGE_ORDER = ("Stage1", "Stage2", "Stage3", "Stage4", "Stage5")
STAGE_LABELS = {
    "Stage1": "Stage1\n0-1 mo",
    "Stage2": "Stage2\n2-3 mo",
    "Stage3": "Stage3\n4-12 mo",
    "Stage4": "Stage4\n1-2 y",
    "Stage5": "Stage5\n>2 y",
}
TARGETS = ("1m", "1y", "5y")
TARGET_LABELS = {
    "1m": "1-month",
    "1y": "1-year",
    "5y": "5-year",
}

MODEL_A = "XGBoost"
MODEL_B = "Qwen3-4B SFT"
VMIN = -0.45
VMAX = 0.45
COLORBAR_TICKS = np.arange(VMIN, VMAX + 0.001, 0.15)


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
        "Stage",
        "Stage_Display",
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
    df = df[(df["Model_A"] == MODEL_A) & (df["Model_B"] == MODEL_B)].copy()
    if df.empty:
        raise ValueError(f"No rows found for {MODEL_A} vs {MODEL_B} in {SOURCE_TABLE}")
    return df


def build_stage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row_index, stage in enumerate(STAGE_ORDER):
        for column_index, target in enumerate(TARGETS):
            match = df[(df["Stage"] == stage) & (df["Target"] == target)]
            if match.empty:
                raise ValueError(f"Missing stage result for {stage}, {target}")
            item = match.iloc[0]
            q_value = float(item["Q_Value_BH"])
            rows.append(
                {
                    "Stage": stage,
                    "Stage_Display": item["Stage_Display"],
                    "Row_Index": row_index,
                    "Target": target,
                    "Column_Index": column_index,
                    "Target_Display": TARGET_LABELS[target],
                    "Model_A": MODEL_A,
                    "Model_B": MODEL_B,
                    "XGBoost_AUROC": float(item["Model_A_Value"]),
                    "Qwen3_4B_SFT_AUROC": float(item["Model_B_Value"]),
                    "Delta_AUROC_XGBoost_Minus_Qwen_SFT": float(item["Delta_A_minus_B"]),
                    "P_Value": float(item["P_Value"]),
                    "Q_Value_BH": q_value,
                    "Significance": significance_stars(q_value),
                }
            )
    return pd.DataFrame.from_records(rows)


def draw_significance_legend(fig: plt.Figure) -> None:
    ax = fig.add_axes([0.68, 0.43, 0.19, 0.22])
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


def draw_stage_heatmap(matrix: pd.DataFrame) -> None:
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=VMIN, vcenter=0.0, vmax=VMAX)

    fig = plt.figure(figsize=(10.2, 6.2))
    fig.text(
        0.08,
        0.94,
        "Stage-specific AUROC Difference: XGBoost vs Qwen3-4B SFT",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )
    ax = fig.add_axes([0.12, 0.17, 0.43, 0.68])
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, len(TARGETS) - 0.5)
    ax.set_ylim(len(STAGE_ORDER) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(TARGETS)))
    ax.set_xticklabels([TARGET_LABELS[target] for target in TARGETS], fontsize=10)
    ax.set_yticks(np.arange(len(STAGE_ORDER)))
    ax.set_yticklabels([STAGE_LABELS[stage] for stage in STAGE_ORDER], fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for _, item in matrix.iterrows():
        x = int(item["Column_Index"])
        y = int(item["Row_Index"])
        value = float(item["Delta_AUROC_XGBoost_Minus_Qwen_SFT"])
        stars = item["Significance"]
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor=cmap(norm(value)),
                edgecolor="#cfcfcf",
                linewidth=0.8,
            )
        )
        label = f"{value:+.3f}" if not stars else f"{value:+.3f}\n{stars}"
        text_color = "white" if abs(value) >= 0.31 else "black"
        ax.text(x, y, label, ha="center", va="center", fontsize=10, color=text_color, linespacing=1.1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.90, 0.20, 0.027, 0.62])
    cbar = fig.colorbar(sm, cax=cax, ticks=COLORBAR_TICKS)
    cbar.ax.set_yticklabels([f"{tick:.2f}".rstrip("0").rstrip(".") for tick in COLORBAR_TICKS])
    cbar.set_label("AUROC difference (XGBoost - Qwen3-4B SFT)", labelpad=7)

    draw_significance_legend(fig)

    for suffix in ("png", "pdf", "svg"):
        output = FIGURES_DIR / f"stage_xgboost_vs_qwen_sft_auc_delta_heatmap.{suffix}"
        kwargs = {}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(output, **kwargs)
    plt.close(fig)


def write_execution_note() -> None:
    note = f"""# Stage-specific Heatmap Execution

Input table:

```text
{SOURCE_TABLE}
```

AUROC source:

```text
{PROJECT_ROOT / "evaluation" / "metrics" / "final_stage_auroc_common_cohort.csv"}
```

The displayed value is:

```text
AUROC(XGBoost) - AUROC(Qwen3-4B SFT)
```

The stage-specific heatmap uses the same fixed color scale as the three overall heatmaps:

```text
{VMIN} to {VMAX}
```
"""
    (DOCS_DIR / "stage_heatmap_execution_note.md").write_text(note, encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    df = load_results()
    matrix = build_stage_matrix(df)
    matrix.to_csv(RESULTS_DIR / "stage_xgboost_qwen_sft_heatmap_matrix.csv", index=False)
    draw_stage_heatmap(matrix)
    write_execution_note()
    print(f"Wrote stage-specific heatmap to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
