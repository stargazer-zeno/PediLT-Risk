from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from significance_common import DOCS_DIR, RESULTS_DIR, ensure_output_dirs, format_float, format_p


REFERENCES = [
    "DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves. Biometrics. 1988. https://pubmed.ncbi.nlm.nih.gov/3203132/",
    "TRIPOD+AI reporting guideline for prediction model studies. BMJ. 2024. https://www.bmj.com/content/385/bmj-2024-078378",
    "Steyerberg EW et al. Assessing the performance of prediction models: a framework for traditional and novel measures. Epidemiology. 2010. https://pubmed.ncbi.nlm.nih.gov/20010215/",
    "Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006. https://pubmed.ncbi.nlm.nih.gov/17099194/",
]


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def sig_label(p: float, q: float | None = None) -> str:
    value = q if q is not None and np.isfinite(q) else p
    if not np.isfinite(value):
        return "not evaluated"
    return "significant" if value < 0.05 else "not significant"


def write_summary(primary_auc: pd.DataFrame, secondary: pd.DataFrame, delong: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Statistical Significance Summary")
    lines.append("")
    lines.append("主分析采用患者级聚类 bootstrap；同一患者的所有随访节点在重采样时作为一个簇保留。DeLong 检验作为 AUROC 敏感性分析。")
    lines.append("")
    if primary_auc.empty:
        lines.append("未找到主分析 AUROC 结果。")
    else:
        lines.append("## Primary AUROC Results")
        lines.append("")
        lines.append("| Comparison | Target | AUROC A | AUROC B | Delta | 95% CI | P | q | Conclusion |")
        lines.append("|---|---:|---:|---:|---:|---|---:|---:|---|")
        for _, row in primary_auc.sort_values(["Target", "Model_A", "Model_B"]).iterrows():
            q_value = row.get("Q_Value_BH_All_AUROC", np.nan)
            lines.append(
                "| "
                f"{row['Model_A']} vs {row['Model_B']} | {row['Target']} | "
                f"{format_float(row['Model_A_Value'])} | {format_float(row['Model_B_Value'])} | "
                f"{format_float(row['Delta_A_minus_B'])} | "
                f"[{format_float(row['CI_95_Low'])}, {format_float(row['CI_95_High'])}] | "
                f"{format_p(row['P_Value'])} | {format_p(q_value)} | "
                f"{sig_label(row['P_Value'], q_value)} |"
            )
    if not secondary.empty:
        lines.append("")
        lines.append("## Secondary Metrics")
        lines.append("")
        lines.append("Brier score 作为补充校准分析，解释时不替代 AUROC 主终点。")
    if not delong.empty:
        lines.append("")
        lines.append("## DeLong Sensitivity")
        lines.append("")
        lines.append("DeLong 检验未显式处理同一患者多随访节点相关性，因此仅作为敏感性分析。")
    lines.append("")
    lines.append("## References")
    lines.append("")
    for item in REFERENCES:
        lines.append(f"- {item}")
    (RESULTS_DIR / "significance_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_methods(primary_auc: pd.DataFrame) -> None:
    n_bootstrap = "NA"
    if not primary_auc.empty and "N_Bootstrap_Requested" in primary_auc.columns:
        n_bootstrap = str(int(primary_auc["N_Bootstrap_Requested"].max()))
    text = f"""# Methods And Results Text

## Methods

Model discrimination was compared using paired analyses on the held-out test set. Because the prediction unit was a follow-up node and multiple nodes could come from the same patient, the primary inferential analysis used patient-level cluster bootstrap resampling. For each bootstrap replicate, patients were sampled with replacement and all available follow-up nodes from sampled patients were retained. Differences in AUROC were calculated as model A minus model B. Two-sided P values were estimated from the empirical bootstrap distribution, and 95% confidence intervals were calculated using percentile intervals. Benjamini-Hochberg false discovery rate correction was applied across the prespecified primary AUROC comparisons. The number of bootstrap replicates requested was {n_bootstrap}. Paired DeLong tests were additionally performed as sensitivity analyses for AUROC comparisons.

Secondary analyses compared Brier score using the same patient-level cluster bootstrap framework. All tests were two-sided with a nominal significance threshold of P<0.05; the primary interpretation used FDR-adjusted q<0.05.

## Results

See `primary_cluster_bootstrap_auc.csv`, `secondary_cluster_bootstrap_metrics.csv`, and `delong_auc_sensitivity.csv` for full numerical results.

## References

"""
    text += "\n".join(f"- {item}" for item in REFERENCES) + "\n"
    (RESULTS_DIR / "methods_and_results_for_paper.md").write_text(text, encoding="utf-8")


def write_docs_plan() -> None:
    plan = """# Statistical Significance Experiment Plan

## Objective

Assess whether model performance differences are statistically significant for pediatric liver transplant dynamic mortality prediction at 1 month, 1 year, and 5 years after each follow-up node.

## Primary Analysis

- Endpoint: AUROC.
- Comparison design: paired model comparison on common test nodes.
- Resampling unit: patient, not node, to account for repeated follow-up nodes.
- Test: two-sided patient-level cluster bootstrap of AUROC differences.
- Multiplicity: Benjamini-Hochberg FDR correction across prespecified primary AUROC comparisons.

## Secondary Analysis

- Metric: Brier score.
- Test: same patient-level cluster bootstrap framework.
## Sensitivity Analysis

- Paired DeLong test for AUROC.
- Interpreted as sensitivity analysis because standard DeLong assumes independent observations and does not explicitly model repeated follow-up nodes within patients.

## Prespecified Primary Comparisons

- XGBoost vs LSTM
- XGBoost vs RSF
- XGBoost vs Qwen3-4B SFT
- Qwen3-4B SFT vs Qwen3-4B baseline
- Qwen3-4B SFT vs Llama3.1-8B
- Qwen3-4B SFT vs Huatuo-O1-7B

## References

"""
    plan += "\n".join(f"- {item}" for item in REFERENCES) + "\n"
    (DOCS_DIR / "statistical_significance_experiment_plan.md").write_text(plan, encoding="utf-8")

    runbook = """# Execution Runbook

Run from the repository root.

```bash
python evaluation/statistical_significance/secondary/prepare_paired_predictions.py --pairs all
python evaluation/statistical_significance/secondary/run_cluster_bootstrap_tests.py --pairs primary --n-bootstrap 10000 --seed 20260528
python evaluation/statistical_significance/secondary/run_cluster_bootstrap_tests.py --pairs all --metrics auroc --n-bootstrap 10000 --seed 20260528 --output-prefix all_pairwise_cluster_bootstrap_auc
python evaluation/statistical_significance/secondary/run_delong_auc_tests.py --pairs all
python evaluation/statistical_significance/secondary/summarize_significance_results.py
python evaluation/statistical_significance/secondary/plot_significance_heatmaps.py
```

For a smoke test, replace `--n-bootstrap 10000` with `--n-bootstrap 100`.
"""
    (DOCS_DIR / "execution_runbook.md").write_text(runbook, encoding="utf-8")

    methods = """# Statistical methods

All statistical tests were two-sided, with a nominal significance threshold of P<0.05. Because multiple follow-up nodes could come from the same patient, differences in model performance were primarily assessed using paired patient-level cluster bootstrap resampling. AUROC was the primary performance metric. For each bootstrap replicate, patients were sampled with replacement and all follow-up nodes from sampled patients were retained. Differences in AUROC were summarized as model A minus model B with percentile 95% confidence intervals and empirical two-sided P values. Benjamini-Hochberg correction was applied across prespecified primary AUROC comparisons. Paired DeLong tests for correlated AUROC estimates were conducted as sensitivity analyses.
"""
    (DOCS_DIR / "statistical_methods.md").write_text(methods, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize statistical significance outputs.")
    parser.parse_args()
    ensure_output_dirs()
    primary_auc = load_csv(RESULTS_DIR / "primary_cluster_bootstrap_auc.csv")
    secondary = load_csv(RESULTS_DIR / "secondary_cluster_bootstrap_metrics.csv")
    delong = load_csv(RESULTS_DIR / "delong_auc_sensitivity.csv")
    write_docs_plan()
    write_summary(primary_auc, secondary, delong)
    write_methods(primary_auc)
    print(f"Wrote {RESULTS_DIR / 'significance_summary.md'}")
    print(f"Wrote {RESULTS_DIR / 'methods_and_results_for_paper.md'}")


if __name__ == "__main__":
    main()
