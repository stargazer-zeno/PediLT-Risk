# Evaluation and statistical analysis

主指标是 AUROC；最终共同队列要求同一目标下七个模型均有有效标签和概率。

总体和阶段AUROC的主置信区间使用 patient-level cluster bootstrap，以患者为重采样单位，保留同一患者多次随访节点的相关性。`evaluation/statistical_significance/scripts/` 中的脚本执行配对 cluster bootstrap、FDR 校正和阶段热图所需统计量。

补充统计代码在 `evaluation/statistical_significance/secondary/` 中提供 Brier score 和 DeLong 敏感性分析。这些脚本同样只读取私有患者级预测，不携带原始结果。

稳定性报告位于 `evaluation/metrics/model_stability_assessment_report.md`，解释跨阶段点估计波动和 CI 宽度；CI 宽不应直接等同于模型不稳定。
