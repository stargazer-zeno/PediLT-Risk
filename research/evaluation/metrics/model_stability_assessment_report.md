# 模型性能稳定性评估报告

数据源：`evaluation/metrics/final_overall_and_stage_auroc_common_cohort.csv`

## 方法

本分析使用最终统一 common cohort 的分阶段 AUROC 结果。AUROC 的主 95% CI 来自 patient-level cluster bootstrap percentile interval，即以患者为重采样单位，而不是以随访节点为独立单位。该方法能够保留同一患者多次随访记录之间的相关性，因此较普通节点级置信区间更保守。

模型稳定性从两个维度评估：第一，跨阶段 AUROC 点估计的波动；第二，跨阶段 95% CI 宽度所反映的估计不确定性。CI 宽说明该阶段性能估计不精确，不能直接等同于模型本身性能不稳定。

## QC

- 分阶段记录数：105 / 预期 105。
- 目标-阶段-模型重复记录数：0。
- CI 顺序与边界检查：通过。
- CI 方法：Patient-level cluster bootstrap percentile。

## 公开结果文件

- `evaluation/metrics/stage_auroc_with_ci_width.csv`：分阶段 AUROC、95% CI、CI 宽度和低事件标记。
- `evaluation/metrics/model_stability_summary_by_target.csv`：按预测时间窗和模型汇总的稳定性指标。

## 稳定性摘要

| Target | Model | AUROC mean | AUROC SD | AUROC range | Mean CI width | Max CI width | Low-event stages | Stability grade |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1m | XGBoost | 0.913 | 0.039 | 0.095 | 0.139 | 0.196 | 2 | High stability |
| 1m | LSTM | 0.878 | 0.056 | 0.137 | 0.224 | 0.275 | 2 | Moderate stability |
| 1m | RSF | 0.850 | 0.079 | 0.215 | 0.234 | 0.407 | 2 | Moderate stability |
| 1m | Qwen3-4B SFT | 0.883 | 0.055 | 0.138 | 0.172 | 0.269 | 2 | Moderate stability |
| 1m | Qwen3-4B baseline | 0.598 | 0.065 | 0.161 | 0.296 | 0.446 | 2 | Uncertain stability |
| 1m | Llama3.1-8B | 0.543 | 0.035 | 0.078 | 0.139 | 0.179 | 2 | High stability |
| 1m | Huatuo-O1-7B | 0.533 | 0.052 | 0.121 | 0.180 | 0.239 | 2 | Moderate stability |
| 1y | XGBoost | 0.792 | 0.084 | 0.213 | 0.224 | 0.427 | 0 | Moderate stability |
| 1y | LSTM | 0.691 | 0.048 | 0.110 | 0.320 | 0.437 | 0 | Uncertain stability |
| 1y | RSF | 0.726 | 0.054 | 0.139 | 0.263 | 0.382 | 0 | Uncertain stability |
| 1y | Qwen3-4B SFT | 0.733 | 0.114 | 0.285 | 0.209 | 0.270 | 0 | Uncertain stability |
| 1y | Qwen3-4B baseline | 0.607 | 0.025 | 0.067 | 0.171 | 0.318 | 0 | Moderate stability |
| 1y | Llama3.1-8B | 0.551 | 0.033 | 0.085 | 0.098 | 0.146 | 0 | High stability |
| 1y | Huatuo-O1-7B | 0.526 | 0.028 | 0.063 | 0.102 | 0.160 | 0 | High stability |
| 5y | XGBoost | 0.748 | 0.059 | 0.138 | 0.277 | 0.492 | 0 | Uncertain stability |
| 5y | LSTM | 0.681 | 0.070 | 0.178 | 0.287 | 0.506 | 0 | Uncertain stability |
| 5y | RSF | 0.673 | 0.061 | 0.163 | 0.268 | 0.412 | 0 | Uncertain stability |
| 5y | Qwen3-4B SFT | 0.669 | 0.027 | 0.061 | 0.267 | 0.376 | 0 | Uncertain stability |
| 5y | Qwen3-4B baseline | 0.582 | 0.041 | 0.106 | 0.160 | 0.306 | 0 | Moderate stability |
| 5y | Llama3.1-8B | 0.529 | 0.022 | 0.056 | 0.093 | 0.129 | 0 | High stability |
| 5y | Huatuo-O1-7B | 0.520 | 0.012 | 0.030 | 0.078 | 0.134 | 0 | High stability |

## CI 最宽的阶段-模型组合

| Target | Stage | Stage label | Model | Patients | Positive N | AUROC | 95% CI | CI width |
|---|---|---|---|---:|---:|---:|---:|---:|
| 5y | Stage5 | >2y | LSTM | 356 | 117 | 0.772 | 0.415-0.921 | 0.506 |
| 5y | Stage5 | >2y | XGBoost | 356 | 117 | 0.824 | 0.459-0.952 | 0.492 |
| 1m | Stage5 | >2y | Qwen3-4B baseline | 848 | 34 | 0.645 | 0.412-0.858 | 0.446 |
| 1y | Stage5 | >2y | LSTM | 763 | 77 | 0.727 | 0.453-0.890 | 0.437 |
| 1y | Stage4 | 1y-2y | XGBoost | 866 | 134 | 0.716 | 0.487-0.914 | 0.427 |
| 5y | Stage5 | >2y | RSF | 356 | 117 | 0.765 | 0.465-0.877 | 0.412 |
| 1m | Stage4 | 1y-2y | RSF | 968 | 38 | 0.831 | 0.554-0.960 | 0.407 |
| 1y | Stage4 | 1y-2y | RSF | 866 | 134 | 0.662 | 0.457-0.838 | 0.382 |
| 1y | Stage4 | 1y-2y | LSTM | 866 | 134 | 0.665 | 0.445-0.825 | 0.380 |
| 5y | Stage4 | 1y-2y | LSTM | 489 | 126 | 0.594 | 0.390-0.770 | 0.380 |

## 结果解释

本研究通过分阶段 AUROC 点估计及其 95% 置信区间评估模型性能稳定性。AUROC 点估计反映模型在不同术后阶段的判别能力，跨阶段 AUROC 标准差和范围反映性能水平的波动；95% CI 宽度反映该阶段性能估计的不确定性。由于本研究采用患者级 cluster bootstrap 计算置信区间，同一患者多次随访记录不会被错误视为完全独立样本，因此置信区间相对保守。部分晚期阶段或短期死亡预测窗口阳性事件数较少，导致有效信息量不足，进而出现较宽的 95% CI。该现象提示这些阶段的 AUROC 估计精度有限，不能简单等同于模型本身性能不稳定。
