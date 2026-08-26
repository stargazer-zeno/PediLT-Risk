# 概率校准实验

[English](README.md)

本目录公开固定 XGBoost 风险模型的患者级概率校准研究代码、聚合结果、图表和 Platt 参数。主分析为 Platt scaling，Isotonic regression 仅用于敏感性分析。仓库不包含 EHR、患者级预测、患者划分、原始输入、模型序列化文件、日志或临床阈值配置。

本次分析使用 392 名患者拟合校准器，并在另外 789 名患者上进行保留测试。该设计是内部、患者不重叠的评估，不是外部验证；XGBoost 模型没有重新训练。

## 使用方式

请在仓库外的授权位置创建私有输入清单。必需预测表字段为 `Patient_ID`、`Sample_ID`、`Target`、`True_Label` 和 `Pred_Prob`。详细清单格式、运行命令和输出边界见 [English README](README.md)。

```bash
python evaluation/calibration/run_calibration_analysis.py \
  --input-manifest path/to/calibration.private.json \
  --output-dir path/to/private_calibration_output

python evaluation/calibration/verify_calibration_outputs.py \
  --input-manifest path/to/calibration.private.json \
  --output-dir path/to/private_calibration_output
```

运行输出含患者级中间文件，必须保存在仓库外的授权位置，不能提交到 Git。公开的 `parameters/platt_parameters.json` 仅用于研究复现，不会被预测系统加载，也不能直接作为临床部署配置。任何前瞻性使用均需独立验证、治理审批和本地重新校准。
