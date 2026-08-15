# PediLT-Risk 实验与研究代码

[English](README.md)

本目录提供小儿肝移植术后随访节点级死亡风险预测的可复现代码，比较 XGBoost、LSTM、RSF 和多种大语言模型在未来 1 个月、1 年和 5 年预测任务上的表现。

公开内容仅包括代码、配置模板、聚合结果和完全合成的示例，不包含真实 EHR、患者级预测、原始 LLM 日志、SFT 数据或训练权重。

## 目录说明

- `machine_learning/train`：预处理、XGBoost、LSTM、RSF 和汇总评估。
- `llm/sft`：SFT 数据构建、患者级划分和 Qwen3-4B LoRA 启动器。
- `llm/grpo`：GRPO 训练与概率奖励函数。
- `llm/inference`：OpenAI-compatible 推理客户端。
- `prompts`：SFT、GRPO 和推理 Prompt 契约。
- `evaluation`：聚合指标与患者级 cluster bootstrap 分析。
- `machine_learning/examples`、`llm/examples`：明确标注的合成样例。

## 数据与特征契约

每个样本对应一个随访节点，只使用该节点之前可获得的信息；训练和测试按患者划分。

- 静态特征：142
- 每个时间步的时序特征：88
- 最大序列长度：256
- XGBoost/RSF 展平维度：22,927
- 目标：`Label_1m`、`Label_1y`、`Label_5y`

通过 `PEDILT_DATA_DIR` 及 `configs/paths.env.example` 中说明的其他环境变量设置仓库外的授权数据位置，不得把真实病例放入本仓库。

公开微调模型：[zeno156/PediLT-Risk-Qwen3-4B](https://huggingface.co/zeno156/PediLT-Risk-Qwen3-4B)。模型输出 `null` 是合法结果，不得使用 ML 预测值补全。

聚合指标位于 `evaluation/metrics/`。完整训练和指标复现需要另行授权的原始临床数据及患者级预测文件。

序列表示、输入契约和输出目录见 `machine_learning/train/TRAINING_PROTOCOL.md`。

```bash
python -m unittest discover -s tests -v
python -m compileall llm machine_learning evaluation tests
```
