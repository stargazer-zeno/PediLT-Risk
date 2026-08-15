# PediLT-Risk

[English](README.md)

PediLT-Risk 是一个面向小儿肝移植术后动态死亡风险预测的开源科研项目，包含可部署的预测系统、模型文件以及机器学习和大语言模型的训练与评估代码。

> **仅限科研用途。** 本软件不是医疗器械，不得用于诊断、治疗或个体临床决策。详见 [DISCLAIMER.md](DISCLAIMER.md)。

## 仓库结构

- [`apps/risk-prediction-system`](apps/risk-prediction-system)：FastAPI + React 预测系统，内置三个 XGBoost 模型，可选接入 vLLM。
- [`research`](research)：训练、推理、评估、Prompt、聚合指标和合成示例。

系统分别输出未来 1 个月、1 年和 5 年死亡风险。ML 与 LLM 两条分支独立展示，不进行加权或融合；LLM 返回的缺失概率保持为 `null`，不会使用 ML 结果回填。

## 快速部署

```bash
cd apps/risk-prediction-system
cp .env.example .env
docker compose up -d --build
```

浏览器访问 <http://localhost:8080>。默认启动前端、后端和内置 XGBoost 模型；如需连接外部 OpenAI-compatible vLLM，在 `.env` 中设置 `LLM_BASE_URL` 与 `LLM_MODEL_NAME`。

可选的本地 NVIDIA GPU vLLM：

```bash
docker compose -f compose.yaml -f compose.vllm.yaml --profile llm up -d --build
```

LLM 模型地址：[zeno156/PediLT-Risk-Qwen3-4B](https://huggingface.co/zeno156/PediLT-Risk-Qwen3-4B)。

## 数据声明

原始临床病例为私有数据，不随仓库发布。公开内容仅包括聚合评估结果和明确标注的合成示例。复现完整队列指标需要另行取得原始数据授权。

代码采用 [Apache-2.0](LICENSE) 许可证；医疗与科研限制见 [DISCLAIMER.md](DISCLAIMER.md)。
