# CareNL 预测系统

[English](README.md)

该系统面向小儿肝移植术后随访死亡风险科研评估。用户上传标准化纵向记录后，系统分别输出未来 1 个月、1 年和 5 年的 XGBoost 与 LLM 风险概率。

> 仅限科研与软件演示，不用于诊断、治疗、预后沟通或个体临床决策。

## 系统结构

```text
React/nginx -> FastAPI -> 输入适配器
                         ├─ 内置 XGBoost 模型
                         └─ 可选 OpenAI-compatible vLLM API
```

ML 与 LLM 结果始终独立展示，不进行加权或融合。LLM 某个时间点返回 `null` 时，API、页面和下载文件均保留 `null`，不会使用 ML 结果回填。

## 机器学习模型

后端镜像内置经过校验的 schema 和三个 XGBoost 模型。校验值和 22,927 维特征契约记录在 [`artifacts/model_manifest.json`](artifacts/model_manifest.json)。训练使用的私有病例及患者级预测结果不公开。

## Docker 部署

### ML-only 或连接外部 vLLM

```bash
cp .env.example .env
docker compose up -d --build
```

- 页面：<http://localhost:8080>
- 后端：<http://localhost:8000>
- API 文档：<http://localhost:8000/docs>

接入外部 OpenAI-compatible 服务时设置：

```dotenv
LLM_BASE_URL=https://your-vllm-host.example/v1
LLM_API_KEY=EMPTY
LLM_MODEL_NAME=zeno156/PediLT-Risk-Qwen3-4B
LLM_PROMPT_MODE=sft_pattern_prob
```

`LLM_BASE_URL` 留空即以 ML-only 模式运行。

### 可选本地 vLLM

本地模式要求 Linux Docker、NVIDIA 驱动、NVIDIA Container Toolkit，以及满足上下文长度要求的 GPU 显存。

```bash
# Hugging Face 模型需要鉴权时才设置 HF_TOKEN。
export HF_TOKEN="replace-with-hugging-face-token"
docker compose -f compose.yaml -f compose.vllm.yaml --profile llm up -d --build
```

vLLM API 映射到 <http://localhost:8001/v1>。可通过 `VLLM_MAX_MODEL_LEN` 和 `VLLM_TENSOR_PARALLEL_SIZE` 调整资源配置。

## 输入与样例

支持 JSON、JSONL、CSV 和 XLSX。`frontend/public/samples/` 中仅保留两个明确标注的合成样例，不包含真实身份、结局标签或精确临床日期。

上传记录和结果仅保存在本机 Docker volume。若部署到共享网络，必须另行配置身份认证、TLS、访问控制、留存策略并通过机构隐私审查。

## 测试

```bash
cd backend
python -m pytest

cd ../frontend
npm ci
npm test
npm run build
```
