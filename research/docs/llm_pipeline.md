# LLM pipeline

## SFT

SFT 使用 Qwen3-4B LoRA。Prompt 和数据构建由 `llm/sft/build_sft_dataset.py` 完成，路径通过命令行参数或环境变量配置，并保留概率-only JSON 输出。推荐先运行患者级拆分工具，再运行 `train_qwen3_4b_sft.sh`。

## Inference

`llm/inference/inference_openai_compatible.py` 复用同一 Prompt 构建函数，调用 OpenAI-compatible 服务并严格解析 `{"1m": ..., "1y": ..., "5y": ...}`。解析失败不会伪造概率。

## GRPO

`llm/grpo/` 使用 TRL 官方 GRPOTrainer，输入由 `prepare_grpo_dataset.py` 从 SFT messages 与私有真实标签组成，奖励函数独立位于 `reward.py`。
