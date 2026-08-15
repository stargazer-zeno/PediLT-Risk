# Data pipeline

原始输入是患者级划分的随访节点 JSON。一个节点包含基础信息、时间序列随访基础信息、数值检验序列、用药序列、临床事件。

LLM侧使用 `llm/sft/build_sft_dataset.py`：先移除可能包含结局的字段/关键词，再把检验、用药和事件拼接成 Prompt。SFT assistant 标签来自 XGBoost OOF 连续概率；无可用结局的时间窗保留 `null`，三个窗口全为 `null` 的样本不写入训练集。

ML侧使用 `machine_learning/train/preprocessing/build_sequence_datasets.py`：训练集独立构建 schema，静态分类变量 one-hot，纵向值按原始时间步解析，缺失值保持 NaN，序列右对齐并用 mask 标识有效步。训练/测试患者交集会触发错误。

真实数据和患者级标签不属于本仓库；`llm/examples/patient_raw_example.json` 是完全合成的 schema 展示。
