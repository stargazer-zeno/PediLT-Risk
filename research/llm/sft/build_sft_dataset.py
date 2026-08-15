"""Build probability-only SFT data for Qwen3-4B.

Clinical records and out-of-fold predictions are external inputs. The command
line interface validates these inputs and writes patient-level message records
without copying source data into the repository.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= 配置 =================
RESEARCH_ROOT = os.environ.get("PEDILT_RESEARCH_ROOT")
if RESEARCH_ROOT:
    RESEARCH_ROOT = os.path.abspath(RESEARCH_ROOT)
else:
    RESEARCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DEFAULT_DATA_ROOT = os.environ.get(
    "PEDILT_DATA_DIR", os.path.join(RESEARCH_ROOT, "data")
)
DEFAULT_TRAIN_JSON = os.path.join(DEFAULT_DATA_ROOT, "train_dataset_gold.json")
DEFAULT_OOF_CSV = os.path.join(
    DEFAULT_DATA_ROOT, "xgb_sequence_train_group_oof_predictions.csv"
)
DEFAULT_OUTPUT_JSONL = os.path.join(
    RESEARCH_ROOT, "artifacts", "sft", "xgb_sft_train.jsonl"
)

SYSTEM_PROMPT = "You are a helpful assistant."

# ================= Prompt 模板 =================

USER_PREFIX_TEMPLATE = """你是一位有丰富经验的小儿肝移植医生。

### 【任务说明】
请基于以下患者截至最后一次随访时点的全部历史资料，对其后续1个月、1年及5年内发生死亡事件的风险进行前瞻性预测，给出对应的死亡风险预测概率值。
--------------------------------------------------
{SUMMARY_TEXT}
【患者病历数据】"""

USER_SUFFIX_TEMPLATE = """--------------------------------------------------

### 【临床指标判读规范与参考范围】

在分析时，可以结合患者年龄、性别以及术后时间，对各项检验指标进行判断。

#### 1. 免疫抑制剂浓度判定规则
涉及免疫抑制剂血药浓度时，请先确认检验项名称及同日用药记录，再选择对应参考范围：
- "免疫抑制剂浓度"如对应当日用药为"他克莫司"，按他克莫司目标浓度范围判断
- "免疫抑制剂浓度"如对应当日用药为"环孢素"，按环孢素目标浓度范围判断
- 检验项若明确写为"雷帕浓度"，统一按他克莫司目标浓度范围判断

#### 2. 检验项目正常参考范围表
| 检验项目 | 单位 | 适用年龄段 | 正常参考范围 (男) | 正常参考范围 (女) |
|---|---|---|---|---|
| WBC | 10^9/L | 28 天～<6 月 | 4.3～14.2 | 同左 |
| WBC | 10^9/L | 6 月～<1 岁 | 4.8～14.6 | 同左 |
| WBC | 10^9/L | 1 岁～<2 岁 | 5.1～14.1 | 同左 |
| WBC | 10^9/L | 2 岁～<6 岁 | 4.4～11.9 | 同左 |
| WBC | 10^9/L | 6 岁～<13 岁 | 4.3～11.3 | 同左 |
| WBC | 10^9/L | 13 岁～18 岁 | 4.1～11.0 | 同左 |
| N(%) | % | 28 天～<6 月 | 7～56 | 同左 |
| N(%) | % | 6 月～<1 岁 | 9～57 | 同左 |
| N(%) | % | 1 岁～<2 岁 | 13～55 | 同左 |
| N(%) | % | 2 岁～<6 岁 | 22～65 | 同左 |
| N(%) | % | 6 岁～<13 岁 | 31～70 | 同左 |
| N(%) | % | 13 岁～18 岁 | 37～77 | 同左 |
| 淋巴细胞绝对值 | 10^9/L | 28 天～<6 月 | 2.4～9.5 | 同左 |
| 淋巴细胞绝对值 | 10^9/L | 6 月～<1 岁 | 2.5～9.0 | 同左 |
| 淋巴细胞绝对值 | 10^9/L | 1 岁～<2 岁 | 2.4～8.7 | 同左 |
| 淋巴细胞绝对值 | 10^9/L | 2 岁～<6 岁 | 1.8～6.3 | 同左 |
| 淋巴细胞绝对值 | 10^9/L | 6 岁～<13 岁 | 1.5～4.6 | 同左 |
| 淋巴细胞绝对值 | 10^9/L | 13 岁～18 岁 | 1.2～3.8 | 同左 |
| 嗜酸性粒细胞百分比 | % | 28 天～<2 岁 | 0.00～0.10 | 同左 |
| 嗜酸性粒细胞百分比 | % | 2 岁～18 岁 | 0.00～0.07 | 同左 |
| HB | g/L | 28 天～<6 月 | 97～183 | 同左 |
| HB | g/L | 6 月～<1 岁 | 97～141 | 同左 |
| HB | g/L | 1 岁～<2 岁 | 107～141 | 同左 |
| HB | g/L | 2 岁～<6 岁 | 112～149 | 同左 |
| HB | g/L | 6 岁～<13 岁 | 118～156 | 同左 |
| HB | g/L | 13 岁～18 岁 | 129～172 | 114～154 |
| PLT | 10^9/L | 28 天～<6 月 | 183～614 | 同左 |
| PLT | 10^9/L | 6 月～<1 岁 | 190～579 | 同左 |
| PLT | 10^9/L | 1 岁～<2 岁 | 190～524 | 同左 |
| PLT | 10^9/L | 2 岁～<6 岁 | 188～472 | 同左 |
| PLT | 10^9/L | 6 岁～<13 岁 | 167～453 | 同左 |
| PLT | 10^9/L | 13 岁～18 岁 | 150～407 | 同左 |
| TP | g/L | 28 天～<6 月 | 49～71 | 同左 |
| TP | g/L | 6 月～<1 岁 | 55～75 | 同左 |
| TP | g/L | 1 岁～<2 岁 | 58～76 | 同左 |
| TP | g/L | 2 岁～<6 岁 | 61～79 | 同左 |
| TP | g/L | 6 岁～<13 岁 | 65～84 | 同左 |
| TP | g/L | 13 岁～18 岁 | 68～88 | 同左 |
| ALB | g/L | 28 天～<6 月 | 35～50 | 同左 |
| ALB | g/L | 6 月～<13 岁 | 39～54 | 同左 |
| ALB | g/L | 13 岁～18 岁 | 42～56 | 同左 |
| ALT | U/L | 28 天～<1 岁 | 8～71 | 同左 |
| ALT | U/L | 1岁 月～<2 岁 | 8～42 | 同左 |
| ALT | U/L | 2 岁～<13 岁 | 7～30 | 同左 |
| ALT | U/L | 13 岁～18 岁 | 7～43 | 6～29 |
| AST | U/L | 28 天～<1 岁 | 21～80 | 同左 |
| AST | U/L | 1岁 月～<2 岁 | 22～59 | 同左 |
| AST | U/L | 2 岁～<13 岁 | 14～44 | 同左 |
| AST | U/L | 13 岁～18 岁 | 12～37 | 10～31 |
| ALP | U/L | 28 天～<6 月 | 98～532 | 同左 |
| ALP | U/L | 6 月～<1 岁 | 106～420 | 同左 |
| ALP | U/L | 1 岁～<2 岁 | 128～432 | 同左 |
| ALP | U/L | 2 岁～<9 岁 | 143～406 | 同左 |
| ALP | U/L | 9 岁～<12 岁 | 146～500 | 同左 |
| ALP | U/L | 12 岁～<14 岁 | 160～610 | 81～454 |
| ALP | U/L | 14 岁～<15 岁 | 82～603 | 63～327 |
| ALP | U/L | 15 岁～<17 岁 | 64～443 | 52～215 |
| ALP | U/L | 17 岁～18 岁 | 51～202 | 43～130 |
| γ-GT | U/L | 28 天～<6 月 | 9～150 | 同左 |
| γ-GT | U/L | 6 月～<1 岁 | 6～31 | 同左 |
| γ-GT | U/L | 1 岁～<13 岁 | 5～19 | 同左 |
| γ-GT | U/L | 13 岁～18 岁 | 8～40 | 6～26 |
| DB | umol/L | ～18 岁 | 0-6.84 | 同左 |
| TB | umol/L | ～18 岁 | 0-23 | 同左 |
| 胆汁酸 | umol/L | ～18 岁 | 0.01-10 | 同左 |
| CR | umol/L | 28 天～＜2 岁 | 13～33 | 同左 |
| CR | umol/L | 2 岁～＜6 岁 | 19～44 | 同左 |
| CR | umol/L | 6 岁～＜13 岁 | 27～66 | 同左 |
| CR | umol/L | 13 岁～＜16 岁 | 37～93 | 33～75 |
| CR | umol/L | 16 岁～18 岁 | 52～101 | 39～76 |
| 血糖 | mmol/L | ～18 岁 | 3.9-6.1 | 同左 |
| 甘油三脂 | mmol/L | ～18 岁 | 适宜：＜1.7，增高：1.7-2.3，很高：＞2.3 | 同左 |
| 总胆固醇 | mmol/L | ～18 岁 | 适宜：＜5.2，增高：5.2-6.2，很高：＞6.2 | 同左 |
| 尿酸 | umol/L | ～18 岁 | 155-428 | 同左 |
| PT | s | ～18 岁 | 9.4-12.5 | 同左 |
| INR | 无 | ～18 岁 | 0.8-1.15 | 同左 |
| 血氨 | umol/L | ～18 岁 | 9-30 | 同左 |
| CMV-DNA | copies/mL | ～18 岁 | ＜400 | 同左 |
| EBV-DNA | copies/mL | ～18 岁 | ＜400 | 同左 |
| HBsAg | COI | ～18 岁 | ＜1 | 同左 |
| HBsAb | mIU/mL | ～18 岁 | ＜10 | 同左 |
| HBeAg | COI | ～18 岁 | ＜1 | 同左 |
| HBeAb | COI | ～18 岁 | ＞1 | 同左 |
| HBcAb | COI | ～18 岁 | ＞1 | 同左 |
| HBV-DNA | IU/mL | ～18 岁 | ＜20 | 同左 |
| NT-proBNP | pg/mL | ～18 岁 | 0-125 | 同左 |
| 甲胎蛋白 | ng/mL | ～18 岁 | 0-7 | 同左 |

#### 3. 免疫抑制剂目标血药浓度参考表
| 项目 | 单位 | 年龄(在此填术后时间) | 男 | 女 |
|---|---|---|---|---|
| 他克莫司浓度 | ng/ml | 术后第1个月内 | 8～12 | 同左 |
|  |  | 术后第2～6个月 | 7～10 | 同左 |
|  |  | 术后第6～12个月 | 5～8 | 同左 |
|  |  | 术后12个月以后 | 5左右 | 同左 |
| 雷帕浓度 | ng/ml | 术后第1个月内 | 8～12 | 同左 |
|  |  | 术后第2～6个月 | 7～10 | 同左 |
|  |  | 术后第6～12个月 | 5～8 | 同左 |
|  |  | 术后12个月以后 | 5左右 | 同左 |
| 环孢素A浓度(C0) | ng/ml | 术后第1个月内 | 150～200 | 同左 |
|  |  | 术后第2～6个月 | 120～150 | 同左 |
|  |  | 术后第6～12个月 | 100～120 | 同左 |
|  |  | 术后12个月以后 | 100左右 | 同左 |
| 环孢素A浓度(C2) | ng/ml | 术后第1个月内 | 1000～1200 | 同左 |
|  |  | 术后第2～6个月 | 800～1000 | 同左 |
|  |  | 术后第6～12个月 | 500～800 | 同左 |
|  |  | 术后12个月以后 | 500左右 | 同左 |


### 【输出指令】
请只输出一个严格的 JSON 对象，包含该患者从最后一次随访算起在未来三个时间窗口内的死亡风险概率。概率必须是 0.0-1.0 之间的浮点数。数值 > 0.5 表示预测结果为在该时间窗口内发生死亡，数值 < 0.5 表示预测结果为存活，数值越大则代表死亡风险越高。

JSON 结构必须严格如下：
{
    "1m": <未来 1 个月内发生死亡的概率>,
    "1y": <未来 1 年内发生死亡的概率>,
    "5y": <未来 5 年内发生死亡的概率>
}
"""

FORBIDDEN_KEYWORDS = ["死", "死亡", "去世", "离世", "亡故", "die", "Die"]


def contains_leakage(text):
    text_str = str(text)
    return any(kw in text_str for kw in FORBIDDEN_KEYWORDS)


def format_ehr_and_get_summary(node_data):
    base_info = node_data.get("基础信息", {}).copy()
    labs = node_data.get("时序检验指标 (纯数值序列)", [])
    meds = node_data.get("时序用药记录 (纯数值序列)", [])
    events = node_data.get("临床事件", [])

    summary_text = base_info.pop("随访记录摘要", "该患者无随访记录摘要。")

    parts = ["【基础信息】"]
    for k, v in base_info.items():
        if not contains_leakage(k) and not contains_leakage(v):
            parts.append(f"{k}: {v}")

    parts.append("\n【时序检验指标】")
    parts.extend(labs if labs else ["无检验记录"])

    parts.append("\n【时序用药记录】")
    parts.extend(meds if meds else ["无用药记录"])

    parts.append("\n【临床事件】")
    clean_events = [e for e in events if not contains_leakage(e)]
    parts.extend(clean_events if clean_events else ["无临床事件记录"])

    formatted_ehr = "\n".join(parts)
    return summary_text, formatted_ehr


def build_final_prompt(node_data):
    summary_text, clean_ehr = format_ehr_and_get_summary(node_data)
    prefix = USER_PREFIX_TEMPLATE.replace("{SUMMARY_TEXT}", summary_text)
    return f"{prefix}\n{clean_ehr}\n{USER_SUFFIX_TEMPLATE}"


# ================= OOF 加载 =================

def load_oof(oof_csv_path):
    df = pd.read_csv(oof_csv_path)
    lookup = {}
    for _, row in df.iterrows():
        sid = str(row["Sample_ID"])
        lookup[sid] = {
            "1m": None if pd.isna(row["OOF_Prob_1m"]) else round(float(row["OOF_Prob_1m"]), 4),
            "1y": None if pd.isna(row["OOF_Prob_1y"]) else round(float(row["OOF_Prob_1y"]), 4),
            "5y": None if pd.isna(row["OOF_Prob_5y"]) else round(float(row["OOF_Prob_5y"]), 4),
        }
    return lookup


# ================= 概率校验 =================

def validate_probs(probs, true_labels):
    """
    逐时间窗口校验 OOF 概率方向与真实标签是否一致。
    规则：prob > 0.5 预测为死亡，prob < 0.5 预测为存活。
      - 预测方向与真实标签一致 → 保留原概率
      - 预测方向与真实标签矛盾 → 标记整条样本为矛盾（has_contradiction=True）
      - 任一侧为 None（截尾或 XGBoost 无预测） → 置为 None（无法校验）
    只要任意一个时间窗口存在方向矛盾，整条样本将被丢弃。
    """
    validated = {}
    has_contradiction = False
    stats = {"kept": 0, "discarded": 0, "unchecked": 0}

    for h in ["1m", "1y", "5y"]:
        oof_prob = probs.get(h)
        true_label = true_labels.get(h)

        if oof_prob is None or true_label is None:
            validated[h] = None
            stats["unchecked"] += 1
        else:
            pred_death = oof_prob > 0.5
            actual_death = (true_label == 1)
            if pred_death == actual_death:
                validated[h] = oof_prob
                stats["kept"] += 1
            else:
                validated[h] = None
                has_contradiction = True  # 整条样本标记为矛盾
                stats["discarded"] += 1

    return validated, has_contradiction, stats

# ================= 主流程 =================

def build_sample(node_data, oof_lookup):
    node_id = str(node_data.get("id", ""))
    probs = oof_lookup.get(node_id, {"1m": None, "1y": None, "5y": None})

    raw_labels = node_data.get("真实标签", {})
    true_labels = {h: raw_labels.get(h) for h in ["1m", "1y", "5y"]}

    validated_probs, has_contradiction, stats = validate_probs(probs, true_labels)

    user_content = build_final_prompt(node_data)
    assistant_content = json.dumps(validated_probs, ensure_ascii=False)
    return {
        "id": node_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }, has_contradiction, stats, all(v is None for v in validated_probs.values())


def main():
    parser = argparse.ArgumentParser(
        description="Create probability-only SFT JSONL from patient-level XGBoost OOF predictions."
    )
    parser.add_argument("--train-json", default=DEFAULT_TRAIN_JSON)
    parser.add_argument("--oof-csv", default=DEFAULT_OOF_CSV)
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL)
    args = parser.parse_args()

    print(f"📂 加载 OOF 预测: {args.oof_csv}")
    oof_lookup = load_oof(args.oof_csv)
    print(f"✅ 加载了 {len(oof_lookup)} 条 OOF 预测记录")

    print(f"📂 加载训练集 JSON: {args.train_json}")
    with open(args.train_json, "r", encoding="utf-8") as f:
        all_nodes = json.load(f)
    print(f"✅ 加载了 {len(all_nodes)} 条 EHR 记录")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)

    total_kept = total_discarded = total_unchecked = 0
    contradiction_count = all_null_count = miss_count = written = 0

    with open(args.output_jsonl, "w", encoding="utf-8") as f_out:
        for node in tqdm(all_nodes, desc="Building SFT samples", ncols=100):
            node_id = str(node.get("id", ""))
            if node_id not in oof_lookup:
                miss_count += 1

            sample, has_contradiction, stats, is_all_null = build_sample(node, oof_lookup)

            total_kept      += stats["kept"]
            total_discarded += stats["discarded"]
            total_unchecked += stats["unchecked"]

            if has_contradiction:
                contradiction_count += 1
                continue  # 任意窗口方向矛盾，丢弃整条样本

            if is_all_null:
                all_null_count += 1
                continue  # 三个时间窗口全为 null，无有效信号

            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n💾 SFT 数据集已保存至: {args.output_jsonl}")
    print(f"📊 原始记录数:         {len(all_nodes)}")
    print(f"✅ 写入样本数:         {written}")
    print(f"❌ 含矛盾窗口丢弃:     {contradiction_count}（任一窗口方向矛盾即丢弃整条）")
    print(f"⬜ 全空丢弃:           {all_null_count}（三窗口均无有效预测）")
    print(f"\n── 时间窗口级校验统计（共 {len(all_nodes)*3} 个窗口）──")
    print(f"  ✅ 通过校验（方向一致）:    {total_kept}")
    print(f"  ❌ 未通过校验（方向矛盾）:  {total_discarded}")
    print(f"  ⬜ 无法校验（截尾或无OOF）: {total_unchecked}")
    if miss_count:
        print(f"  ⚠️  OOF CSV 无匹配记录:    {miss_count}")


if __name__ == "__main__":
    main()
