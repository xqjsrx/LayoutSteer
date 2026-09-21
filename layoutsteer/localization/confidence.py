"""定位置信度：把定位共识信号（best_weight）标定为区域覆盖概率。

置信度语义: c = P̂(干预区域覆盖 GT | 实体, best_weight 实体内分位)。
标定曲线（SROIE 全量报告）单调: 全体 Q0-20 -> 78%, Q80-100 -> 98%；
address/company 恒高（93~100%），date/total 从 ~60% 爬到 ~93-100%——
低置信任务的干预强度按 c 打折，减少"往打偏的区域搬注意力"的伤害。

标定表 JSON 格式:
  {entity: {"weights": [各分位桶的 best_weight 中位数], "cover": [桶内覆盖率]}}
查表: np.interp 线性插值（两端自动钳位）。
"""
import numpy as np

# 分位桶边界（实体内分位）
BUCKET_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def build_calibration(per_sample_report):
    """从定位报告构建标定表: 实体内 best_weight 分位桶 -> 区域覆盖率。"""
    rows = {}
    for sr in per_sample_report:
        for ent, m in sr["entities"].items():
            if m.get("has_gt") and "conf" in m:
                rows.setdefault(ent, []).append(
                    (m["conf"]["best_weight"], bool(m["region_covers_gt"])))

    table = {}
    for ent, data in rows.items():
        ws = np.array([w for w, _ in data])
        weights, cover = [], []
        for lo, hi in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
            w_lo, w_hi = np.quantile(ws, lo), np.quantile(ws, hi)
            sub = [(w, c) for w, c in data if w_lo <= w <= w_hi]
            if not sub:
                continue
            weights.append(float(np.median([w for w, _ in sub])))
            cover.append(float(np.mean([c for _, c in sub])))
        table[ent] = {"weights": weights, "cover": cover, "n": len(data)}
    return table


def confidence_lookup(table, entity, best_weight):
    """查表 + 线性插值得到置信度；无该实体标定时返回 1.0。"""
    ent = table.get(entity)
    if not ent or not ent["weights"]:
        return 1.0
    return float(np.interp(best_weight, ent["weights"], ent["cover"]))


def annotate_results(results, per_sample_report, table):
    """把置信度写入定位结果（结果条目新增 "confidence" 字段）。"""
    conf_map = {}
    for sr in per_sample_report:
        for ent, m in sr["entities"].items():
            if "conf" in m:
                conf_map[(sr["sample_name"], ent)] = m["conf"]["best_weight"]
    for r in results:
        key = (r["sample_name"], r["question_type"])
        if key in conf_map:
            r["confidence"] = round(
                confidence_lookup(table, key[1], conf_map[key]), 4)
    return results
