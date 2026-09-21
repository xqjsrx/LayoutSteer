"""记忆机制（Memory Bank）：高置信定位结果升格为伪模板加入检索库。

流程（两遍批量式）:
  Pass 1  纯训练库定位 -> report 携带 conf 置信信号（anchor_idx / best_weight 等）
  入库    build_memory_bank 按实体分位门控筛选高置信条目
  Pass 2  train ∪ memory 混合检索重定位（run_localization 传入 memory_bank）

四项防护:
  1. 自排除:   定位样本 X 时跳过 source==X 的记忆条目（防评估作弊）
  2. 单轮入库: 仅 Pass 1 结果入库，Pass 2 不回灌（防级联污染）
  3. 降权:     伪模板投票权 × mem_weight（它不是 GT 模板）
  4. 配额:     Stage1 记忆模板最多 mem_max_k 个（保 GT 模板多数票源）

伪模板三项资产全部复用既有缓存（零 GPU 成本）:
  全局 embedding      = test_global/{source}.npy
  实体局部 embedding  = test_local/{source}.npy 第 anchor_idx 行（top1 投票候选）
  锚点归一化 y        = Pass 1 conf 中记录的 anchor_y（位置先验用）
"""
import numpy as np


def build_memory_bank(per_sample_report, mem_cfg, entity_types):
    """从 Pass 1 报告门控筛选入库条目。

    门槛 = 各实体自身 best_weight 分布的 gate_weight_quantile 分位
    （按实体分别设阈，避免只有好定位的实体入库）。
    返回可 JSON 序列化的 bank dict。
    """
    candidates = {e: [] for e in entity_types}
    for sr in per_sample_report:
        for ent, m in sr["entities"].items():
            conf = m.get("conf")
            if conf is None or "anchor_idx" not in conf:
                continue
            candidates.setdefault(ent, []).append((sr["sample_name"], m))

    entries, thresholds = {}, {}
    for ent, cand in candidates.items():
        if not cand:
            entries[ent] = []
            thresholds[ent] = 0.0
            continue
        ws = np.array([m["conf"]["best_weight"] for _, m in cand])
        thr = float(np.quantile(ws, mem_cfg.gate_weight_quantile))
        thresholds[ent] = thr
        sel = []
        for name, m in cand:
            c = m["conf"]
            if c["best_weight"] < thr:
                continue
            if mem_cfg.gate_single_region and m["n_regions"] != 1:
                continue
            sel.append({
                "source": name,
                "anchor_idx": int(c["anchor_idx"]),
                "anchor_y": float(c["anchor_y"]),
                "best_weight": float(c["best_weight"]),
                "n_regions": int(m["n_regions"]),
            })
        entries[ent] = sel

    return {
        "gate": {"weight_quantile": mem_cfg.gate_weight_quantile,
                 "single_region": mem_cfg.gate_single_region},
        "thresholds": thresholds,
        "entries": entries,
    }


class MemoryBank:
    """入库条目的检索侧封装：全局索引 + (source, entity) 伪模板查询。"""

    def __init__(self, bank_dict):
        self.entries = bank_dict["entries"]
        self.thresholds = bank_dict.get("thresholds", {})
        self.sources = sorted({e["source"] for lst in self.entries.values()
                               for e in lst})
        self._by_key = {(e["source"], ent): e
                        for ent, lst in self.entries.items() for e in lst}
        self._gmat = None

    def __len__(self):
        return sum(len(v) for v in self.entries.values())

    def stats(self):
        return {ent: len(lst) for ent, lst in sorted(self.entries.items())}

    def match_sources(self, test_emb, exclude_sample, topk, store):
        """Stage1: 全局最相似的 topk 记忆源样本（排除自身）。"""
        if not self.sources or topk <= 0:
            return []
        if self._gmat is None:
            self._gmat = np.stack(
                [store.test_global(s) for s in self.sources])
        sims = self._gmat @ test_emb
        out = []
        for i in np.argsort(sims)[::-1]:
            s = self.sources[i]
            if s == exclude_sample:
                continue
            out.append((s, float(sims[i])))
            if len(out) >= topk:
                break
        return out

    def entry(self, source, entity):
        """(source, entity) 的入库条目；未入库返回 None。"""
        return self._by_key.get((source, entity))


def compare_passes(report1, report2):
    """Pass1 vs Pass2 逐条对比：修复/退化分解（top1 与 region_cover 两个口径）。"""
    m1 = {(sr["sample_name"], ent): m
          for sr in report1 for ent, m in sr["entities"].items()
          if m.get("has_gt")}
    diff = {}
    for sr in report2:
        for ent, m2 in sr["entities"].items():
            key = (sr["sample_name"], ent)
            m1e = m1.get(key)
            if m1e is None or not m2.get("has_gt"):
                continue
            d = diff.setdefault(ent, {"n": 0,
                                      "top1_fixed": 0, "top1_regressed": 0,
                                      "cover_fixed": 0, "cover_regressed": 0})
            d["n"] += 1
            d["top1_fixed"] += int(not m1e["top1_correct"] and m2["top1_correct"])
            d["top1_regressed"] += int(m1e["top1_correct"] and not m2["top1_correct"])
            d["cover_fixed"] += int(not m1e["region_covers_gt"]
                                    and m2["region_covers_gt"])
            d["cover_regressed"] += int(m1e["region_covers_gt"]
                                        and not m2["region_covers_gt"])
    return diff
