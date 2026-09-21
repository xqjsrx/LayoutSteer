"""定位匹配核心（纯 numpy）：供 localize.py 单次运行与 sweep 扫参共用。

EmbStore 将四类 embedding 缓存按需载入内存并记忆化——
扫参时同一份 embedding 被多组超参复用，仅重算检索/投票/聚类。
"""
import os

import numpy as np

from .global_match import global_match
from .local_match import vote_and_cluster
from .layout_render import bbox_center
from .store import cache_dirs, load_npy, load_train_index


class EmbStore:
    """embedding 缓存的内存记忆化加载器。"""

    def __init__(self, ds, loc_cfg, template_dir=None, model: str = "qwen25vl"):
        self.dirs = cache_dirs(ds.name, loc_cfg, model=model)
        if template_dir:
            # 自举模板库（注意力 anchor 自动标注）替换 GT 模板目录
            self.dirs = dict(self.dirs, template=template_dir)
        self.hint = f"--dataset {ds.name}"
        self._train_matrix = {}
        self._mem = {}

    def train_index(self, train_names):
        key = (len(train_names), train_names[0], train_names[-1])
        if key not in self._train_matrix:
            self._train_matrix[key] = load_train_index(
                self.dirs["train_global"], train_names, self.hint)
        return self._train_matrix[key]

    def _load(self, kind, filename):
        key = (kind, filename)
        if key not in self._mem:
            self._mem[key] = load_npy(
                os.path.join(self.dirs[kind], filename), self.hint)
        return self._mem[key]

    def test_global(self, sample_name):
        return self._load("test_global", f"{sample_name}.npy")

    def test_local(self, sample_name):
        return self._load("test_local", f"{sample_name}.npy")

    def template(self, train_id, entity):
        return self._load("template", f"{train_id}__{entity}.npy")


def union_box(boxes):
    """一组框的外接矩形 [x1, y1, x2, y2]。"""
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _center_in(box, region):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def _norm_y(box, extent):
    """框中心在文档内容范围内的归一化 y。"""
    return ((box[1] + box[3]) / 2 - extent[1]) / max(extent[3] - extent[1], 1)


def template_anchor_y(layout_info, entity):
    """模板实体锚点（与 embed 一致取首个实体框）的归一化 y；无该实体返回 None。"""
    items = layout_info["items"]
    e_idx = next((i for i, it in enumerate(items) if it.entity == entity), None)
    if e_idx is None:
        return None
    boxes = [it.box for it in items]
    return _norm_y(boxes[e_idx], union_box(boxes))


def build_regions(clusters, boxes, extent, loc_cfg):
    """从簇列表构造最终区域框列表（最多 max_regions 个）。

    每个入选簇独立处理: 剔除离群成员 -> 外接 -> padding。
    多区域输出避免了单外接框合并远簇时把簇间空白区域也框进来。
    返回 [(region_box, members), ...]，按簇权重降序。
    """
    selected = [clusters[0]]
    if loc_cfg.merge_weight_ratio > 0:
        thr = loc_cfg.merge_weight_ratio * clusters[0]["total_weight"]
        for c in clusters[1:]:
            if c["total_weight"] >= thr and len(selected) < loc_cfg.max_regions:
                selected.append(c)

    regions = []
    for cluster in selected:
        members = list(cluster["bbox_indices"])

        # 离群剔除: 压缩被远点拉大的外接框
        if loc_cfg.trim_mult > 0 and len(members) >= 3:
            centers = np.array([bbox_center(boxes[i]) for i in members])
            d = np.linalg.norm(centers - centers.mean(axis=0), axis=1)
            med = np.median(d) + 1e-6
            kept = [m for m, di in zip(members, d) if di <= loc_cfg.trim_mult * med]
            if kept:
                members = kept

        region = union_box([boxes[i] for i in members])

        # padding: 按文档内容尺寸比例外扩（裁剪到内容范围）
        if loc_cfg.region_pad > 0:
            px = loc_cfg.region_pad * (extent[2] - extent[0])
            py = loc_cfg.region_pad * (extent[3] - extent[1])
            region = [max(extent[0], region[0] - px), max(extent[1], region[1] - py),
                      min(extent[2], region[2] + px), min(extent[3], region[3] + py)]

        regions.append((region, members))

    return regions


def run_localization(ds, loc_cfg, store, test_layouts, train_names,
                     train_layouts=None, n_test: int = 0, verbose: bool = False,
                     memory_bank=None, mem_cfg=None, bootstrap_meta=None,
                     key_matcher=None):
    """全量定位匹配，返回 (results, per_sample_report)。

    最终输出的 bbox = 每入选簇一个区域外接框（最多 max_regions 个）。
    position_prior > 0 时需提供 train_layouts 以计算模板锚点 y。
    memory_bank 非空时（Pass 2）: Stage1 额外检索最多 mem_max_k 个记忆源
    （排除自身），Stage2 混入对应实体的伪模板（票权 × mem_weight）。
    key_matcher 非空时: 键从"精确同键"松弛为"语义相近键"（开放词表用），
    相似键模板票权按相似度打折。
    """
    test_names = sorted(test_layouts.keys())
    if n_test > 0:
        test_names = test_names[:n_test]
    train_embs = store.train_index(train_names)

    use_prior = loc_cfg.position_prior > 0 and train_layouts is not None
    anchor_y_cache = {}

    def get_anchor_y(tid, entity):
        key = (tid, entity)
        if key not in anchor_y_cache:
            anchor_y_cache[key] = template_anchor_y(train_layouts[tid], entity)
        return anchor_y_cache[key]

    results, report = [], []
    for si, sample_name in enumerate(test_names):
        layout_info = test_layouts[sample_name]
        items = layout_info["items"]
        boxes = [it.box for it in items]
        # 文档内容范围（全部 OCR 框的外接），用于归一化区域面积
        extent = union_box(boxes)
        extent_area = max((extent[2] - extent[0]) * (extent[3] - extent[1]), 1)

        test_emb = store.test_global(sample_name)
        top_templates = global_match(
            test_emb, train_embs, train_names, loc_cfg.global_topk)
        # 实体感知候补序列: 门控稀疏库（如自举库）下 top-k 邻居可能某实体
        # 全无模板——沿相似度序列向后补位凑满，消除覆盖缺口
        cand_templates = global_match(
            test_emb, train_embs, train_names,
            min(len(train_names), loc_cfg.global_topk * 10))
        # 记忆检索（Pass 2）: 全局最相似的记忆源样本，排除自身
        mem_top = (memory_bank.match_sources(
            test_emb, sample_name, mem_cfg.mem_max_k, store)
            if memory_bank is not None else [])

        candidate_embs = store.test_local(sample_name)
        if candidate_embs.shape[0] != len(items):
            if verbose:
                print(f"[警告] {sample_name}: 缓存候选数与布局元素数不符，跳过")
            continue

        sample_report = {"sample_name": sample_name, "entities": {}}
        candidate_ys = np.array([_norm_y(b, extent) for b in boxes])
        for entity in ds.sample_entities(sample_name):
            template_embs, anchor_ys, train_weights = [], [], []
            key_cands = (key_matcher.candidates(entity) if key_matcher
                         else [(entity, 1.0)])
            n_tids = 0
            for tid, _ in cand_templates:
                if n_tids >= loc_cfg.global_topk:
                    break
                # 精确键优先，未命中再退到相似键（票权按相似度打折）
                e, ekey, esim = None, entity, 1.0
                for k2, sim in key_cands:
                    e = store.template(tid, k2)
                    if e is not None:
                        ekey, esim = k2, sim
                        break
                if e is None:
                    continue
                n_tids += 1
                if bootstrap_meta is not None:
                    # 自举模板: [K, D] 多 anchor 候选全部展开参与投票，
                    # anchor_y 来自自举元数据（非 GT）
                    e2d = e if e.ndim == 2 else e[None]
                    ys = bootstrap_meta.get(f"{tid}__{ekey}", [])
                    for k in range(e2d.shape[0]):
                        template_embs.append(e2d[k])
                        anchor_ys.append(ys[k] if k < len(ys) else 0.0)
                        train_weights.append(esim)
                    continue
                template_embs.append(e)
                anchor_ys.append(get_anchor_y(tid, ekey) if use_prior else 0.0)
                train_weights.append(esim)
            n_train_templates = len(template_embs)
            # 混入伪模板: 局部 embedding = 源样本候选矩阵的 anchor_idx 行
            used_mem_sources = []
            for src, _ in mem_top:
                mem_e = memory_bank.entry(src, entity)
                if mem_e is None:
                    continue
                src_cands = store.test_local(src)
                if src_cands is None or mem_e["anchor_idx"] >= src_cands.shape[0]:
                    continue
                template_embs.append(src_cands[mem_e["anchor_idx"]])
                anchor_ys.append(mem_e["anchor_y"])
                used_mem_sources.append(src)
            n_mem_templates = len(template_embs) - n_train_templates
            if not template_embs:
                continue
            template_weights = None
            if n_mem_templates > 0 or any(w < 1.0 for w in train_weights):
                mem_w = mem_cfg.mem_weight if n_mem_templates > 0 else 1.0
                template_weights = np.array(
                    train_weights + [mem_w] * n_mem_templates)

            votes, clusters = vote_and_cluster(
                np.stack(template_embs), candidate_embs, boxes, loc_cfg,
                template_anchor_ys=np.array(anchor_ys) if use_prior else None,
                candidate_ys=candidate_ys if use_prior else None,
                template_weights=template_weights)
            if not clusters:
                continue
            pred_indices = clusters[0]["bbox_indices"]
            regions = build_regions(clusters, boxes, extent, loc_cfg)
            all_members = [m for _, members in regions for m in members]

            # 空间置信场（布局掩码的紧凑表示）: 全部得票框 + 归一化权重，
            # 推理侧栅格化到当次图像 token 网格后作逐 token 干预强度
            member_set = set(all_members)
            max_w = max(votes[0]["total_weight"], 1e-6)
            weight_map = [{
                "box": [int(v) for v in boxes[vt["bbox_idx"]]],
                "weight": round(vt["total_weight"] / max_w, 4),
                "in_region": vt["bbox_idx"] in member_set,
            } for vt in votes]

            results.append({
                "sample_name": sample_name,
                "question_type": entity,
                # 检索到的邻居样本（同时驱动 layout 掩码与 dK 记忆查询）
                "retrieved": [tid for tid, _ in top_templates],
                # 参与投票的在线记忆邻居（dK 查询的第二来源）
                "retrieved_mem": used_mem_sources,
                # 最终 bbox = 每个入选簇一个区域外接框（最多 max_regions 个）
                "matching_bboxes": [{
                    "text": " | ".join(items[i].text for i in members)[:120],
                    "box": [int(v) for v in region],
                    "entity": entity,
                } for region, members in regions],
                "member_boxes": [
                    {"text": items[i].text, "box": items[i].box}
                    for i in all_members],
                "weight_map": weight_map,
            })

            gt_indices = {i for i, it in enumerate(items) if it.entity == entity}
            gt_rank = next((r + 1 for r, v in enumerate(votes)
                            if v["bbox_idx"] in gt_indices), -1)
            region_area = sum((r[2] - r[0]) * (r[3] - r[1]) for r, _ in regions)
            sample_report["entities"][entity] = {
                "top1_correct": pred_indices[0] in gt_indices,
                "gt_in_best_cluster": bool(gt_indices & set(pred_indices)),
                # 任一区域框覆盖任一 GT 框中心（干预真正关心的指标）
                "region_covers_gt": any(
                    _center_in(boxes[g], region)
                    for g in gt_indices for region, _ in regions),
                # 面积 = 各区域框面积之和（多区域不计簇间空白）
                "region_area_ratio": region_area / extent_area,
                "gt_rank": gt_rank,
                "n_pred_boxes": len(all_members),
                "n_regions": len(regions),
                "has_gt": bool(gt_indices),
                # ── 置信度信号（决策时可得，无 GT，供记忆机制入库门控）──
                "conf": {
                    "global_top1_sim": float(top_templates[0][1]),
                    "n_templates": len(template_embs),
                    "n_mem_templates": n_mem_templates,
                    "best_weight": float(clusters[0]["total_weight"]),
                    "best_votes": int(clusters[0]["n_votes"]),
                    # 次簇/最优簇权重比（无次簇记 0，越小共识越强）
                    "second_ratio": (float(clusters[1]["total_weight"]
                                           / clusters[0]["total_weight"])
                                     if len(clusters) > 1 else 0.0),
                    "top_vote_weight": float(votes[0]["total_weight"]),
                    # 伪模板锚点 = top1 预测候选（入库后 Stage2 直接取行）
                    "anchor_idx": int(pred_indices[0]),
                    "anchor_y": float(candidate_ys[pred_indices[0]]),
                },
            }

        report.append(sample_report)
        if verbose and (si + 1) % 100 == 0:
            print(f"  [{si + 1}/{len(test_names)}]")

    return results, report


def summarize_report(per_sample):
    """按实体聚合定位指标。"""
    summary = {}
    for sr in per_sample:
        for entity, m in sr["entities"].items():
            if not m["has_gt"]:
                continue
            s = summary.setdefault(entity, {"n": 0, "top1": 0, "in_cluster": 0,
                                            "region_cover": 0, "area_sum": 0.0,
                                            "top3": 0, "top5": 0})
            s["n"] += 1
            s["top1"] += int(m["top1_correct"])
            s["in_cluster"] += int(m["gt_in_best_cluster"])
            s["region_cover"] += int(m.get("region_covers_gt", False))
            s["area_sum"] += m.get("region_area_ratio", 0.0)
            s["top3"] += int(0 < m["gt_rank"] <= 3)
            s["top5"] += int(0 < m["gt_rank"] <= 5)

    for s in summary.values():
        n = max(s["n"], 1)
        s["top1_acc"] = s["top1"] / n
        s["in_cluster_acc"] = s["in_cluster"] / n
        s["region_cover_acc"] = s["region_cover"] / n
        s["mean_area_ratio"] = s["area_sum"] / n
        s["top3_acc"] = s["top3"] / n
        s["top5_acc"] = s["top5"] / n
    return summary


def macro_avg(summary, metric):
    """实体宏平均指标。"""
    vals = [s[metric] for s in summary.values()]
    return sum(vals) / len(vals) if vals else 0.0
