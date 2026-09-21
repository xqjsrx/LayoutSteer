"""定位过程可视化（纯缓存重放，无需 GPU / 模型）。

每个 (样本, 实体) 生成一张图，包含:
  Row 1: 测试原图 + 全部候选框(灰) + 投票框(按权重着色) + 簇划分(彩色边框)
         + 最终选中簇(青色粗框) + GT(绿色)
  Row 2: 全局 top-K 相似训练样本缩略图(标实体 GT 框与相似度)
  Row 3+: top-5 投票候选的局部布局 vs 最佳匹配模板局部布局

用法:
  python scripts/visualize_localization.py --dataset sroie --n 4
  python scripts/visualize_localization.py --dataset sroie --samples X00016469670
  python scripts/visualize_localization.py --dataset sroie --n 6 --filter wrong --entities total
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from layoutsteer.config import LocalizationConfig, OUTPUT_DIR
from layoutsteer.datasets import get_dataset
from layoutsteer.localization import (global_match, vote_and_cluster,
                                      cache_dirs, load_train_index)
from layoutsteer.localization.pipeline import (build_regions, union_box,
                                                template_anchor_y, _norm_y)
from layoutsteer.localization.layout_render import (
    bbox_center, get_local_bboxes, generate_local_layout)

CLUSTER_COLORS = ["tab:orange", "tab:purple", "tab:brown", "tab:pink",
                  "tab:olive", "tab:red", "tab:blue", "tab:gray"]


def load_cached(path):
    if not os.path.exists(path):
        return None
    arr = np.load(path)
    return None if arr.size == 0 else arr


def render_local_layout(layout_info, center_idx, loc_cfg):
    boxes = [it.box for it in layout_info["items"]]
    img = Image.open(layout_info["image_path"])
    img_size = img.size
    img.close()
    local_bboxes, region = get_local_bboxes(
        boxes, center_idx, loc_cfg.n_local_bboxes, img_size,
        wrap_gap_ratio=loc_cfg.wrap_gap_ratio)
    return generate_local_layout(local_bboxes, region)


def replay(sample_name, entity, test_layouts, train_layouts,
           train_embs, train_names, loc_cfg, cache_dirs):
    """从缓存重放该样本该实体的完整定位过程。"""
    test_emb = load_cached(
        os.path.join(cache_dirs["global"], f"{sample_name}.npy"))
    candidate_embs = load_cached(
        os.path.join(cache_dirs["local"], f"{sample_name}.npy"))
    if test_emb is None or candidate_embs is None:
        return None

    top_templates = global_match(test_emb, train_embs, train_names,
                                 loc_cfg.global_topk)

    template_embs, template_ids, anchor_ys = [], [], []
    for tid, _ in top_templates:
        emb = load_cached(
            os.path.join(cache_dirs["template"], f"{tid}__{entity}.npy"))
        if emb is not None:
            template_embs.append(emb)
            template_ids.append(tid)
            anchor_ys.append(template_anchor_y(train_layouts[tid], entity))
    if not template_embs:
        return None
    template_embs = np.stack(template_embs)

    items = test_layouts[sample_name]["items"]
    boxes = [it.box for it in items]
    # 与 pipeline 一致: 启用位置先验时传入锚点/候选归一化 y
    use_prior = loc_cfg.position_prior > 0
    extent = union_box(boxes)
    candidate_ys = np.array([_norm_y(b, extent) for b in boxes])
    votes, clusters = vote_and_cluster(
        template_embs, candidate_embs, boxes, loc_cfg,
        template_anchor_ys=np.array(anchor_ys) if use_prior else None,
        candidate_ys=candidate_ys if use_prior else None)

    # 每个候选的最佳匹配模板（用于局部对比展示）
    sim_matrix = candidate_embs @ template_embs.T  # [N_cand, N_tmpl]
    return {
        "top_templates": top_templates,
        "template_ids": template_ids,
        "votes": votes,
        "clusters": clusters,
        "sim_matrix": sim_matrix,
        "items": items,
        "boxes": boxes,
    }


def visualize(sample_name, entity, rep, test_layouts, train_layouts,
              loc_cfg, out_path):
    items, boxes = rep["items"], rep["boxes"]
    votes, clusters = rep["votes"], rep["clusters"]
    gt_indices = {i for i, it in enumerate(items) if it.entity == entity}
    best = clusters[0] if clusters else None
    pred_indices = best["bbox_indices"] if best else []
    n_show = min(5, len(votes))
    n_topk = len(rep["top_templates"])

    fig = plt.figure(figsize=(22, 14 + n_show * 3.2))
    gs = fig.add_gridspec(2 + n_show, max(n_topk, 6),
                          height_ratios=[8, 2.2] + [2.2] * n_show,
                          hspace=0.35, wspace=0.15)

    # ── Row 1: 测试图 + 候选/投票/簇/选中/GT ──
    test_img = Image.open(test_layouts[sample_name]["image_path"]).convert("RGB")
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(test_img)
    for x1, y1, x2, y2 in boxes:  # 全部候选: 灰
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, lw=0.5,
                               edgecolor="gray", facecolor="none", alpha=0.4))
    # 簇划分: 每簇一种颜色
    idx2cluster = {}
    for ci, c in enumerate(clusters):
        for bi in c["bbox_indices"]:
            idx2cluster[bi] = ci
    max_w = max((v["total_weight"] for v in votes), default=1.0)
    for v in votes:  # 投票框: 按权重填充红黄
        bi = v["bbox_idx"]
        x1, y1, x2, y2 = boxes[bi]
        w = v["total_weight"] / max_w
        ci = idx2cluster.get(bi, 0)
        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.5 + 1.5 * w,
                               edgecolor=color, facecolor=(1, 1 - w * 0.8, 0),
                               alpha=0.25 + 0.3 * w))
        ax.text(x1, y1 - 3, f"[{bi}] w={v['total_weight']:.2f} C{ci}",
                fontsize=6, color=color, fontweight="bold")
    for bi in pred_indices:  # 选中簇成员: 青色细虚线
        x1, y1, x2, y2 = boxes[bi]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.5,
                               edgecolor="cyan", facecolor="none", ls="--"))
    # 最终输出: 区域框（最多 max_regions 个），青色粗实线，与 pipeline 完全一致
    extent = union_box(boxes)
    for region, _ in build_regions(clusters, boxes, extent, loc_cfg) if clusters else []:
        rx1, ry1, rx2, ry2 = region
        ax.add_patch(Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, lw=4,
                               edgecolor="cyan", facecolor="cyan", alpha=0.12))
        ax.add_patch(Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, lw=4,
                               edgecolor="cyan", facecolor="none"))
    for gi in gt_indices:  # GT: 绿
        x1, y1, x2, y2 = boxes[gi]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2.5,
                               edgecolor="lime", facecolor="none"))
    for ci, c in enumerate(clusters[:5]):  # 簇中心
        centers = np.array([bbox_center(boxes[bi]) for bi in c["bbox_indices"]])
        cx, cy = centers.mean(axis=0)
        ax.plot(cx, cy, "*", color=CLUSTER_COLORS[ci % len(CLUSTER_COLORS)],
                markersize=16, markeredgecolor="black")
        ax.text(cx + 5, cy, f"C{ci} w={c['total_weight']:.2f} n={c['n_votes']}",
                fontsize=8, color="blue", fontweight="bold")

    top1_idx = pred_indices[0] if pred_indices else -1
    correct = top1_idx in gt_indices
    gt_rank = next((r + 1 for r, v in enumerate(votes)
                    if v["bbox_idx"] in gt_indices), -1)
    ax.set_title(
        f"{sample_name} / {entity}  |  top1={'CORRECT' if correct else 'WRONG'}"
        f"  gt_rank={gt_rank}  clusters={len(clusters)}  "
        f"pred=[{top1_idx}] '{items[top1_idx].text[:30] if top1_idx >= 0 else ''}'\n"
        f"gray=candidates  colored=votes(per-cluster color)  "
        f"cyan solid=OUTPUT regions(max {loc_cfg.max_regions})  green=GT  star=cluster center",
        fontsize=11, fontweight="bold")
    ax.axis("off")

    # ── Row 2: 全局 top-K 相似训练样本 ──
    for j, (tid, sim) in enumerate(rep["top_templates"]):
        axg = fig.add_subplot(gs[1, j])
        info = train_layouts.get(tid)
        if info:
            timg = Image.open(info["image_path"]).convert("RGB")
            axg.imshow(timg)
            for it in info["items"]:
                if it.entity == entity:
                    x1, y1, x2, y2 = it.box
                    axg.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2,
                                            edgecolor="lime", facecolor="none"))
        has_e = tid in rep["template_ids"]
        axg.set_title(f"#{j + 1} {tid[:13]}\nsim={sim:.4f}"
                      f"{'' if has_e else ' (no ' + entity + ')'}",
                      fontsize=7, color="black" if has_e else "red")
        axg.axis("off")

    # ── Row 3+: top-5 候选局部布局 vs 最佳模板局部布局 ──
    half = max(n_topk, 6) // 2
    for i in range(n_show):
        v = votes[i]
        bi = v["bbox_idx"]
        # 测试侧局部布局
        axl = fig.add_subplot(gs[2 + i, 0:half])
        test_local = render_local_layout(test_layouts[sample_name], bi, loc_cfg)
        axl.imshow(test_local)
        mark = " [GT]" if bi in gt_indices else ""
        axl.set_title(f"vote#{i + 1} cand[{bi}]{mark} '{items[bi].text[:28]}' "
                      f"w={v['total_weight']:.2f} votes={v['n_votes']}",
                      fontsize=8, fontweight="bold",
                      color="green" if bi in gt_indices else "black")
        axl.axis("off")
        # 最佳匹配模板局部布局
        t_best = int(np.argmax(rep["sim_matrix"][bi]))
        tid = rep["template_ids"][t_best]
        axr = fig.add_subplot(gs[2 + i, half:])
        info = train_layouts.get(tid)
        e_idx = next((k for k, it in enumerate(info["items"])
                      if it.entity == entity), None)
        if e_idx is not None:
            tmpl_local = render_local_layout(info, e_idx, loc_cfg)
            axr.imshow(tmpl_local)
        e_text = next((it.text for it in info["items"] if it.entity == entity), "")
        axr.set_title(f"best template {tid[:15]} sim={rep['sim_matrix'][bi, t_best]:.4f} "
                      f"ans='{e_text[:25]}'", fontsize=8)
        axr.axis("off")

    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="定位过程可视化（缓存重放）")
    parser.add_argument("--dataset", type=str, default="sroie")
    parser.add_argument("--samples", type=str, default="",
                        help="逗号分隔样本名；空则按 --n/--filter 自动选取")
    parser.add_argument("--n", type=int, default=4, help="自动选取的样本数")
    parser.add_argument("--entities", type=str, default="all")
    parser.add_argument("--filter", type=str, default="mixed",
                        choices=["all", "correct", "wrong", "mixed"],
                        help="按 top1 对错筛选样本 (mixed=对错各半)")
    args = parser.parse_args()

    loc_cfg = LocalizationConfig()
    ds = get_dataset(args.dataset)
    test_layouts = ds.load_layout_items("test")
    train_layouts = ds.load_layout_items("train")

    import json
    train_names = sorted(train_layouts.keys())
    dirs = cache_dirs(ds.name, loc_cfg)
    train_embs = load_train_index(dirs["train_global"], train_names)

    replay_dirs = {
        "global": dirs["test_global"],
        "local": dirs["test_local"],
        "template": dirs["template"],
    }
    entities = ds.entity_types if args.entities == "all" else args.entities.split(",")

    # 选样本: 指定名单，或按定位报告对错筛选
    if args.samples:
        chosen = {e: args.samples.split(",") for e in entities}
    else:
        report_path = os.path.join(OUTPUT_DIR, ds.name, "localization_report.json")
        with open(report_path) as f:
            per_sample = json.load(f)["per_sample"]
        chosen = {}
        for e in entities:
            corr = [s["sample_name"] for s in per_sample
                    if s["entities"].get(e, {}).get("top1_correct")]
            wrong = [s["sample_name"] for s in per_sample
                     if s["entities"].get(e) and not s["entities"][e]["top1_correct"]]
            if args.filter == "correct":
                chosen[e] = corr[:args.n]
            elif args.filter == "wrong":
                chosen[e] = wrong[:args.n]
            elif args.filter == "mixed":
                chosen[e] = corr[:args.n - args.n // 2] + wrong[:args.n // 2]
            else:
                chosen[e] = (corr + wrong)[:args.n]

    out_dir = os.path.join(OUTPUT_DIR, ds.name, "loc_vis")
    os.makedirs(out_dir, exist_ok=True)
    n_done = 0
    for entity in entities:
        for sample_name in chosen[entity]:
            rep = replay(sample_name, entity, test_layouts, train_layouts,
                         train_embs, train_names, loc_cfg, replay_dirs)
            if rep is None:
                print(f"跳过 {sample_name}/{entity}: 缓存不全")
                continue
            out_path = os.path.join(out_dir, f"{sample_name}_{entity}.png")
            visualize(sample_name, entity, rep, test_layouts, train_layouts,
                      loc_cfg, out_path)
            print(f"已生成: {out_path}")
            n_done += 1
    print(f"\n共 {n_done} 张，目录: {out_dir}")


if __name__ == "__main__":
    main()
