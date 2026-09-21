"""OCR-free 对称化定位: 训练侧与测试侧统一使用形态学伪文本行框渲染。

相对 run_ocrfree_localize.py（仅测试侧 morph, 训练侧 GT 模板）的改动:
  train_global  = 训练样本 morph 框全局布局 embedding
  template      = 以"距 GT 实体框中心最近的 morph 框"为锚点的局部布局
                  （训练侧用 GT 选锚点是允许的; 全链路零标注需换自举注意力锚）
  anchor_y      经 bootstrap_meta 通道传入 run_localization（管线零改动）
测试侧 morph embedding 直接复用 run_ocrfree_localize 的缓存。

产出:
  output/sroie/localized_bboxes_sroie_morphsym.json
  output/sroie/ocrfree_symmetric_report.json

用法: CUDA_VISIBLE_DEVICES=0 python scripts/run_ocrfree_symmetric.py \
        [--shard 0/3 --embed-only]
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from layoutsteer.config import (ModelConfig, LocalizationConfig, CACHE_DIR,
                                OUTPUT_DIR, morph_tag)
from layoutsteer.datasets import get_dataset
from layoutsteer.model_loader import setup_seeds, load_model_and_processor
from layoutsteer.localization import PostMergerEmbedder
from layoutsteer.localization.layout_render import (
    generate_layout, get_local_bboxes, generate_local_layout, bbox_center)
from layoutsteer.localization.pipeline import (EmbStore, run_localization,
                                               union_box, _norm_y)
from layoutsteer.localization.store import save_npy

from run_ocrfree_localize import build_morph_layouts, evaluate

BATCH_SINGLE = 16


def build_template_meta(gt_train_layouts, morph_train_layouts, entity_types):
    """(tid, entity) -> (morph 锚点框索引, 归一化 anchor_y)；无 GT 实体为 None。"""
    meta = {}
    for tid, morph_info in morph_train_layouts.items():
        gt_items = gt_train_layouts[tid]["items"]
        morph_boxes = [it.box for it in morph_info["items"]]
        extent = union_box(morph_boxes)
        centers = np.array([bbox_center(b) for b in morph_boxes])
        for entity in entity_types:
            gt_box = next((it.box for it in gt_items if it.entity == entity),
                          None)
            if gt_box is None:
                meta[(tid, entity)] = None
                continue
            gc = np.array(bbox_center(gt_box))
            idx = int(np.argmin(np.linalg.norm(centers - gc, axis=1)))
            meta[(tid, entity)] = (idx, _norm_y(morph_boxes[idx], extent))
    return meta


def embed_train_side(morph_train_layouts, tpl_meta, loc_cfg, tg_dir, tpl_dir,
                     entity_types, shard=""):
    """补算训练侧 morph 全局 + 模板 embedding（逐文件缓存, 分片安全）。"""
    names = sorted(morph_train_layouts.keys())
    g_jobs = [n for n in names
              if not os.path.exists(os.path.join(tg_dir, f"{n}.npy"))]
    t_jobs = [(n, e) for n in names for e in entity_types
              if not os.path.exists(os.path.join(tpl_dir, f"{n}__{e}.npy"))]
    # 无 GT 实体的模板: 直接落空标记, 不占 GPU
    empty = [(n, e) for n, e in t_jobs if tpl_meta[(n, e)] is None]
    for n, e in empty:
        save_npy(os.path.join(tpl_dir, f"{n}__{e}.npy"), None)
    t_jobs = [j for j in t_jobs if tpl_meta[j] is not None]
    if shard:
        si, sn = map(int, shard.split("/"))
        g_jobs = g_jobs[si::sn]
        t_jobs = t_jobs[si::sn]
        print(f"分片 {shard}: train_global {len(g_jobs)} / template "
              f"{len(t_jobs)} 条")
    if not g_jobs and not t_jobs:
        print("训练侧 morph embedding 缓存全部命中")
        return

    model, processor = load_model_and_processor(
        ModelConfig(), for_localization=True)
    embedder = PostMergerEmbedder(model, processor, loc_cfg.global_prompt)
    t0 = time.time()

    for c0 in range(0, len(g_jobs), BATCH_SINGLE):
        chunk = g_jobs[c0:c0 + BATCH_SINGLE]
        imgs = []
        for n in chunk:
            info = morph_train_layouts[n]
            size = Image.open(info["image_path"]).size
            imgs.append(generate_layout(size,
                                        [it.box for it in info["items"]]))
        for n, e in zip(chunk, embedder.extract_batch(imgs)):
            save_npy(os.path.join(tg_dir, f"{n}.npy"), e)
        print(f"train_global {min(c0 + BATCH_SINGLE, len(g_jobs))}"
              f"/{len(g_jobs)}", flush=True)

    for c0 in range(0, len(t_jobs), BATCH_SINGLE):
        chunk = t_jobs[c0:c0 + BATCH_SINGLE]
        imgs = []
        for n, e in chunk:
            info = morph_train_layouts[n]
            boxes = [it.box for it in info["items"]]
            size = Image.open(info["image_path"]).size
            anchor_idx, _ = tpl_meta[(n, e)]
            lb, region = get_local_bboxes(
                boxes, anchor_idx, loc_cfg.n_local_bboxes, size,
                loc_cfg.wrap_gap_ratio, loc_cfg.use_wrap)
            imgs.append(generate_local_layout(lb, region))
        for (n, e), emb in zip(chunk, embedder.extract_batch(imgs)):
            save_npy(os.path.join(tpl_dir, f"{n}__{e}.npy"), emb)
        if (c0 // BATCH_SINGLE) % 10 == 0:
            done = min(c0 + BATCH_SINGLE, len(t_jobs))
            elapsed = time.time() - t0
            print(f"template {done}/{len(t_jobs)} ({elapsed:.0f}s)",
                  flush=True)
    embedder.remove_hook()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dilate-ratio", type=float, default=0.04)
    parser.add_argument("--shard", type=str, default="")
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--dataset", type=str, default="sroie")
    parser.add_argument("--global-only", action="store_true",
                        help="只算 train_global（自举 strict 链路所需的唯一产物）; "
                             "跳过需要 GT 实体标签的模板 embedding")
    args = parser.parse_args()

    setup_seeds()
    ds = get_dataset(args.dataset)
    loc_cfg = LocalizationConfig()
    tag = morph_tag(args.dilate_ratio)
    g_dir = os.path.join(CACHE_DIR, f"{ds.name}_test_global_{tag}")
    l_dir = os.path.join(CACHE_DIR,
                         f"{ds.name}_test_local_{loc_cfg.local_tag}_{tag}")
    tg_dir = os.path.join(CACHE_DIR, f"{ds.name}_train_global_{tag}")
    tpl_dir = os.path.join(CACHE_DIR,
                           f"{ds.name}_template_local_{loc_cfg.local_tag}_{tag}")
    for d in (tg_dir, tpl_dir):
        os.makedirs(d, exist_ok=True)

    gt_test_layouts = ds.load_layout_items("test")
    gt_train_layouts = ds.load_layout_items("train")
    train_names = sorted(gt_train_layouts.keys())

    print("形态学提框（train + test）...")
    morph_train = build_morph_layouts(ds, train_names, gt_train_layouts,
                                      args.dilate_ratio)
    morph_test = build_morph_layouts(ds, sorted(gt_test_layouts.keys()),
                                     gt_test_layouts, args.dilate_ratio)
    train_names = sorted(morph_train.keys())
    print(f"train {len(morph_train)} / test {len(morph_test)} 样本有效")

    if args.global_only:
        # 自举 strict 链路只需 train_global；模板 embedding 依赖 GT 实体标签，跳过
        embed_train_side(morph_train, {}, loc_cfg, tg_dir, tpl_dir,
                         [], shard=args.shard)
        print(f"train_global morph embedding 完成 -> {tg_dir}")
        return

    tpl_meta = build_template_meta(gt_train_layouts, morph_train,
                                   ds.entity_types)
    embed_train_side(morph_train, tpl_meta, loc_cfg, tg_dir, tpl_dir,
                     ds.entity_types, shard=args.shard)
    if args.embed_only:
        print("embedding 分片完成")
        return

    store = EmbStore(ds, loc_cfg)
    store.dirs = dict(store.dirs, test_global=g_dir, test_local=l_dir,
                      train_global=tg_dir, template=tpl_dir)
    # anchor_y 经 bootstrap_meta 通道注入（模板为 [D], 管线自动扩为 [1,D]）
    boot_meta = {f"{n}__{e}": [m[1]]
                 for (n, e), m in tpl_meta.items() if m is not None}

    print("定位匹配...")
    results, _ = run_localization(ds, loc_cfg, store, morph_test,
                                  train_names,
                                  train_layouts=gt_train_layouts,
                                  verbose=True, bootstrap_meta=boot_meta)

    gt_bboxes = ds.load_bboxes("gt")
    per_entity = evaluate(results, gt_bboxes)
    macro = float(np.mean([s["region_cover_acc"]
                           for s in per_entity.values()]))

    asym_path = os.path.join(OUTPUT_DIR, ds.name,
                             "ocrfree_localization_report.json")
    asym = (json.load(open(asym_path)) if os.path.exists(asym_path) else {})
    report = {"n_samples": len(morph_test),
              "dilate_ratio": args.dilate_ratio,
              "per_entity": per_entity,
              "macro_region_cover": macro,
              "asymmetric_morph": {
                  e: s["region_cover_acc"]
                  for e, s in asym.get("per_entity", {}).items()},
              "baseline_gt_boxes": asym.get("baseline_gt_boxes", {})}
    out_boxes = os.path.join(
        OUTPUT_DIR, ds.name, f"localized_bboxes_{ds.name}_morphsym.json")
    out_report = os.path.join(OUTPUT_DIR, ds.name,
                              "ocrfree_symmetric_report.json")
    with open(out_boxes, "w") as f:
        json.dump(results, f)
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"-> {out_boxes}\n-> {out_report}")


if __name__ == "__main__":
    main()
