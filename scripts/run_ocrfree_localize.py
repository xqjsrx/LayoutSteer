"""OCR-free 端到端定位: 测试侧全部使用形态学伪文本行框（零标注）。

测试侧 morph 框 -> 全局布局图 + 局部候选布局 -> embedding（独立缓存目录）
-> 复用 run_localization（训练侧 GT 模板不变）-> 用 GT 答案框仅做评估:
   region_cover = 任一区域框覆盖任一 GT 答案框中心（与基线报告同口径）。

产出:
  output/sroie/localized_bboxes_sroie_morph.json
  output/sroie/ocrfree_localization_report.json（含基线对比）

用法: CUDA_VISIBLE_DEVICES=0 python scripts/run_ocrfree_localize.py [--n 0]
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image

from layoutsteer.config import (ModelConfig, LocalizationConfig, CACHE_DIR,
                                OUTPUT_DIR, morph_tag)
from layoutsteer.datasets import get_dataset
from layoutsteer.datasets.base import LayoutItem
from layoutsteer.model_loader import setup_seeds, load_model_and_processor
from layoutsteer.localization import PostMergerEmbedder
from layoutsteer.localization.layout_render import (
    generate_layout, get_local_bboxes, generate_local_layout)
from layoutsteer.localization.pipeline import EmbStore, run_localization
from layoutsteer.localization.store import save_npy

from visualize_ocrfree import extract_text_boxes

BATCH_SINGLE, BATCH_CAND = 16, 8


def build_morph_layouts(ds, names, gt_layouts, dilate_ratio):
    """测试样本 -> 形态学框布局（entity 全部 other, 零标注）。"""
    layouts = {}
    for name in names:
        info = gt_layouts[name]
        img = Image.open(info["image_path"]).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        boxes = extract_text_boxes(bgr, dilate_ratio=dilate_ratio)
        img.close()
        if not boxes:
            continue
        layouts[name] = {
            "image_path": info["image_path"],
            "items": [LayoutItem(text="", box=b, entity="other")
                      for b in boxes],
        }
    return layouts


def embed_morph(morph_layouts, loc_cfg, g_dir, l_dir, shard=""):
    """补算缺失的 morph 全局/局部 embedding（磁盘缓存，可断点续跑）。

    shard="i/n": 仅处理第 i 份任务（原子写缓存，多卡分片并发安全）。
    """
    names = sorted(morph_layouts.keys())
    g_jobs = [n for n in names
              if not os.path.exists(os.path.join(g_dir, f"{n}.npy"))]
    l_jobs = [n for n in names
              if not os.path.exists(os.path.join(l_dir, f"{n}.npy"))]
    if shard:
        si, sn = map(int, shard.split("/"))
        g_jobs = g_jobs[si::sn]
        l_jobs = l_jobs[si::sn]
        print(f"分片 {shard}: global {len(g_jobs)} / local {len(l_jobs)} 条")
    if not g_jobs and not l_jobs:
        print("morph embedding 缓存全部命中")
        return

    model, processor = load_model_and_processor(
        ModelConfig(), for_localization=True)
    embedder = PostMergerEmbedder(model, processor, loc_cfg.global_prompt)
    t0 = time.time()

    for c0 in range(0, len(g_jobs), BATCH_SINGLE):
        chunk = g_jobs[c0:c0 + BATCH_SINGLE]
        imgs = []
        for n in chunk:
            info = morph_layouts[n]
            size = Image.open(info["image_path"]).size
            imgs.append(generate_layout(size,
                                        [it.box for it in info["items"]]))
        for n, e in zip(chunk, embedder.extract_batch(imgs)):
            save_npy(os.path.join(g_dir, f"{n}.npy"), e)
        print(f"global {min(c0 + BATCH_SINGLE, len(g_jobs))}/{len(g_jobs)}",
              flush=True)

    import torch
    for i, n in enumerate(l_jobs):
        info = morph_layouts[n]
        boxes = [it.box for it in info["items"]]
        size = Image.open(info["image_path"]).size
        cand_imgs = []
        for ci in range(len(boxes)):
            lb, region = get_local_bboxes(
                boxes, ci, loc_cfg.n_local_bboxes, size,
                loc_cfg.wrap_gap_ratio, loc_cfg.use_wrap)
            cand_imgs.append(generate_local_layout(lb, region))
        embs = []
        for c0 in range(0, len(cand_imgs), BATCH_CAND):
            embs.extend(embedder.extract_batch(cand_imgs[c0:c0 + BATCH_CAND]))
        save_npy(os.path.join(l_dir, f"{n}.npy"), np.stack(embs))
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(l_jobs) - i - 1)
            print(f"local {i + 1}/{len(l_jobs)} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)
            torch.cuda.empty_cache()
    embedder.remove_hook()


def evaluate(results, gt_bboxes):
    """region_cover 口径: 任一区域框覆盖任一 GT 答案框中心。"""
    per_entity = {}
    for r in results:
        key = (r["sample_name"], r["question_type"])
        gts = gt_bboxes.get(key)
        if not gts:
            continue
        regions = [mb["box"] for mb in r["matching_bboxes"]]
        cover = any(
            reg[0] <= (g[0] + g[2]) / 2 <= reg[2]
            and reg[1] <= (g[1] + g[3]) / 2 <= reg[3]
            for g in gts for reg in regions)
        s = per_entity.setdefault(r["question_type"],
                                  {"n": 0, "cover": 0})
        s["n"] += 1
        s["cover"] += int(cover)
    for s in per_entity.values():
        s["region_cover_acc"] = s["cover"] / max(s["n"], 1)
    return per_entity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0, help="0=全量")
    parser.add_argument("--dilate-ratio", type=float, default=0.04)
    parser.add_argument("--shard", type=str, default="",
                        help="i/n 多卡分片(仅 embedding 阶段)")
    parser.add_argument("--embed-only", action="store_true",
                        help="只算 embedding, 不做定位匹配")
    parser.add_argument("--dataset", type=str, default="sroie")
    args = parser.parse_args()

    setup_seeds()
    ds = get_dataset(args.dataset)
    loc_cfg = LocalizationConfig()
    tag = morph_tag(args.dilate_ratio)
    g_dir = os.path.join(CACHE_DIR, f"{ds.name}_test_global_{tag}")
    l_dir = os.path.join(CACHE_DIR,
                         f"{ds.name}_test_local_{loc_cfg.local_tag}_{tag}")
    os.makedirs(g_dir, exist_ok=True)
    os.makedirs(l_dir, exist_ok=True)

    gt_layouts = ds.load_layout_items("test")
    train_layouts = ds.load_layout_items("train")
    train_names = sorted(train_layouts.keys())
    names = sorted(gt_layouts.keys())
    if args.n > 0:
        names = names[:args.n]

    print("形态学提框...")
    morph_layouts = build_morph_layouts(ds, names, gt_layouts,
                                        args.dilate_ratio)
    print(f"{len(morph_layouts)}/{len(names)} 样本有效")

    embed_morph(morph_layouts, loc_cfg, g_dir, l_dir, shard=args.shard)
    if args.embed_only:
        print("embedding 分片完成")
        return

    store = EmbStore(ds, loc_cfg)
    store.dirs = dict(store.dirs, test_global=g_dir, test_local=l_dir)

    print("定位匹配...")
    results, _ = run_localization(ds, loc_cfg, store, morph_layouts,
                                  train_names, train_layouts=train_layouts,
                                  verbose=True)

    gt_bboxes = ds.load_bboxes("gt")
    per_entity = evaluate(results, gt_bboxes)
    macro = float(np.mean([s["region_cover_acc"]
                           for s in per_entity.values()]))

    # 基线（GT 框定位）同口径对比
    baseline = {}
    base_path = os.path.join(OUTPUT_DIR, ds.name, "localization_report.json")
    if os.path.exists(base_path):
        base = json.load(open(base_path))
        for ent, s in base.get("summary", {}).items():
            if isinstance(s, dict) and "region_cover_acc" in s:
                baseline[ent] = s["region_cover_acc"]

    report = {"n_samples": len(morph_layouts),
              "dilate_ratio": args.dilate_ratio,
              "per_entity": per_entity,
              "macro_region_cover": macro,
              "baseline_gt_boxes": baseline}
    out_boxes = os.path.join(
        OUTPUT_DIR, ds.name, f"localized_bboxes_{ds.name}_morph.json")
    out_report = os.path.join(OUTPUT_DIR, ds.name,
                              "ocrfree_localization_report.json")
    with open(out_boxes, "w") as f:
        json.dump(results, f)
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"-> {out_boxes}\n-> {out_report}")


if __name__ == "__main__":
    main()
