"""morph 自举资产准备: anchors 几何转移 + 自举模板 morph 重建。

1. anchors 转移（CPU）: anchors{,_test}.json 的 anchor_idxs 指向 GT 布局框;
   转移规则 = GT 锚框中心 -> 最近 morph 框索引（近似原始注意力峰对齐,
   原始峰像素坐标未存档; 严格版需重跑 bootstrap_anchors 捕获）,
   anchor_y 按 morph 框/extent 重算, 去重后写 anchors{,_test}_morph.json。
2. 自举模板重建（GPU）: 过 lp 门控的 (train, entity) 用 morph 框渲染
   锚点局部布局 -> embedding [K,D], 存 cache/sroie_bootstrap_template_
   {local_tag}_morph04/。dK bank 与框无关, 直接复用。

用法:
  python scripts/prep_morph_bootstrap.py --anchors-only          # 步骤1
  CUDA_VISIBLE_DEVICES=4 python scripts/prep_morph_bootstrap.py --shard 0/2
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from layoutsteer.config import (ModelConfig, LocalizationConfig, CACHE_DIR,
                                bootstrap_dir, morph_tag)
from layoutsteer.datasets import get_dataset
from layoutsteer.model_loader import setup_seeds, load_model_and_processor
from layoutsteer.localization import PostMergerEmbedder
from layoutsteer.localization.store import save_npy
from layoutsteer.localization.layout_render import (
    get_local_bboxes, generate_local_layout, bbox_center)
from layoutsteer.localization.pipeline import union_box, _norm_y

from run_ocrfree_localize import build_morph_layouts

LP_THR = -0.05


def transfer_anchors(anchors, gt_layouts, morph_layouts):
    """GT 锚框中心 -> 最近 morph 框索引; anchor_y 按 morph extent 重算。"""
    out = {}
    for name, ents in anchors.items():
        gi = gt_layouts.get(name)
        mi = morph_layouts.get(name)
        if gi is None or mi is None:
            continue
        gt_boxes = [it.box for it in gi["items"]]
        m_boxes = [it.box for it in mi["items"]]
        extent = union_box(m_boxes)
        centers = np.array([bbox_center(b) for b in m_boxes])
        entry = {}
        for ent, a in ents.items():
            idxs, atts, seen = [], [], set()
            for k, gidx in enumerate(a["anchor_idxs"]):
                if gidx >= len(gt_boxes):
                    continue
                gc = np.array(bbox_center(gt_boxes[gidx]))
                mi_idx = int(np.argmin(np.linalg.norm(centers - gc, axis=1)))
                if mi_idx in seen:
                    continue
                seen.add(mi_idx)
                idxs.append(mi_idx)
                atts.append(a["anchor_atts"][k]
                            if k < len(a.get("anchor_atts", [])) else 0.0)
            if not idxs:
                continue
            entry[ent] = {
                "pred": a["pred"], "mean_logprob": a["mean_logprob"],
                "anchor_idxs": idxs, "anchor_atts": atts,
                "anchor_ys": [round(_norm_y(m_boxes[i], extent), 4)
                              for i in idxs],
            }
        out[name] = entry
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dilate-ratio", type=float, default=0.04)
    parser.add_argument("--anchors-only", action="store_true")
    parser.add_argument("--shard", type=str, default="")
    parser.add_argument("--strict", action="store_true",
                        help="用严格版 anchors_morphstrict.json（峰直接对齐 morph 框, "
                             "无 GT 中转）, 仅执行步骤2, 模板目录 _morphstrict04")
    parser.add_argument("--dataset", type=str, default="sroie")
    args = parser.parse_args()

    setup_seeds()
    ds = get_dataset(args.dataset)
    loc_cfg = LocalizationConfig()
    tag = morph_tag(args.dilate_ratio)
    boot_dir = bootstrap_dir(ds.name)

    gt_train = ds.load_layout_items("train")
    gt_test = ds.load_layout_items("test")
    print("形态学提框（train + test）...")
    morph_train = build_morph_layouts(ds, sorted(gt_train.keys()), gt_train,
                                      args.dilate_ratio)
    morph_test = build_morph_layouts(ds, sorted(gt_test.keys()), gt_test,
                                     args.dilate_ratio)

    # ── 步骤1: anchors 几何转移（strict 版无需, anchors 已直接对齐 morph 框）──
    if not args.strict:
        for split, gt_l, m_l in (("train", gt_train, morph_train),
                                 ("test", gt_test, morph_test)):
            src = os.path.join(boot_dir, "anchors.json" if split == "train"
                               else "anchors_test.json")
            dst = src.replace(".json", "_morph.json")
            if os.path.exists(dst):
                print(f"{dst} 已存在, 跳过")
                continue
            transferred = transfer_anchors(json.load(open(src)), gt_l, m_l)
            json.dump(transferred, open(dst, "w"), ensure_ascii=False, indent=1)
            n = sum(len(v) for v in transferred.values())
            print(f"{split}: {len(transferred)} 样本 / {n} 条 -> {dst}")
    if args.anchors_only:
        return

    # ── 步骤2: morph 自举模板重建 ──
    tpl_tag = tag.replace("morph", "morphstrict") if args.strict else tag
    out_dir = os.path.join(
        CACHE_DIR,
        f"{ds.name}_bootstrap_template_{loc_cfg.local_tag}_{tpl_tag}")
    os.makedirs(out_dir, exist_ok=True)
    anchors_name = ("anchors_morphstrict.json" if args.strict
                    else "anchors_morph.json")
    anchors_m = json.load(open(os.path.join(boot_dir, anchors_name)))

    jobs = []
    for name, ents in anchors_m.items():
        info = morph_train.get(name)
        if info is None:
            continue
        for ent, a in ents.items():
            path = os.path.join(out_dir, f"{name}__{ent}.npy")
            if os.path.exists(path):
                continue
            if a["mean_logprob"] < LP_THR:
                save_npy(path, None)
                continue
            jobs.append((name, ent, a["anchor_idxs"], path))
    if args.shard:
        si, sn = map(int, args.shard.split("/"))
        jobs = jobs[si::sn]
    print(f"模板渲染任务 {len(jobs)} 条")
    if not jobs:
        return

    model, processor = load_model_and_processor(
        ModelConfig(), for_localization=True)
    embedder = PostMergerEmbedder(model, processor, loc_cfg.global_prompt)
    for ji, (name, ent, idxs, path) in enumerate(jobs):
        info = morph_train[name]
        boxes = [it.box for it in info["items"]]
        size = Image.open(info["image_path"]).size
        imgs = []
        for ai in idxs:
            lb, region = get_local_bboxes(
                boxes, ai, loc_cfg.n_local_bboxes, size,
                wrap_gap_ratio=loc_cfg.wrap_gap_ratio,
                use_wrap=loc_cfg.use_wrap)
            imgs.append(generate_local_layout(lb, region))
        embs = embedder.extract_batch(imgs)
        save_npy(path, np.stack(embs))
        if (ji + 1) % 50 == 0:
            print(f"template {ji + 1}/{len(jobs)}", flush=True)
    embedder.remove_hook()
    print(f"完成 -> {out_dir}")


if __name__ == "__main__":
    main()
