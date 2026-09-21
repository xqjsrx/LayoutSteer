"""零标注自举 Step2: 注意力 anchor -> 自举模板库（局部布局 embedding）。

读 Step1 的 anchors.json，按 logprob 门控筛选，对每个通过的 (样本, 实体)
用 top3 注意力 anchor 框各渲染一张局部布局图 -> embedding，堆叠为 [K, D]
存入 bootstrap 模板缓存目录（EmbStore 兼容命名）。未过门控存空标记。

消歧策略: 不消歧——3 个 anchor 候选全部入库，检索时的多模板投票聚类
天然过滤噪声峰（真峰跨样本空间收敛成多数票，噪声峰分散聚不成簇）。

用法:
  CUDA_VISIBLE_DEVICES=3 python scripts/bootstrap_templates.py --dataset sroie
  python scripts/bootstrap_templates.py --dataset cord --shard 0/2   # 分片可续跑
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from layoutsteer.config import (LocalizationConfig, CACHE_DIR,
                                bootstrap_dir, make_model_config)
from layoutsteer.datasets import get_dataset
from layoutsteer.model_loader import setup_seeds
from layoutsteer.adapters import get_adapter
from layoutsteer.localization import PostMergerEmbedder
from layoutsteer.localization.store import save_npy
from layoutsteer.localization.layout_render import (
    get_local_bboxes, generate_local_layout)
from PIL import Image

BATCH = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sroie")
    parser.add_argument("--model", type=str, default="qwen25vl")
    parser.add_argument("--lp-thr", type=float, default=-0.05,
                        help="入库门控: normal mean_logprob 阈值")
    parser.add_argument("--shard", type=str, default="",
                        help='按样本切分，格式 "i/n"')
    args = parser.parse_args()

    setup_seeds()
    loc_cfg = LocalizationConfig()
    ds = get_dataset(args.dataset)
    boot_dir = bootstrap_dir(ds.name, args.model)
    anchors_path = os.path.join(boot_dir, "anchors.json")
    meta_path = os.path.join(boot_dir, "bootstrap_meta.json")
    _msfx = "" if args.model == "qwen25vl" else f"_{args.model}"
    out_dir = os.path.join(
        CACHE_DIR, f"{ds.name}_bootstrap_template_{loc_cfg.local_tag}{_msfx}")
    anchors = json.load(open(anchors_path))
    layouts = ds.load_layout_items("train")

    names = sorted(anchors.keys())
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        names = names[i::n]
        print(f"分片 {i}/{n}: 负责 {len(names)} 样本")

    # 门控 + 渲染任务收集
    jobs, meta, n_gated, n_cached = [], {}, 0, 0
    for name in names:
        ents = anchors[name]
        info = layouts.get(name)
        if info is None:
            continue
        boxes = [it.box for it in info["items"]]
        img_size = Image.open(info["image_path"]).size \
            if os.path.exists(info["image_path"]) else None
        for entity, a in ents.items():
            path = os.path.join(out_dir, f"{name}__{entity}.npy")
            if a["mean_logprob"] < args.lp_thr or img_size is None:
                save_npy(path, None)   # 未过门控: 空标记
                n_gated += 1
                continue
            meta[f"{name}__{entity}"] = a["anchor_ys"][:len(a["anchor_idxs"])]
            if os.path.exists(path) and os.path.getsize(path) > 128:
                n_cached += 1          # 已算过（空标记文件极小，不会误判）
                continue
            imgs = []
            for ai in a["anchor_idxs"]:
                local_bboxes, region = get_local_bboxes(
                    boxes, ai, loc_cfg.n_local_bboxes, img_size,
                    wrap_gap_ratio=loc_cfg.wrap_gap_ratio,
                    use_wrap=loc_cfg.use_wrap)
                imgs.append(generate_local_layout(local_bboxes, region))
            jobs.append((path, imgs))

    print(f"待算 {len(jobs)} 条 (门控拒绝 {n_gated}, 缓存命中 {n_cached}), "
          f"共 {sum(len(im) for _, im in jobs)} 张局部布局图")

    # 补齐空标记: 定位流水线会查询任意 (训练文档, 键) 组合，缓存须对全部组合
    # 可答（与 embed.py 的 GT 模板缓存一致），未采集到锚点的组合记为空。
    # 范围是全部训练文档而非仅有锚点的样本——锚点抽样时未采集的文档也会被查询。
    os.makedirs(out_dir, exist_ok=True)
    fill_names = names if args.shard else sorted(layouts.keys())
    n_marker = 0
    for name in fill_names:
        for entity in ds.entity_types:
            path = os.path.join(out_dir, f"{name}__{entity}.npy")
            if not os.path.exists(path):
                save_npy(path, None)
                n_marker += 1
    if n_marker:
        print(f"补齐 {n_marker} 个空标记（该文档未采集到该键的锚点）")

    if jobs:
        adapter = get_adapter(make_model_config(args.model))
        adapter.load_for_localization()
        embedder = PostMergerEmbedder(adapter.model, adapter.processor,
                                      loc_cfg.global_prompt)

        # 扁平批量提取后按条目重组
        flat = [(ji, im) for ji, (_, imgs) in enumerate(jobs) for im in imgs]
        embs = []
        for c0 in range(0, len(flat), BATCH):
            embs.extend(
                embedder.extract_batch([im for _, im in flat[c0:c0 + BATCH]]))
            if (c0 // BATCH) % 20 == 0:
                print(f"  {c0}/{len(flat)}", flush=True)
                torch.cuda.empty_cache()
        by_job = {}
        for (ji, _), e in zip(flat, embs):
            by_job.setdefault(ji, []).append(e)
        for ji, (path, _) in enumerate(jobs):
            save_npy(path, np.stack(by_job[ji]))
        embedder.remove_hook()

    # 合并已有 meta（分片/续跑不互相覆盖）
    if os.path.exists(meta_path):
        try:
            old = json.load(open(meta_path))
            old.update(meta)
            meta = old
        except json.JSONDecodeError:
            pass
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"自举模板库 -> {out_dir}")
    print(f"anchor_y 元数据 ({len(meta)} 条) -> {meta_path}")


if __name__ == "__main__":
    main()
