"""离线 embedding 提取——定位流水线中唯一需要 GPU 的阶段。

全量预计算四类 embedding 写入 cache/（每条目一个 .npy，原子写、可断点、可分片）:
  1. train_global:   全部训练样本的全局布局 embedding
  2. test_global:    全部测试样本的全局布局 embedding
  3. test_local:     全部测试样本 x 全部候选框的局部布局 embedding
  4. template_local: 全部训练样本 x 全部实体的局部布局 embedding
                     （穷举预计算，与检索结果解耦；无该实体的写空标记）

之后 scripts/localize.py 为纯 numpy 匹配，秒级完成，可任意调参重跑。

用法:
  python scripts/embed.py --dataset sroie                  # 单卡
  # 多卡数据并行（按条目轮转切分，写独立缓存文件，无需合并）:
  CUDA_VISIBLE_DEVICES=2 python scripts/embed.py --dataset sroie --shard 0/6 &
  CUDA_VISIBLE_DEVICES=3 python scripts/embed.py --dataset sroie --shard 1/6 &
  ...
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from layoutsteer.config import LocalizationConfig, make_model_config
from layoutsteer.datasets import get_dataset
from layoutsteer.model_loader import setup_seeds
from layoutsteer.adapters import get_adapter
from layoutsteer.localization import (
    PostMergerEmbedder, sample_layout_image, cache_dirs, save_npy)
from layoutsteer.localization.layout_render import (
    get_local_bboxes, generate_local_layout)
from PIL import Image

# 批处理大小: 单图任务（global/template）16，候选任务（test_local）每子批 8
BATCH_SINGLE = 16
BATCH_CAND = 8


def _image_size(layout_info):
    img = Image.open(layout_info["image_path"])
    size = img.size
    img.close()
    return size


def render_template_layout(layout_info, entity, loc_cfg):
    """渲染模板实体局部布局图；无该实体返回 None。"""
    items = layout_info["items"]
    e_idx = next((i for i, it in enumerate(items) if it.entity == entity), None)
    if e_idx is None:
        return None
    boxes = [it.box for it in items]
    local_bboxes, region = get_local_bboxes(
        boxes, e_idx, loc_cfg.n_local_bboxes, _image_size(layout_info),
        wrap_gap_ratio=loc_cfg.wrap_gap_ratio, use_wrap=loc_cfg.use_wrap)
    return generate_local_layout(local_bboxes, region)


def render_candidate_layouts(layout_info, loc_cfg):
    """渲染测试样本全部候选的局部布局图列表。"""
    boxes = [it.box for it in layout_info["items"]]
    img_size = _image_size(layout_info)
    imgs = []
    for ci in range(len(boxes)):
        local_bboxes, region = get_local_bboxes(
            boxes, ci, loc_cfg.n_local_bboxes, img_size,
            wrap_gap_ratio=loc_cfg.wrap_gap_ratio, use_wrap=loc_cfg.use_wrap)
        imgs.append(generate_local_layout(local_bboxes, region))
    return imgs


def collect_jobs(ds, train_layouts, test_layouts, dirs):
    """枚举全部 embedding 条目，过滤掉已有缓存的。"""
    jobs = []
    for name in sorted(train_layouts.keys()):
        jobs.append(("train_global", name,
                     os.path.join(dirs["train_global"], f"{name}.npy")))
    for name in sorted(test_layouts.keys()):
        jobs.append(("test_global", name,
                     os.path.join(dirs["test_global"], f"{name}.npy")))
        jobs.append(("test_local", name,
                     os.path.join(dirs["test_local"], f"{name}.npy")))
    for name in sorted(train_layouts.keys()):
        for entity in ds.entity_types:
            jobs.append(("template", (name, entity),
                         os.path.join(dirs["template"], f"{name}__{entity}.npy")))
    total = len(jobs)
    jobs = [j for j in jobs if not os.path.exists(j[2])]
    return jobs, total


def main():
    parser = argparse.ArgumentParser(description="LayoutSteer 离线 embedding 提取")
    parser.add_argument("--dataset", type=str, default="sroie")
    parser.add_argument("--model", type=str, default="qwen25vl",
                        help="embedding 提取模型（缓存按模型命名空间隔离）")
    parser.add_argument("--shard", type=str, default="",
                        help='数据并行分片，格式 "i/n"（如 0/6）')
    parser.add_argument("--no-wrap", action="store_true",
                        help="局部邻域不做环面拼接")
    args = parser.parse_args()

    setup_seeds()
    loc_cfg = LocalizationConfig()
    if args.no_wrap:
        loc_cfg.use_wrap = False
    ds = get_dataset(args.dataset)
    dirs = cache_dirs(ds.name, loc_cfg, model=args.model)

    train_layouts = ds.load_layout_items("train")
    test_layouts = ds.load_layout_items("test")

    jobs, total = collect_jobs(ds, train_layouts, test_layouts, dirs)
    print(f"embedding 条目总数 {total}, 待计算 {len(jobs)} (其余缓存命中)")

    if args.shard:
        shard_idx, shard_n = map(int, args.shard.split("/"))
        jobs = jobs[shard_idx::shard_n]
        print(f"分片 {shard_idx}/{shard_n}: 负责 {len(jobs)} 条")
    if not jobs:
        print("全部缓存已就绪，无需计算。")
        return

    adapter = get_adapter(make_model_config(args.model))
    adapter.load_for_localization()
    embedder = PostMergerEmbedder(adapter.model, adapter.processor,
                                  loc_cfg.global_prompt)

    # 按任务类型分组，整批渲染 + 批量 vision-tower 前向
    job_groups = {}
    for j in jobs:
        job_groups.setdefault(j[0], []).append(j)

    t0 = time.time()
    n_done = 0

    def _report(kind, i, total):
        elapsed = time.time() - t0
        eta = elapsed / max(n_done, 1) * (len(jobs) - n_done)
        print(f"  [{n_done}/{len(jobs)}] {kind} chunk {i} "
              f"({elapsed:.0f}s 已用, ETA {eta:.0f}s)", flush=True)

    # ── 单图任务: train_global / test_global / template ──
    for kind in ("train_global", "test_global", "template"):
        group = job_groups.get(kind, [])
        for c0 in range(0, len(group), BATCH_SINGLE):
            chunk = group[c0:c0 + BATCH_SINGLE]
            imgs, paths = [], []
            for _, key, path in chunk:
                if kind == "template":
                    img = render_template_layout(train_layouts[key[0]], key[1], loc_cfg)
                else:
                    layouts = train_layouts if kind == "train_global" else test_layouts
                    img = sample_layout_image(layouts[key])
                imgs.append(img)
                paths.append(path)
            valid = [(im, p) for im, p in zip(imgs, paths) if im is not None]
            if valid:
                embs = embedder.extract_batch([im for im, _ in valid])
                for (_, p), e in zip(valid, embs):
                    save_npy(p, e)
            for im, p in zip(imgs, paths):  # 无实体的模板: 空标记
                if im is None:
                    save_npy(p, None)
            n_done += len(chunk)
            _report(kind, c0 // BATCH_SINGLE + 1, len(group))
        if group:
            torch.cuda.empty_cache()

    # ── 候选任务: test_local（同一样本的多张局部布局分小批）──
    for _, sample_name, path in job_groups.get("test_local", []):
        cand_imgs = render_candidate_layouts(test_layouts[sample_name], loc_cfg)
        embs = []
        for c0 in range(0, len(cand_imgs), BATCH_CAND):
            embs.extend(embedder.extract_batch(cand_imgs[c0:c0 + BATCH_CAND]))
        import numpy as np
        save_npy(path, np.stack(embs))
        n_done += 1
        if n_done % 20 == 0:
            _report("test_local", 0, 0)
            torch.cuda.empty_cache()

    embedder.remove_hook()
    print(f"\n完成，共 {len(jobs)} 条，耗时 {time.time() - t0:.0f}s")
    print(f"缓存目录: {list(dirs.values())}")


if __name__ == "__main__":
    main()
