"""零标注自举 Step1: normal 推理 + 注意力 top3 峰 -> 自动 anchor 标注。

对每个样本 × 该样本实际被提问的键（ds.sample_keys）:
  normal 推理（delta=0 capture L22）+ 逐 token logprob
  生成期平均注意力图 -> top3 峰 -> 各自对齐最近 OCR 几何框
输出 anchor 候选与门控信号，供 Step2 构建自举模板库。
门控在 Step2 施加（logprob 阈值），本脚本全量记录。

用法:
  CUDA_VISIBLE_DEVICES=3 python scripts/bootstrap_anchors.py --dataset sroie --shard 0/3
  python scripts/bootstrap_anchors.py --dataset sroie --merge
  python scripts/bootstrap_anchors.py --dataset cord --split test
"""
import os
import sys
import json
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from layoutsteer.config import RunConfig, bootstrap_dir, make_model_config
from layoutsteer.datasets import get_dataset
from layoutsteer.model_loader import setup_seeds
from layoutsteer.adapters import get_adapter

CAPTURE_LAYER = 22  # Qwen2.5-VL(28层) 扫得最优 grounding 层; Qwen3-VL 用 --capture-layer 指定


def generate_with_scores(model, processor, inputs, max_new_tokens=64):
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            output_scores=True, return_dict_in_generate=True)
    input_len = inputs["input_ids"].shape[1]
    seq = out.sequences[:, input_len:]
    pred = processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
    lps = []
    for step, logits in enumerate(out.scores):
        lp = torch.log_softmax(logits[0].float(), dim=-1)[seq[0, step]]
        lps.append(float(lp))
    return pred, (sum(lps) / len(lps) if lps else -99.0)


def top3_anchor_boxes(att, grid_hw, boxes_px, img_size):
    """注意力图 top3 峰 -> [(框idx, 峰注意力值), ...]（去重框）。"""
    h, w = grid_hw
    W, H = img_size
    centers = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
                        for b in boxes_px])
    flat = att.flatten()
    out, seen = [], set()
    for idx in np.argsort(flat)[::-1][:3]:
        r, c = np.unravel_index(idx, att.shape)
        px, py = (c + 0.5) / w * W, (r + 0.5) / h * H
        bi = int(np.argmin(np.hypot(centers[:, 0] - px, centers[:, 1] - py)))
        if bi not in seen:
            seen.add(bi)
            out.append((bi, float(flat[idx])))
    return out


def merge(out_dir, split, morph=False):
    suffix = "_morphstrict" if morph else ""
    prefix = (f"anchors_{split}" if split != "train" else "anchors") + suffix
    merged = {}
    for p in sorted(glob.glob(os.path.join(out_dir, f"{prefix}.shard*.json"))):
        merged.update(json.load(open(p)))
    out = os.path.join(out_dir, f"{prefix}.json")
    json.dump(merged, open(out, "w"), ensure_ascii=False, indent=1)
    n = sum(len(v) for v in merged.values())
    print(f"合并: {len(merged)} 样本 / {n} (样本,实体) 条 -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sroie")
    parser.add_argument("--model", type=str, default="qwen25vl",
                        help="锚点提取模型（Tier A 连续网格模型）")
    parser.add_argument("--capture-layer", type=int, default=CAPTURE_LAYER,
                        help="注意力捕获层（须为该模型 grounding 最优层）")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"])
    parser.add_argument("--shard", type=str, default="")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--morph", action="store_true",
                        help="OCR-free: 注意力峰直接对齐形态学框（无 GT 框中转）")
    args = parser.parse_args()

    cfg = RunConfig(model=make_model_config(args.model))
    setup_seeds()
    ds = get_dataset(args.dataset)
    out_dir = bootstrap_dir(ds.name, args.model)
    os.makedirs(out_dir, exist_ok=True)
    if args.merge:
        merge(out_dir, args.split, morph=args.morph)
        return

    # 每键的标准 prompt（与测试一致）；已验证四数据集每键仅一种问句
    entity_prompts = {}
    for task in ds.iter_tasks():
        entity_prompts.setdefault(task.task_type, ds.build_prompt(task))

    if args.morph:
        from run_ocrfree_localize import build_morph_layouts
        gt_layouts = ds.load_layout_items(args.split)
        layouts = build_morph_layouts(ds, sorted(gt_layouts.keys()),
                                      gt_layouts, 0.04)
    else:
        layouts = ds.load_layout_items(args.split)
    names = sorted(layouts.keys())
    tag = ""
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        names = names[i::n]
        tag = f".shard{i}of{n}"
    suffix = "_morphstrict" if args.morph else ""
    prefix = (f"anchors_{args.split}" if args.split != "train"
              else "anchors") + suffix
    out_path = os.path.join(out_dir, f"{prefix}{tag}.json")
    result = {}
    if os.path.exists(out_path):
        try:
            result = json.load(open(out_path))
            print(f"断点续跑: 已有 {len(result)} 样本")
        except json.JSONDecodeError:
            result = {}

    adapter = get_adapter(cfg.model)
    adapter.load()
    model, processor = adapter.model, adapter.processor
    probe = adapter.create_probe(capture_layer=args.capture_layer)

    n_gen, n_noprompt = 0, 0
    for idx, name in enumerate(names):
        if name in result:
            continue
        info = layouts[name]
        if not os.path.exists(info["image_path"]):
            continue
        boxes = [it.box for it in info["items"]]
        ys = [b[1] for b in boxes] + [b[3] for b in boxes]
        y_lo, y_hi = min(ys), max(ys)
        raw = Image.open(info["image_path"]).convert("RGB")
        orig_size = raw.size
        image = adapter.prepare_image(raw)
        entry = {}
        for entity in ds.sample_keys(name, args.split):
            prompt = entity_prompts.get(entity)
            if prompt is None:
                n_noprompt += 1  # 训练侧出现但测试侧没问过的键，无标准问句
                continue
            inputs = adapter.build_inputs(image, prompt)
            probe.setup_image_range(inputs["input_ids"],
                                    inputs["image_grid_thw"])
            probe.set_region(None)
            with probe.attach(delta=0.0, capture=True):
                pred, mean_lp = generate_with_scores(model, processor, inputs)
            n_gen += 1
            maps = probe.get_attention_maps()
            if not maps:
                continue
            att = np.mean(np.stack(maps), axis=0)
            anchors = top3_anchor_boxes(att, probe.spatial_shape, boxes,
                                        orig_size)
            entry[entity] = {
                "pred": pred, "mean_logprob": round(mean_lp, 5),
                "anchor_idxs": [a for a, _ in anchors],
                "anchor_atts": [round(v, 6) for _, v in anchors],
                "anchor_ys": [
                    round(((boxes[a][1] + boxes[a][3]) / 2 - y_lo)
                          / max(y_hi - y_lo, 1), 4) for a, _ in anchors],
            }
            del inputs
        result[name] = entry
        if (idx + 1) % 20 == 0:
            json.dump(result, open(out_path + ".tmp", "w"),
                      ensure_ascii=False, indent=1)
            os.replace(out_path + ".tmp", out_path)
            torch.cuda.empty_cache()
            print(f"[{tag or 'all'}] {len(result)}/{len(names)}", flush=True)
    json.dump(result, open(out_path + ".tmp", "w"), ensure_ascii=False, indent=1)
    os.replace(out_path + ".tmp", out_path)
    print(f"完成 {len(result)} 样本 / {n_gen} 次生成 -> {out_path}")
    if n_noprompt:
        print(f"跳过 {n_noprompt} 个 (样本,键)：该键在测试侧无标准问句")


if __name__ == "__main__":
    main()
