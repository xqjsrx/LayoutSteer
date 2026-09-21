"""布局掩码干预全量运行: 逐 token 强度 δ_i = δ·conf·w_i。

与 run_oneshot（bbox 均匀 δ, 93.30%）唯一差异: set_region 附带掩码权重。
  --mask-mode feather: A版, 区域框羽化软边（≈bbox 表述升级, 预期打平）
  --mask-mode votes:   B版, 投票置信热力图（空间逐点置信加权）

用法:
  CUDA_VISIBLE_DEVICES=4 python scripts/run_mask.py --mask-mode votes --shard 0/2
汇总: python scripts/run_mask.py --mask-mode votes --report --normal-from <normal.json>
"""
import os
import sys
import json
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from layoutsteer.config import (RunConfig, OUTPUT_DIR, parse_target_layers,
                                shard_tasks)
from layoutsteer.datasets import get_dataset
from layoutsteer.evaluation.metrics import is_correct_prediction
from layoutsteer.model_loader import (
    setup_seeds, load_model_and_processor, resize_image_by_pixel_limit,
    build_inputs, generate_and_decode)
from layoutsteer.intervention import (
    ScoreDeltaProbe, weight_map_to_grid, grid_to_indices_weights,
    layout_mask_grid, structured_mask_grid)


def report(out_dir, normal_path, ref_glob, mode):
    normal = {(r["sample_name"], r["task"]): r["is_correct"]
              for r in json.load(open(normal_path))}
    base_acc = sum(normal.values()) / len(normal)

    def load_glob(pat):
        merged = {}
        for p in glob.glob(pat):
            for r in json.load(open(p)):
                merged[(r["sample_name"], r["task"])] = r
        return merged

    mask_res = load_glob(os.path.join(out_dir, "mask*.json"))
    ref = load_glob(ref_glob)
    print(f"基线 (normal): {base_acc:.2%}  (n={len(normal)})")
    for name, merged in (("oneshot bbox均匀(对照)", ref),
                         (f"mask[{mode}]", mask_res)):
        if not merged:
            print(f"{name}: 无数据")
            continue
        accs = [r["is_correct"] for k, r in merged.items() if k in normal]
        acc = sum(accs) / len(accs)
        print(f"{name}: acc={acc:.2%} ({acc - base_acc:+.2%}, n={len(accs)})")
    common = [k for k in mask_res if k in ref]
    if common:
        same = sum(1 for k in common
                   if mask_res[k]["prediction"].strip()
                   == ref[k]["prediction"].strip())
        print(f"mask vs bbox均匀 预测一致率: "
              f"{same}/{len(common)} ({same/len(common):.1%})")


def main():
    parser = argparse.ArgumentParser(description="布局掩码干预全量运行")
    parser.add_argument("--dataset", type=str, default="sroie")
    parser.add_argument("--bbox-source", type=str, default="retrieval")
    parser.add_argument("--mask-mode", type=str, default="votes",
                        choices=["feather", "votes", "layout", "structured"])
    parser.add_argument("--feather-sigma", type=float, default=1.0)
    parser.add_argument("--bg-weight", type=float, default=0.1,
                        help="layout/structured 模式底纹权重")
    parser.add_argument("--core-gap", type=float, default=0.7,
                        help="structured 模式核心矩形间隙权重")
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--layers", type=str, default="19-27")
    parser.add_argument("--conf-gamma", type=float, default=1.0)
    parser.add_argument("--normal-from", type=str, default="")
    parser.add_argument("--ref-glob", type=str,
                        default="output/sroie/runs/oneshot_retrieval_conf_d2/oneshot*.json")
    parser.add_argument("--shard", type=str, default="")
    parser.add_argument("--n-tasks", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--bbox-json", type=str, default="",
                        help="覆盖定位结果路径（如 OCR-free morphsym 产物）")
    parser.add_argument("--layout-source", type=str, default="gt",
                        choices=["gt", "morph"],
                        help="layout/structured 底纹布局框来源; morph=零标注形态学框")
    parser.add_argument("--out-tag", type=str, default="",
                        help="输出目录附加标记（区分不同 bbox-json 实验）")
    args = parser.parse_args()

    ds = get_dataset(args.dataset)
    if args.bbox_json:
        ds.set_retrieval_bbox_path(args.bbox_json)
    if args.mask_mode == "layout":
        mode_tag = f"mask_layout_bg{args.bg_weight:g}"
    elif args.mask_mode == "structured":
        mode_tag = f"mask_structured_gap{args.core_gap:g}"
    else:
        mode_tag = f"mask_{args.mask_mode}_sg{args.feather_sigma:g}"
    if args.out_tag:
        mode_tag = f"{mode_tag}_{args.out_tag}"
    out_dir = os.path.join(
        OUTPUT_DIR, ds.name, "runs",
        f"{mode_tag}_{args.bbox_source}_d{args.delta:g}")
    os.makedirs(out_dir, exist_ok=True)
    if args.report:
        report(out_dir, args.normal_from, args.ref_glob, args.mask_mode)
        return

    cfg = RunConfig(dataset=args.dataset, bbox_source=args.bbox_source)
    setup_seeds()
    wmap_map = ds.load_weight_maps(args.bbox_source)
    conf_map = ds.load_confidences(args.bbox_source)
    bbox_map = ds.load_bboxes(args.bbox_source)
    # layout/structured 模式: 全文档布局框作低权底纹
    layout_boxes = {}
    if args.mask_mode in ("layout", "structured"):
        if args.layout_source == "morph":
            import cv2
            import numpy as np
            from visualize_ocrfree import extract_text_boxes
            for name, info in ds.load_layout_items("test").items():
                img = Image.open(info["image_path"]).convert("RGB")
                bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                layout_boxes[name] = extract_text_boxes(bgr)
                img.close()
        else:
            layout_boxes = {
                name: [it.box for it in info["items"]]
                for name, info in ds.load_layout_items("test").items()}

    tasks = ds.iter_tasks()
    if args.n_tasks > 0:
        tasks = tasks[:args.n_tasks]
    tag = ""
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        tasks = shard_tasks(tasks, i, n)
        tag = f".shard{i}of{n}"

    out_path = os.path.join(out_dir, f"mask{tag}.json")
    results, done = [], set()
    if os.path.exists(out_path):
        try:
            results = json.load(open(out_path))
            done = {(r["sample_name"], r["task"]) for r in results}
            print(f"断点续跑: 已完成 {len(done)} 任务")
        except json.JSONDecodeError:
            results = []

    model, processor = load_model_and_processor(cfg.model)
    probe = ScoreDeltaProbe(model, processor)
    target_layers = parse_target_layers(args.layers, probe.n_layers)

    def save():
        tmp = out_path + ".tmp"
        json.dump(results, open(tmp, "w"), ensure_ascii=False, indent=2)
        os.replace(tmp, out_path)

    for idx, task in enumerate(tasks):
        if (task.sample_name, task.task_type) in done:
            continue
        wmap = wmap_map.get(task.key)
        has_src = (bbox_map.get(task.key) if args.mask_mode == "layout"
                   else wmap)
        if not has_src or not os.path.exists(task.image_path):
            continue
        if args.mask_mode == "structured" and not bbox_map.get(task.key):
            continue
        raw = Image.open(task.image_path).convert("RGB")
        orig_size = raw.size
        image = resize_image_by_pixel_limit(raw, cfg.model.max_image_pixels)
        inputs = build_inputs(processor, image, ds.build_prompt(task),
                              model.device)
        probe.setup_image_range(inputs["input_ids"], inputs["image_grid_thw"])
        # 空间置信场 -> 当次 token 网格掩码 -> (索引, 逐 token 权重)
        if args.mask_mode == "layout":
            grid = layout_mask_grid(
                bbox_map.get(task.key, []),
                layout_boxes.get(task.sample_name, []),
                orig_size, probe.spatial_shape, bg_weight=args.bg_weight)
        elif args.mask_mode == "structured":
            grid = structured_mask_grid(
                bbox_map.get(task.key, []), wmap,
                layout_boxes.get(task.sample_name, []),
                orig_size, probe.spatial_shape,
                core_gap_weight=args.core_gap)
        else:
            grid = weight_map_to_grid(wmap, orig_size, probe.spatial_shape,
                                      mode=args.mask_mode,
                                      feather_sigma=args.feather_sigma)
        grid_idx, weights = grid_to_indices_weights(grid)
        probe.set_region(grid_idx, weights)

        conf = float(conf_map.get(task.key, 1.0))
        delta_eff = args.delta * conf ** args.conf_gamma

        with probe.attach(delta=delta_eff, target_layers=target_layers,
                          persistent=True):
            pred = generate_and_decode(model, processor, inputs,
                                       args.max_new_tokens)
        results.append({
            "sample_name": task.sample_name, "task": task.task_type,
            "question": task.question, "prediction": pred,
            "ground_truth": task.answer,
            "is_correct": is_correct_prediction(pred, task.answer),
            "confidence": round(conf, 4), "delta_eff": round(delta_eff, 4),
            "n_mask_tokens": int(len(grid_idx)),
        })
        del inputs
        if (idx + 1) % 20 == 0:
            save()
            torch.cuda.empty_cache()
            print(f"[{tag or 'all'}] {len(results)}/{len(tasks)}", flush=True)
    save()
    print(f"完成 {len(results)} 任务 -> {out_path}")


if __name__ == "__main__":
    main()
