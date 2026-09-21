"""δ 强度批量扫描：同一任务一次生成内并行多个 δ（batch 维度），normal 基线复用。

加速原理:
  1. normal 与干预参数无关，从 --normal-from 复用（省一半生成）
  2. 不同 δ 只改 probe 加到 K 上的增量系数，输入完全相同 ->
     batch=len(deltas) 一次生成同时产出全部 δ 档位（GPU 原本利用率低，batch 几乎免费）
  3. 置信度加权（--conf-weight）在 δ_eff 列表上逐任务生效

用法:
  CUDA_VISIBLE_DEVICES=2 python scripts/sweep_delta.py --dataset sroie \
      --bbox-source retrieval --deltas 0.5,1,2,5 --conf-weight \
      --normal-from output/sroie/runs/retrieval_delta1_confg1_20260724_030211/normal_results.json \
      --shard 0/4
输出: output/{ds}/runs/deltasweep_{source}[_conf]/delta{d}[.shard].json
跑完全部分片后: python scripts/sweep_delta.py ... --report
"""
import os
import sys
import json
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from layoutsteer.config import RunConfig, MemoryConfig, OUTPUT_DIR, shard_tasks
from layoutsteer.datasets import get_dataset
from layoutsteer.evaluation.metrics import is_correct_prediction
from layoutsteer.model_loader import (
    setup_seeds, load_model_and_processor, resize_image_by_pixel_limit)
from layoutsteer.intervention import (
    ScoreDeltaProbe, bboxes_px_to_rel, bboxes_to_grid_indices)


def build_batch_inputs(processor, image, prompt_text, device, batch):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt_text},
        {"type": "image", "image": image},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text] * batch, images=[image] * batch,
                       padding=True, return_tensors="pt").to(device)
    return inputs


def report(out_dir, deltas, normal_path, ds):
    """汇总各 δ 档位结果（合并分片、对比基线）。"""
    normal = {(r["sample_name"], r["question"]): r
              for r in json.load(open(normal_path))}
    base_acc = sum(r["is_correct"] for r in normal.values()) / len(normal)
    print(f"基线 (normal): {base_acc:.2%}  (n={len(normal)})")
    print(f"{'delta':>6} {'n':>5} {'acc':>8} {'Δ':>7} | per-entity Δ")
    for d in deltas:
        merged = {}
        for p in glob.glob(os.path.join(out_dir, f"delta{d:g}*.json")):
            for r in json.load(open(p)):
                merged[(r["sample_name"], r["task"])] = r
        if not merged:
            print(f"{d:>6g} {'—':>5}")
            continue
        accs, ent_diff = [], {}
        for (sname, ttype), r in merged.items():
            nr = next((v for k, v in normal.items() if k[0] == sname
                       and v["task"] == ttype), None)
            if nr is None:
                continue
            accs.append(r["is_correct"])
            e = ent_diff.setdefault(ttype, [0, 0, 0])
            e[0] += r["is_correct"]; e[1] += nr["is_correct"]; e[2] += 1
        acc = sum(accs) / len(accs)
        ent_str = " ".join(
            f"{t[:3]}{(v[0] - v[1]) / max(v[2], 1):+.1%}"
            for t, v in sorted(ent_diff.items()))
        print(f"{d:>6g} {len(accs):>5} {acc:>8.2%} {acc - base_acc:>+7.2%} | {ent_str}")


def main():
    parser = argparse.ArgumentParser(description="δ 批量扫描（batch 内并行）")
    parser.add_argument("--dataset", type=str, default="sroie")
    parser.add_argument("--bbox-source", type=str, default="retrieval",
                        choices=["gt", "retrieval"])
    parser.add_argument("--deltas", type=str, default="0.5,1,2,5")
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--conf-weight", action="store_true")
    parser.add_argument("--conf-gamma", type=float, default=1.0)
    parser.add_argument("--normal-from", type=str, required=True,
                        help="normal_results.json 路径（基线复用 + 汇总对比）")
    parser.add_argument("--shard", type=str, default="")
    parser.add_argument("--n-tasks", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--report", action="store_true", help="仅汇总已有结果")
    args = parser.parse_args()

    deltas = [float(x) for x in args.deltas.split(",")]
    ds = get_dataset(args.dataset)
    conf_tag = "_conf" if args.conf_weight else ""
    layer_tag = "" if args.layers == "all" else f"_L{args.layers}"
    out_dir = os.path.join(OUTPUT_DIR, ds.name, "runs",
                           f"deltasweep_{args.bbox_source}{conf_tag}{layer_tag}")
    os.makedirs(out_dir, exist_ok=True)

    if args.report:
        report(out_dir, deltas, args.normal_from, ds)
        return

    cfg = RunConfig(dataset=args.dataset, bbox_source=args.bbox_source)
    setup_seeds()
    bbox_map = ds.load_bboxes(args.bbox_source)
    conf_map = ds.load_confidences(args.bbox_source) if args.conf_weight else {}

    tasks = ds.iter_tasks()
    if args.n_tasks > 0:
        tasks = tasks[:args.n_tasks]
    tag = ""
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        tasks = shard_tasks(tasks, i, n)
        tag = f".shard{i}of{n}"

    # 断点续跑（按第一个 δ 档位的输出判断）
    out_paths = {d: os.path.join(out_dir, f"delta{d:g}{tag}.json") for d in deltas}
    results = {d: [] for d in deltas}
    done = set()
    p0 = out_paths[deltas[0]]
    if os.path.exists(p0):
        try:
            for d in deltas:
                results[d] = json.load(open(out_paths[d]))
            done = {(r["sample_name"], r["task"]) for r in results[deltas[0]]}
            print(f"断点续跑: 已完成 {len(done)} 任务")
        except (json.JSONDecodeError, FileNotFoundError):
            results = {d: [] for d in deltas}

    model, processor = load_model_and_processor(cfg.model)
    probe = ScoreDeltaProbe(model, processor)
    from layoutsteer.config import parse_target_layers
    target_layers = parse_target_layers(args.layers, probe.n_layers)

    def save():
        for d in deltas:
            tmp = out_paths[d] + ".tmp"
            json.dump(results[d], open(tmp, "w"), ensure_ascii=False, indent=2)
            os.replace(tmp, out_paths[d])

    B = len(deltas)
    for idx, task in enumerate(tasks):
        if (task.sample_name, task.task_type) in done:
            continue
        boxes_px = bbox_map.get(task.key)
        if not boxes_px or not os.path.exists(task.image_path):
            continue
        raw = Image.open(task.image_path).convert("RGB")
        orig_size = raw.size
        image = resize_image_by_pixel_limit(raw, cfg.model.max_image_pixels)
        inputs = build_batch_inputs(
            processor, image, ds.build_prompt(task), model.device, B)
        probe.setup_image_range(inputs["input_ids"], inputs["image_grid_thw"])
        grid_idx = bboxes_to_grid_indices(
            bboxes_px_to_rel(boxes_px, orig_size), probe.spatial_shape)
        probe.set_region(grid_idx)

        conf = float(conf_map.get(task.key, 1.0))
        delta_effs = [d * conf ** args.conf_gamma for d in deltas]

        with probe.attach(delta=delta_effs, target_layers=target_layers):
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        preds = processor.batch_decode(
            gen[:, input_len:], skip_special_tokens=True)

        for d, de, pred in zip(deltas, delta_effs, preds):
            pred = pred.strip()
            results[d].append({
                "sample_name": task.sample_name, "task": task.task_type,
                "question": task.question, "prediction": pred,
                "ground_truth": task.answer,
                "is_correct": is_correct_prediction(pred, task.answer),
                "confidence": round(conf, 4), "delta_eff": round(de, 4),
            })
        del inputs, gen
        if (idx + 1) % 20 == 0:
            save()
            torch.cuda.empty_cache()
            print(f"[{tag or 'all'}] {len(results[deltas[0]])}/{len(tasks)}",
                  flush=True)
    save()
    print(f"完成 {len(results[deltas[0]])} 任务 x {B} 档位 -> {out_dir}")


if __name__ == "__main__":
    main()
