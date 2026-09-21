"""target_layers 批量扫描：多组层配置在一次生成内并行（batch 维度），normal 复用。

层配置 spec 语法: "all" / "22" / "20,21,22" / "10-18"（闭区间），多组用 "|" 分隔。

用法:
  CUDA_VISIBLE_DEVICES=2 python scripts/sweep_layers.py --dataset sroie \
      --bbox-source retrieval --delta 1 --conf-weight \
      --layer-specs "0-9|10-18|19-27|14-27|22" \
      --normal-from <normal_results.json> --shard 0/4
汇总: 同参数加 --report
"""
import os
import sys
import json
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from layoutsteer.config import RunConfig, OUTPUT_DIR, shard_tasks
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
    return processor(text=[text] * batch, images=[image] * batch,
                     padding=True, return_tensors="pt").to(device)


def parse_spec(spec: str, n_layers: int) -> set:
    if spec == "all":
        return set(range(n_layers))
    if "-" in spec:
        lo, hi = spec.split("-")
        return set(range(int(lo), int(hi) + 1))
    return {int(x) for x in spec.split(",") if x.strip()}


def spec_tag(spec: str) -> str:
    return "L" + spec.replace(",", ".")


def report(out_dir, specs, normal_path):
    normal = {(r["sample_name"], r["question"]): r
              for r in json.load(open(normal_path))}
    base_acc = sum(r["is_correct"] for r in normal.values()) / len(normal)
    n_index = {}
    for r in normal.values():
        n_index[(r["sample_name"], r["task"])] = r["is_correct"]
    print(f"基线 (normal): {base_acc:.2%}  (n={len(normal)})")
    print(f"{'layers':>10} {'n':>5} {'acc':>8} {'Δ':>7} | per-entity Δ")
    for spec in specs:
        merged = {}
        for p in glob.glob(os.path.join(out_dir, f"{spec_tag(spec)}*.json")):
            for r in json.load(open(p)):
                merged[(r["sample_name"], r["task"])] = r
        if not merged:
            print(f"{spec:>10} {'—':>5}")
            continue
        accs, ent = [], {}
        for k, r in merged.items():
            if k not in n_index:
                continue
            accs.append(r["is_correct"])
            e = ent.setdefault(k[1], [0, 0, 0])
            e[0] += r["is_correct"]; e[1] += n_index[k]; e[2] += 1
        acc = sum(accs) / len(accs)
        ent_str = " ".join(f"{t[:3]}{(v[0]-v[1])/max(v[2],1):+.1%}"
                           for t, v in sorted(ent.items()))
        print(f"{spec:>10} {len(accs):>5} {acc:>8.2%} {acc-base_acc:>+7.2%} | {ent_str}")


def main():
    parser = argparse.ArgumentParser(description="target_layers 批量扫描")
    parser.add_argument("--dataset", type=str, default="sroie")
    parser.add_argument("--bbox-source", type=str, default="retrieval")
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--layer-specs", type=str,
                        default="0-9|10-18|19-27|14-27|22")
    parser.add_argument("--conf-weight", action="store_true")
    parser.add_argument("--conf-gamma", type=float, default=1.0)
    parser.add_argument("--normal-from", type=str, required=True)
    parser.add_argument("--shard", type=str, default="")
    parser.add_argument("--n-tasks", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    specs = args.layer_specs.split("|")
    ds = get_dataset(args.dataset)
    conf_tag = "_conf" if args.conf_weight else ""
    out_dir = os.path.join(OUTPUT_DIR, ds.name, "runs",
                           f"layersweep_{args.bbox_source}{conf_tag}_d{args.delta:g}")
    os.makedirs(out_dir, exist_ok=True)

    if args.report:
        report(out_dir, specs, args.normal_from)
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

    out_paths = {s: os.path.join(out_dir, f"{spec_tag(s)}{tag}.json") for s in specs}
    results = {s: [] for s in specs}
    done = set()
    if os.path.exists(out_paths[specs[0]]):
        try:
            for s in specs:
                results[s] = json.load(open(out_paths[s]))
            done = {(r["sample_name"], r["task"]) for r in results[specs[0]]}
            print(f"断点续跑: 已完成 {len(done)} 任务")
        except (json.JSONDecodeError, FileNotFoundError):
            results = {s: [] for s in specs}

    model, processor = load_model_and_processor(cfg.model)
    probe = ScoreDeltaProbe(model, processor)
    layer_sets = [parse_spec(s, probe.n_layers) for s in specs]

    def save():
        for s in specs:
            tmp = out_paths[s] + ".tmp"
            json.dump(results[s], open(tmp, "w"), ensure_ascii=False, indent=2)
            os.replace(tmp, out_paths[s])

    B = len(specs)
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
        delta_eff = args.delta * conf ** args.conf_gamma

        with probe.attach(delta=[delta_eff] * B, target_layers=layer_sets):
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        preds = processor.batch_decode(gen[:, input_len:], skip_special_tokens=True)

        for s, pred in zip(specs, preds):
            pred = pred.strip()
            results[s].append({
                "sample_name": task.sample_name, "task": task.task_type,
                "question": task.question, "prediction": pred,
                "ground_truth": task.answer,
                "is_correct": is_correct_prediction(pred, task.answer),
                "confidence": round(conf, 4), "delta_eff": round(delta_eff, 4),
            })
        del inputs, gen
        if (idx + 1) % 20 == 0:
            save()
            torch.cuda.empty_cache()
            print(f"[{tag or 'all'}] {len(results[specs[0]])}/{len(tasks)}", flush=True)
    save()
    print(f"完成 {len(results[specs[0]])} 任务 x {B} 层配置 -> {out_dir}")


if __name__ == "__main__":
    main()
