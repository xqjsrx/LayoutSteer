"""δ 注入落点数值验证: 干预后目标 token 的 pre-softmax score 变化是否精确 = δ。

多模型适配的硬检验——bbox->token 映射或钩子任何一环出错，Δscore 都不会
恰好为 δ。对每个任务:
  1. capture-only（δ=0）捕获生成位 Q 与 prefill K -> 基线 score
  2. δ 干预 + capture 捕获修改后 K -> 干预 score
  3. 断言: 目标索引处 Δscore ≈ δ（逐头精确），非目标处 ≈ 0

用法:
  python scripts/verify_delta_injection.py --model qwen25vl --delta 5 --layer 22
  python scripts/verify_delta_injection.py --model qwen3vl --delta 5 --layer 27
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from layoutsteer.config import make_model_config
from layoutsteer.datasets import get_dataset
from layoutsteer.model_loader import setup_seeds
from layoutsteer.adapters import get_adapter


def prefill_scores(probe):
    """生成位逐头 score 向量 [heads, kv_len]。

    Tier A/B: 捕获 Q 与 prefill K 重算；Tier C (ScoreAddProbe):
    直接取捕获的生成位 score 行。
    """
    if getattr(probe, "_captured_scores", None):
        return probe._captured_scores[0].float().squeeze(0).squeeze(1)
    q = probe._captured_q[0].float()          # [1, H, 1, d]
    k = probe._k_list[0].float()              # [1, KV, seq, d]
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
    scale = q.shape[-1] ** -0.5
    return (q @ k.transpose(-2, -1)).squeeze(0).squeeze(1) * scale  # [H, seq]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen25vl")
    parser.add_argument("--dataset", type=str, default="sroie")
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--layer", type=int, default=22,
                        help="干预层 = 捕获层（单层验证）")
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--tol", type=float, default=0.05)
    args = parser.parse_args()

    setup_seeds()
    ds = get_dataset(args.dataset)
    bbox_map = ds.load_bboxes("gt")
    adapter = get_adapter(make_model_config(args.model))
    adapter.load()
    adapter.bind_dataset(ds)
    probe = adapter.create_probe(capture_layer=args.layer)

    n_pass = n_total = 0
    for task in ds.iter_tasks():
        if n_total >= args.n_tasks:
            break
        boxes_px = bbox_map.get(task.key)
        if not boxes_px or not os.path.exists(task.image_path):
            continue
        raw = Image.open(task.image_path).convert("RGB")
        image = adapter.prepare_image(raw)
        inputs = adapter.build_inputs(image, ds.build_prompt(task),
                                      sample_name=task.sample_name)
        adapter.set_intervention_target(probe, inputs, boxes_px, raw.size)

        with probe.attach(delta=0.0, capture=True):
            adapter.generate(inputs, max_new_tokens=1)
        s_normal = prefill_scores(probe)

        with probe.attach(delta=args.delta, target_layers={args.layer},
                          capture=True):
            adapter.generate(inputs, max_new_tokens=1)
        s_steered = prefill_scores(probe)

        diff = s_steered - s_normal                     # [H, seq]
        kv_len = diff.shape[1]
        idx, _ = probe._target_seq_indices(kv_len, diff.device)
        if idx is None or len(idx) == 0:
            print(f"{task.key}: 目标 0 token（框落在视觉裁剪外）-> 跳过",
                  flush=True)
            del inputs
            torch.cuda.empty_cache()
            continue
        mask = torch.zeros(kv_len, dtype=torch.bool)
        mask[idx.cpu()] = True
        d_target = diff[:, mask]
        d_other = diff[:, ~mask]
        err_t = (d_target - args.delta).abs().max().item()
        err_o = d_other.abs().max().item()
        ok = err_t < args.tol and err_o < args.tol
        n_pass += int(ok)
        n_total += 1
        print(f"{task.key}: 目标 {int(mask.sum())} token  "
              f"Δ目标={d_target.mean().item():.4f}(期望 {args.delta:g}, "
              f"最大误差 {err_t:.4f})  非目标最大|Δ|={err_o:.4f}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        del inputs
        torch.cuda.empty_cache()

    print(f"\n{args.model} L{args.layer} δ={args.delta:g}: "
          f"{n_pass}/{n_total} 通过")
    sys.exit(0 if n_pass == n_total and n_total > 0 else 1)


if __name__ == "__main__":
    main()
