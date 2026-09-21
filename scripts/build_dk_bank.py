"""离线构建训练集 dK bank: 检索记忆的"干预方向"部分。

为检索库（训练集）每样本×每实体捕获问题末 token Q 并求解 dK(δ=1)。
运行时: 测试样本检索到的 top-k 邻居 -> 邻居 dK 均值 = 近似干预方向。
与 layout 掩码共用同一次检索——"一次检索读出 where(掩码) + how(dK)"。

用法:
  CUDA_VISIBLE_DEVICES=4 python scripts/build_dk_bank.py --shard 0/4
  python scripts/build_dk_bank.py --merge
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from layoutsteer.config import RunConfig, OUTPUT_DIR, parse_target_layers
from layoutsteer.datasets import get_dataset
from layoutsteer.model_loader import (
    setup_seeds, load_model_and_processor, resize_image_by_pixel_limit,
    build_inputs)

BANK_DIR = os.path.join(OUTPUT_DIR, "sroie", "qtemplate")


def bank_path(split):
    return os.path.join(BANK_DIR, f"dk_bank_{split}.pt")


class QRecorder:
    """只读捕获各目标层末位置 Q。"""

    def __init__(self, target_layers):
        self.target_layers = target_layers
        self.q_last = {}
        self._orig = None

    def __enter__(self):
        self._orig = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        def wrapper(module, q, k, v, attention_mask=None, **kw):
            layer = getattr(module, "layer_idx", None)
            if layer in self.target_layers and q.shape[2] > 1:
                self.q_last[layer] = q[:, :, -1, :].detach().float().cpu()
            return self._orig(module, q, k, v, attention_mask=attention_mask, **kw)

        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = wrapper
        return self

    def __exit__(self, *a):
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = self._orig


def solve_dk_all(q_last, num_kv=4):
    """{layer: [1,H,d]} -> {layer: tensor[num_kv, d]}（δ=1 标定, float16）。"""
    out = {}
    for layer, q in q_last.items():
        H, d = q.shape[1], q.shape[2]
        groups = H // num_kv
        dks = torch.zeros(num_kv, d)
        for h in range(num_kv):
            q_h = q[0, h * groups:(h + 1) * groups, :]
            QQt = q_h @ q_h.T + 1e-6 * torch.eye(groups)
            Q_pinv = q_h.T @ torch.linalg.inv(QQt)
            dks[h] = (d ** 0.5) * (Q_pinv @ torch.ones(groups))
        out[layer] = dks.half()
    return out


def merge(split):
    bank = {}
    for p in sorted(glob.glob(os.path.join(
            BANK_DIR, f"dk_bank_{split}.shard*.pt"))):
        bank.update(torch.load(p))
    torch.save(bank, bank_path(split))
    n_entities = sum(len(v) for v in bank.values())
    size_mb = os.path.getsize(bank_path(split)) / 1e6
    print(f"合并完成: {len(bank)} 样本 / {n_entities} (样本,实体) 条目 "
          f"-> {bank_path(split)} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"])
    parser.add_argument("--shard", type=str, default="")
    parser.add_argument("--layers", type=str, default="19-27")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge(args.split)
        return

    cfg = RunConfig()
    setup_seeds()
    ds = get_dataset("sroie")
    os.makedirs(BANK_DIR, exist_ok=True)

    # 每实体标准 prompt（问题文本模板化，取测试任务首例）
    entity_prompts = {}
    for task in ds.iter_tasks():
        if task.task_type not in entity_prompts:
            entity_prompts[task.task_type] = ds.build_prompt(task)
        if len(entity_prompts) == len(ds.entity_types):
            break

    train_layouts = ds.load_layout_items(args.split)
    names = sorted(train_layouts.keys())
    tag = ""
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        names = names[i::n]
        tag = f".shard{i}of{n}"
    out_path = os.path.join(BANK_DIR, f"dk_bank_{args.split}{tag}.pt")

    bank = {}
    if os.path.exists(out_path):
        bank = torch.load(out_path)
        print(f"断点续跑: 已有 {len(bank)} 样本")

    model, processor = load_model_and_processor(cfg.model)
    target_layers = parse_target_layers(args.layers,
                                        model.config.num_hidden_layers)

    for idx, name in enumerate(names):
        if name in bank:
            continue
        img_path = train_layouts[name]["image_path"]
        if not os.path.exists(img_path):
            continue
        raw = Image.open(img_path).convert("RGB")
        image = resize_image_by_pixel_limit(raw, cfg.model.max_image_pixels)
        entry = {}
        for entity, prompt in entity_prompts.items():
            inputs = build_inputs(processor, image, prompt, model.device)
            with torch.no_grad(), QRecorder(target_layers) as rec:
                model(**inputs, use_cache=False)
            entry[entity] = solve_dk_all(rec.q_last)
            del inputs
        bank[name] = entry
        if (idx + 1) % 20 == 0:
            torch.save(bank, out_path + ".tmp")
            os.replace(out_path + ".tmp", out_path)
            torch.cuda.empty_cache()
            print(f"[{tag or 'all'}] {len(bank)}/{len(names)}", flush=True)
    torch.save(bank, out_path + ".tmp")
    os.replace(out_path + ".tmp", out_path)
    print(f"完成 {len(bank)} 样本 -> {out_path}")


if __name__ == "__main__":
    main()
