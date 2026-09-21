"""冒烟测试：3 个 SROIE 样本上验证 score_delta 干预链路。

验证点:
  1. attach/detach 后 ALL_ATTENTION_FUNCTIONS 恢复干净
  2. 干预确实改变目标区域注意力统计（区域内注意力占比上升）
  3. 正常 / 干预推理均能正常生成
用法:
  python scripts/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from layoutsteer.config import ModelConfig
from layoutsteer.datasets import get_dataset
from layoutsteer.model_loader import (
    setup_seeds, load_model_and_processor, resize_image_by_pixel_limit,
    build_inputs, generate_and_decode)
from layoutsteer.intervention import (
    ScoreDeltaProbe, bboxes_px_to_rel, bboxes_to_grid_indices)


def region_attention_ratio(att_maps, grid_idx):
    """目标区域注意力占比（对所有生成 token 求平均）。"""
    if not att_maps:
        return 0.0
    ratios = []
    for m in att_maps:
        flat = m.flatten()
        ratios.append(float(flat[grid_idx.cpu().numpy()].sum() / (flat.sum() + 1e-8)))
    return sum(ratios) / len(ratios)


def main():
    setup_seeds()
    cfg = ModelConfig()
    ds = get_dataset("sroie")
    tasks = ds.iter_tasks()[:3]
    gt_bboxes = ds.load_bboxes("gt")

    model, processor = load_model_and_processor(cfg)
    original_fa2 = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
    probe = ScoreDeltaProbe(model, processor)

    for task in tasks:
        print(f"\n===== {task.sample_name} / {task.task_type} =====")
        raw_image = Image.open(task.image_path).convert("RGB")
        orig_size = raw_image.size
        image = resize_image_by_pixel_limit(raw_image, cfg.max_image_pixels)

        prompt = ds.build_prompt(task)
        inputs = build_inputs(processor, image, prompt, model.device)
        probe.setup_image_range(inputs["input_ids"], inputs["image_grid_thw"])

        boxes_px = gt_bboxes.get(task.key)
        assert boxes_px, f"缺少 GT bbox: {task.key}"
        boxes_rel = bboxes_px_to_rel(boxes_px, orig_size)
        grid_idx = bboxes_to_grid_indices(boxes_rel, probe.spatial_shape)
        probe.set_region(grid_idx)
        print(f"图像网格: {probe.spatial_shape}, 干预 token 数: {len(grid_idx)}")

        # 基线（仅捕获）
        with probe.attach(delta=0.0, capture=True):
            answer_normal = generate_and_decode(model, processor, inputs)
        ratio_normal = region_attention_ratio(probe.get_attention_maps(), grid_idx)

        # 干预
        with probe.attach(delta=5.0, capture=True):
            answer_steered = generate_and_decode(model, processor, inputs)
        ratio_steered = region_attention_ratio(probe.get_attention_maps(), grid_idx)

        print(f"GT:       {task.answer}")
        print(f"正常回答: {answer_normal}")
        print(f"干预回答: {answer_steered}")
        print(f"区域注意力占比: normal={ratio_normal:.4f} -> steered={ratio_steered:.4f}")
        assert ratio_steered > ratio_normal, "干预未提升目标区域注意力!"

        torch.cuda.empty_cache()

    assert ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is original_fa2, \
        "detach 后 FA2 函数未恢复!"
    print("\n冒烟测试全部通过: attach/detach 干净, 干预生效。")


if __name__ == "__main__":
    main()
