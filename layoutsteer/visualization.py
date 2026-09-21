"""可选可视化：注意力热图叠加（基于 probe 捕获层的 Q/K）。"""
import os

import numpy as np
import cv2
from PIL import Image


def overlay_heatmap(heatmap_raw, image: Image.Image) -> np.ndarray:
    """热图叠加到原图（沿用旧版渲染风格）。"""
    if heatmap_raw is None:
        return np.array(image)
    cam = cv2.resize(heatmap_raw, image.size, interpolation=cv2.INTER_LINEAR)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = heatmap * 0.5 + np.array(image) * 0.5
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_attention_comparison(normal_maps, steered_maps, image, save_dir):
    """保存正常 / 干预（生成 token 平均）注意力叠加图与差异图。"""
    if not normal_maps or not steered_maps:
        return
    os.makedirs(save_dir, exist_ok=True)

    normal_avg = np.mean(normal_maps, axis=0)
    steered_avg = np.mean(steered_maps, axis=0)

    Image.fromarray(overlay_heatmap(normal_avg, image)).save(
        os.path.join(save_dir, "attention_normal.jpg"))
    Image.fromarray(overlay_heatmap(steered_avg, image)).save(
        os.path.join(save_dir, "attention_steered.jpg"))
    Image.fromarray(overlay_heatmap(steered_avg - normal_avg, image)).save(
        os.path.join(save_dir, "attention_difference.jpg"))
