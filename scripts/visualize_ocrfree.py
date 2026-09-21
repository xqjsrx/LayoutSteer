"""OCR-free 布局提取可视化: 形态学连通域伪文本行框 vs GT 框。

不识字、不含语义: 二值化 -> 水平膨胀粘连成行 -> 连通域外接框。
每样本输出 4 联图: 原图+形态学框(红) | 原图+GT框(绿) | 形态学布局图 | GT布局图

用法: python scripts/visualize_ocrfree.py --n 4
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image, ImageDraw

from layoutsteer.config import OUTPUT_DIR
from layoutsteer.datasets import get_dataset
from layoutsteer.localization.layout_render import generate_layout


def extract_text_boxes(img_bgr, dilate_ratio=0.04, min_area_ratio=1e-5,
                       max_h_ratio=0.25, line_h_ratio=0.008, line_w_ratio=0.5):
    """纯几何伪文本行框: Otsu 二值化 + 水平膨胀 + 连通域。

    过滤: 噪点 / 大竖条 / 分隔线（近全宽且极薄——文本行高度至少约
    1% 图高, 实线/虚线远低于此; 长宽比口径会误杀长文本行）。
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 票据为浅底深字, Otsu 反相后文本为白
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kw = max(3, int(W * dilate_ratio)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < W * H * min_area_ratio:
            continue                     # 噪点
        if h > H * max_h_ratio:
            continue                     # 大竖条(装订线/边框)
        if w < 3 or h < 3:
            continue
        if w > W * line_w_ratio and h < H * line_h_ratio:
            continue                     # 分隔线/虚线
        boxes.append([int(x), int(y), int(x + w), int(y + h)])
    return boxes


def draw_boxes(img, boxes, color):
    vis = img.convert("RGB").copy()
    d = ImageDraw.Draw(vis)
    for b in boxes:
        d.rectangle(b, outline=color, width=3)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--dilate-ratio", type=float, default=0.04)
    parser.add_argument("--samples", type=str, default="",
                        help="逗号分隔样本名，缺省取前 n 个")
    args = parser.parse_args()

    ds = get_dataset("sroie")
    layouts = ds.load_layout_items("test")
    names = (args.samples.split(",") if args.samples
             else sorted(layouts.keys())[:args.n])

    out_dir = os.path.join(OUTPUT_DIR, "sroie", "ocrfree_vis")
    os.makedirs(out_dir, exist_ok=True)

    for name in names:
        info = layouts[name]
        img = Image.open(info["image_path"]).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gt_boxes = [it.box for it in info["items"]]
        morph_boxes = extract_text_boxes(img_bgr,
                                         dilate_ratio=args.dilate_ratio)

        panels = [
            draw_boxes(img, morph_boxes, "red"),
            draw_boxes(img, gt_boxes, "green"),
            generate_layout(img.size, morph_boxes),
            generate_layout(img.size, gt_boxes),
        ]
        w, h = img.size
        canvas = Image.new("RGB", (w * 4 + 30, h), (128, 128, 128))
        for i, p in enumerate(panels):
            canvas.paste(p, (i * (w + 10), 0))
        out_path = os.path.join(out_dir, f"{name}.jpg")
        canvas.save(out_path, quality=88)
        print(f"{name}: morph={len(morph_boxes)} gt={len(gt_boxes)} "
              f"-> {out_path}")


if __name__ == "__main__":
    main()
