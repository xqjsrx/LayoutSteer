"""Stage 1: 全局结构匹配（布局图 + post-merger embedding + cosine 检索）。

embedding 的计算在 scripts/embed.py 离线完成，本模块只提供
布局图渲染 + embedding 计算函数与纯 numpy 检索。
"""
import numpy as np
from PIL import Image

from .layout_render import generate_layout


def sample_layout_image(layout_info):
    """样本的全局布局图（黑底白框，画布=原图尺寸）。"""
    img = Image.open(layout_info["image_path"])
    size = img.size
    img.close()
    return generate_layout(size, [it.box for it in layout_info["items"]])


def compute_global_embedding(embedder, layout_info):
    """单个样本的全局布局 embedding。"""
    return embedder.extract(sample_layout_image(layout_info))


def global_match(test_emb, train_embs, train_names, topk):
    """用预计算的测试 embedding 检索 topk 相似训练样本名及相似度。"""
    sims = train_embs @ test_emb
    top_idx = np.argsort(sims)[::-1][:topk]
    return [(train_names[i], float(sims[i])) for i in top_idx]
