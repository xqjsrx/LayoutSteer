"""Stage 2: 局部匹配（模板局部布局 embedding + 投票 + 空间聚类）。"""
import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering

from .layout_render import bbox_center, get_local_bboxes, generate_local_layout
from ..config import LocalizationConfig


def _image_size(layout_info):
    img = Image.open(layout_info["image_path"])
    size = img.size
    img.close()
    return size


def _local_embedding(embedder, boxes, center_idx, img_size, cfg):
    local_bboxes, region = get_local_bboxes(
        boxes, center_idx, cfg.n_local_bboxes, img_size,
        wrap_gap_ratio=cfg.wrap_gap_ratio, use_wrap=cfg.use_wrap)
    layout_img = generate_local_layout(local_bboxes, region)
    return embedder.extract(layout_img)


def template_local_embedding(embedder, layout_info, entity, cfg):
    """训练模板：以实体 GT 框为中心的局部布局 embedding（无该实体返回 None）。"""
    items = layout_info["items"]
    entity_indices = [i for i, it in enumerate(items) if it.entity == entity]
    if not entity_indices:
        return None
    boxes = [it.box for it in items]
    img_size = _image_size(layout_info)
    return _local_embedding(embedder, boxes, entity_indices[0], img_size, cfg)


def candidate_local_embeddings(embedder, layout_info, cfg):
    """测试样本：每个 OCR 框为中心各生成一个局部布局 embedding。"""
    boxes = [it.box for it in layout_info["items"]]
    img_size = _image_size(layout_info)
    return np.stack([
        _local_embedding(embedder, boxes, ci, img_size, cfg)
        for ci in range(len(boxes))
    ])


def vote_and_cluster(template_embs, candidate_embs, candidate_boxes,
                     cfg: LocalizationConfig,
                     template_anchor_ys=None, candidate_ys=None,
                     template_weights=None):
    """多模板投票 + 层次聚类，返回 (votes, clusters)。

    votes:    [{bbox_idx, total_weight}]，按权重降序
    clusters: [{bbox_indices, total_weight, n_votes}]，按权重降序

    position_prior > 0 且提供锚点/候选归一化 y 时，投票相似度减去
    λ*|y_cand - y_anchor|，抵消环面拼接造成的页首/页底邻域内容对称性。
    template_weights: 逐模板票权系数（记忆机制的伪模板降权），None=全 1。
    """
    use_prior = (cfg.position_prior > 0 and template_anchor_ys is not None
                 and candidate_ys is not None)
    vote_dict = {}
    for t_idx, t_emb in enumerate(template_embs):
        w_t = 1.0 if template_weights is None else float(template_weights[t_idx])
        sims = candidate_embs @ t_emb
        if use_prior:
            sims = sims - cfg.position_prior * np.abs(
                candidate_ys - template_anchor_ys[t_idx])
        for c_idx in np.argsort(sims)[::-1][:cfg.local_topk]:
            entry = vote_dict.setdefault(int(c_idx), {"total_weight": 0.0, "n": 0})
            entry["total_weight"] += w_t * float(sims[c_idx])
            entry["n"] += 1

    votes = [{"bbox_idx": k, "total_weight": v["total_weight"], "n_votes": v["n"]}
             for k, v in vote_dict.items()]
    votes.sort(key=lambda x: x["total_weight"], reverse=True)
    if not votes:
        return [], []

    if len(votes) >= 2:
        centers = np.array([bbox_center(candidate_boxes[v["bbox_idx"]])
                            for v in votes])
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=cfg.cluster_dist).fit_predict(centers)
        clusters = []
        for c in range(labels.max() + 1):
            cluster_votes = [votes[i] for i in range(len(votes)) if labels[i] == c]
            clusters.append({
                "bbox_indices": [v["bbox_idx"] for v in cluster_votes],
                "total_weight": sum(v["total_weight"] for v in cluster_votes),
                "n_votes": len(cluster_votes),
            })
        clusters.sort(key=lambda x: x["total_weight"], reverse=True)
    else:
        clusters = [{"bbox_indices": [votes[0]["bbox_idx"]],
                     "total_weight": votes[0]["total_weight"],
                     "n_votes": 1}]

    return votes, clusters
