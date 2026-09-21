"""embedding 缓存的统一存取层。

四类缓存全部按"每条目一个 .npy 文件"组织:
  train_global/{sample}.npy            训练样本全局布局 embedding
  test_global/{sample}.npy             测试样本全局布局 embedding
  test_local_{tag}/{sample}.npy        测试样本全部候选的局部 embedding 矩阵 (N, D)
  template_local_{tag}/{id}__{ent}.npy 训练模板实体局部 embedding
                                       (shape (0,) 空数组 = 该模板无此实体)
tag 携带影响局部渲染的超参 (n_local / wrap_gap)，参数变更自动走新目录。
"""
import os

import numpy as np

from ..config import CACHE_DIR


def cache_dirs(ds_name: str, loc_cfg, model: str = "qwen25vl") -> dict:
    """embedding 缓存目录（模型相关: embedding 由各自模型的视觉塔提取）。

    非默认模型全部目录加 _{model} 后缀, qwen25vl 保持历史路径不变。
    """
    sfx = "" if model == "qwen25vl" else f"_{model}"
    return {
        "train_global": os.path.join(
            CACHE_DIR, f"{ds_name}_train_global{sfx}"),
        "test_global": os.path.join(
            CACHE_DIR, f"{ds_name}_test_global{sfx}"),
        "test_local": os.path.join(
            CACHE_DIR, f"{ds_name}_test_local_{loc_cfg.local_tag}{sfx}"),
        "template": os.path.join(
            CACHE_DIR, f"{ds_name}_template_local_{loc_cfg.local_tag}{sfx}"),
    }


def save_npy(path, result):
    """原子写入（tmp + os.replace），多进程并发安全；None 存为空数组标记。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}.npy"
    np.save(tmp, np.empty(0) if result is None else result)
    os.replace(tmp, path)


def load_npy(path, hint: str = ""):
    """加载缓存；文件缺失报错并提示先跑 embed，空数组标记返回 None。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"embedding 缓存缺失: {path}\n"
            f"请先运行离线 embedding: python scripts/embed.py {hint}")
    arr = np.load(path)
    return None if arr.size == 0 else arr


def load_train_index(train_global_dir: str, names, hint: str = ""):
    """按名单加载训练全局 embedding 并堆叠为检索矩阵 (N, D)。"""
    embs = [load_npy(os.path.join(train_global_dir, f"{n}.npy"), hint)
            for n in names]
    return np.stack(embs)
