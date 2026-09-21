"""数据集注册表：名称 -> 适配器。"""
from .sroie import SROIEDataset
from .standard import CORDDataset, FUNSDDataset, POIEDataset

_REGISTRY = {
    "sroie": SROIEDataset,
    "cord": CORDDataset,
    "funsd": FUNSDDataset,
    "poie": POIEDataset,
}


def get_dataset(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"未知数据集 '{name}'，可选: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()
