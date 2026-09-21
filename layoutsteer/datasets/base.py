"""数据集统一接口。

每个数据集适配器提供:
  - iter_tasks():         推理任务列表 (Task)
  - load_bboxes(source):  (sample_name, task_type) -> [bbox_px, ...]
  - load_layout_items():  定位阶段所需的 OCR 布局元素 (含 entity 标签)
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Task:
    """单条 KIE 推理任务。"""
    sample_name: str
    image_path: str
    question: str
    answer: str
    task_type: str          # 实体类别，如 company/date/address/total

    @property
    def key(self):
        return (self.sample_name, self.task_type)


@dataclass
class LayoutItem:
    """一个 OCR 元素：文本 + 像素框 + 实体标签（测试侧标签仅用于评估定位效果）。"""
    text: str
    box: list               # [x1, y1, x2, y2] 像素坐标
    entity: str


class KIEDataset(ABC):
    """KIE 数据集抽象基类。"""

    name: str = ""
    entity_types: list = []  # 参与定位的实体类别

    @abstractmethod
    def iter_tasks(self) -> list:
        """返回全部推理任务 list[Task]。"""

    @abstractmethod
    def load_bboxes(self, source: str = "gt") -> dict:
        """返回 {(sample_name, task_type): [box_px, ...]}。

        source: "gt" 读取 GT 答案框；"retrieval" 读取离线定位结果。
        """

    @abstractmethod
    def load_layout_items(self, split: str) -> dict:
        """返回 {sample_name: {"image_path": str, "items": [LayoutItem, ...]}}。"""

    @abstractmethod
    def build_prompt(self, task: Task) -> str:
        """构建发送给模型的完整问题文本。"""

    def load_confidences(self, source: str = "gt") -> dict:
        """返回 {(sample_name, task_type): 定位置信度}。

        默认空（消费侧对缺失 key 取 1.0，行为退化为固定强度干预）。
        """
        return {}

    def sample_entities(self, sample_name: str) -> list:
        """该测试样本需要定位的键列表（定位流水线用）。

        闭集数据集为全部实体；开放词表数据集（funsd）按样本返回被提问的字段键。
        """
        return self.entity_types

    def sample_keys(self, sample_name: str, split: str = "test") -> list:
        """该样本在该 split 标注中实际被提问的键（自举脚本用）。

        自举锚点来自注意力峰而非 GT 框，没有"实体缺失"信号，必须按样本裁剪键，
        否则会为文档中不存在的字段伪造锚点。默认退化为全部实体（SROIE 每文档
        四实体齐全）。
        """
        return self.entity_types

    def load_weight_maps(self, source: str = "gt") -> dict:
        """返回 {(sample_name, task_type): weight_map}（空间置信场，
        得票框+归一化权重列表）。默认空（消费侧回退到 bbox 均匀干预）。
        """
        return {}

    def retrieval_bbox_path(self) -> str:
        """离线定位结果的默认输出路径（可被 set_retrieval_bbox_path 覆盖，
        在线记忆多轮实验用不同轮次的定位产物而不覆盖主文件）。"""
        override = getattr(self, "_retrieval_bbox_override", None)
        if override:
            return override
        from ..config import OUTPUT_DIR
        return os.path.join(OUTPUT_DIR, self.name, f"localized_bboxes_{self.name}.json")

    def set_retrieval_bbox_path(self, path: str):
        """覆盖定位结果读写路径（load_bboxes/weight_maps/confidences/retrieved 均生效）。"""
        self._retrieval_bbox_override = path
