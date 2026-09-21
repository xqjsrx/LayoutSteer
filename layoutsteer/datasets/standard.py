"""通用 KIE 数据集适配器（CORD / FUNSD / POIE）。

三个数据集与 SROIE 共享同一 qa_test.json / answer_bboxes.json schema，
差异仅在图片扩展名与 task_type 的提取方式：
  - task_type 不能像 SROIE 那样从问句子串猜（cord 是自然短语、funsd 是
    表单字段名、poie 是营养字段模板），统一改从 test answer_bboxes.json
    构建 (sample_name, question) -> question_type 精确映射。
  - 无映射的 QA 条目（如 funsd 两条空字段名问题）直接跳过。
"""
import os
import json
import re

from .base import KIEDataset, Task, LayoutItem
from ..config import DATASET_PATHS


class StandardKIEDataset(KIEDataset):
    IMAGE_EXT = ".jpg"
    # task_type 取哪个字段: "question_type"（问句/字段名）或 "target_entity"（实体码）。
    # 定位流水线按 all_gt_items 的 entity 匹配模板，实体码风格的数据集（cord/poie）
    # 必须用 target_entity 才能对上；funsd 的 target_entity 恒为 answer，只能用问句字段名。
    TASK_TYPE_FIELD = "question_type"
    # 开放词表数据集（funsd）: 字段名归一化后作检索键，兼作缓存文件名（仅 [a-z0-9]）
    NORMALIZE_QTYPE = False

    def __init__(self):
        self.paths = DATASET_PATHS[self.name]
        self._qtype_map = None
        self._entity_types = None
        self._split_keys = {}

    # ── task_type 映射与实体列表（lazy） ─────────────────────────
    @staticmethod
    def _norm_key(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def _item_qtype(self, item):
        """从 bbox json 条目取 task_type。检索产物无 target_entity，回退 question_type。"""
        if self.TASK_TYPE_FIELD == "target_entity":
            te = item.get("target_entity")
            if te:
                return te.upper()
        qt = item["question_type"]
        return self._norm_key(qt) if self.NORMALIZE_QTYPE else qt

    def _question_type_map(self):
        if self._qtype_map is None:
            with open(self.paths.test_bbox_json, "r") as f:
                bbox_data = json.load(f)
            self._qtype_map = {(it["sample_name"], it["question"]): self._item_qtype(it)
                               for it in bbox_data}
        return self._qtype_map

    @property
    def entity_types(self):
        if self._entity_types is None:
            self._entity_types = sorted(
                {v for v in self._question_type_map().values() if v})
        return self._entity_types

    def sample_keys(self, sample_name, split="test"):
        if split not in self._split_keys:
            bbox_json = (self.paths.train_bbox_json if split == "train"
                         else self.paths.test_bbox_json)
            with open(bbox_json, "r") as f:
                bbox_data = json.load(f)
            keys = {}
            for it in bbox_data:
                qt = self._item_qtype(it)
                if qt:
                    keys.setdefault(it["sample_name"], set()).add(qt)
            self._split_keys[split] = {k: sorted(v) for k, v in keys.items()}
        return self._split_keys[split].get(sample_name, [])

    def iter_tasks(self):
        with open(self.paths.test_qa_json, "r") as f:
            qa_data = json.load(f)
        qtype_map = self._question_type_map()

        tasks, skipped = [], 0
        for item in qa_data:
            metadata = item.get("metadata", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata or "{}")
            sample_name = metadata.get("sample_name", "unknown")
            task_type = qtype_map.get((sample_name, item["question"]))
            if not task_type:
                skipped += 1
                continue
            tasks.append(Task(
                sample_name=sample_name,
                image_path=os.path.join(
                    self.paths.test_image_dir, f"{sample_name}{self.IMAGE_EXT}"),
                question=item["question"],
                answer=item["answer"],
                task_type=task_type,
            ))
        if skipped:
            print(f"[{self.name}] 跳过 {skipped} 条无 bbox 映射的 QA（如空字段名）")
        return tasks

    def load_bboxes(self, source: str = "gt"):
        if source == "gt":
            bbox_json = self.paths.test_bbox_json
        elif source == "retrieval":
            bbox_json = self.retrieval_bbox_path()
            if not os.path.exists(bbox_json):
                raise FileNotFoundError(
                    f"未找到定位结果 {bbox_json}，请先运行 "
                    f"scripts/localize.py --dataset {self.name}")
        else:
            raise ValueError(f"未知 bbox source: {source}")

        with open(bbox_json, "r") as f:
            bbox_data = json.load(f)

        bbox_dict = {}
        for item in bbox_data:
            qtype = self._item_qtype(item)
            key = (item["sample_name"], qtype)
            boxes = [mb["box"] for mb in item.get("matching_bboxes", [])]
            if not boxes:
                boxes = [gt["box"] for gt in item.get("all_gt_items", [])
                         if gt["entity"] == qtype]
            if boxes:
                bbox_dict[key] = boxes
        return bbox_dict

    def load_confidences(self, source: str = "gt"):
        if source != "retrieval":
            return {}
        with open(self.retrieval_bbox_path(), "r") as f:
            bbox_data = json.load(f)
        return {(item["sample_name"], self._item_qtype(item)): item["confidence"]
                for item in bbox_data if "confidence" in item}

    def load_weight_maps(self, source: str = "gt"):
        if source != "retrieval":
            return {}
        with open(self.retrieval_bbox_path(), "r") as f:
            bbox_data = json.load(f)
        return {(item["sample_name"], self._item_qtype(item)): item["weight_map"]
                for item in bbox_data if item.get("weight_map")}

    def load_retrieved(self, source: str = "retrieval"):
        if source != "retrieval":
            return {}
        with open(self.retrieval_bbox_path(), "r") as f:
            bbox_data = json.load(f)
        return {(item["sample_name"], self._item_qtype(item)): item["retrieved"]
                for item in bbox_data if item.get("retrieved")}

    def load_layout_items(self, split: str):
        if split == "train":
            bbox_json, image_dir = self.paths.train_bbox_json, self.paths.train_image_dir
        elif split == "test":
            bbox_json, image_dir = self.paths.test_bbox_json, self.paths.test_image_dir
        else:
            raise ValueError(f"未知 split: {split}")

        with open(bbox_json, "r") as f:
            bbox_data = json.load(f)

        layouts = {}
        for item in bbox_data:
            sample_name = item["sample_name"]
            if sample_name in layouts:
                continue
            gt_items = item.get("all_gt_items")
            if not gt_items:
                continue
            image_path = os.path.join(image_dir, f"{sample_name}{self.IMAGE_EXT}")
            if not os.path.exists(image_path):
                continue
            layouts[sample_name] = {
                "image_path": image_path,
                "items": [LayoutItem(text=g["text"], box=g["box"], entity=g["entity"])
                          for g in gt_items],
            }
        return layouts

    def build_prompt(self, task: Task) -> str:
        return f"{task.question} Please only answer with the value, no explanation."


class CORDDataset(StandardKIEDataset):
    name = "cord"
    IMAGE_EXT = ".png"
    TASK_TYPE_FIELD = "target_entity"


class FUNSDDataset(StandardKIEDataset):
    """开放词表表单 QA：归一化字段名作为检索键（闭集实体的推广）。

    模板池 P(key) = 含该字段的训练表单，锚点 = 该字段 QA 的答案框——
    通过把答案框对应的布局元素 entity 重标为 key，完整复用实体级定位流水线。
    """
    name = "funsd"
    IMAGE_EXT = ".png"
    NORMALIZE_QTYPE = True

    def sample_entities(self, sample_name):
        return self.sample_keys(sample_name, "test")

    def load_layout_items(self, split: str):
        layouts = super().load_layout_items(split)
        bbox_json = (self.paths.train_bbox_json if split == "train"
                     else self.paths.test_bbox_json)
        with open(bbox_json, "r") as f:
            bbox_data = json.load(f)

        box_key = {}
        for it in bbox_data:
            if it.get("target_entity") != "answer" or not it.get("question_type"):
                continue
            key = self._norm_key(it["question_type"])
            if not key:
                continue
            for mb in it.get("matching_bboxes", []):
                box_key.setdefault((it["sample_name"], tuple(mb["box"])), key)

        for sname, info in layouts.items():
            for item in info["items"]:
                key = box_key.get((sname, tuple(item.box)))
                # 未命中的保留原角色标签并加前缀，与归一化键域（[a-z0-9]）隔离
                item.entity = key if key else "_" + item.entity
        return layouts


class POIEDataset(StandardKIEDataset):
    name = "poie"
    IMAGE_EXT = ".jpg"
    TASK_TYPE_FIELD = "target_entity"
