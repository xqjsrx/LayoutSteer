"""SROIE2019 数据集适配器。

数据来自统一目录约定 dataset/sroie/（内容与旧版 refactor_fix/run_sroie.py 对齐）:
  - QA:        test/qa_test.json (LayTextLLM 格式)
  - GT 答案框: {test,train}/answer_bboxes.json (裁剪图坐标系)
  - 图片:      {test,train}/images (裁剪图)
"""
import os
import json

from .base import KIEDataset, Task, LayoutItem
from ..config import DATASET_PATHS

ANSWER_PROMPTS = {
    "company": "Please only answer with the company name.",
    "date": "Please only answer with the date.",
    "address": "Please only answer with the address.",
    "total": "Please only answer with the total amount.",
}


def _task_type_from_question(question: str) -> str:
    q = question.lower()
    for t in ("company", "address", "total", "date"):
        if t in q:
            return t
    return "other"


class SROIEDataset(KIEDataset):
    name = "sroie"
    entity_types = ["company", "date", "address", "total"]

    def __init__(self):
        self.paths = DATASET_PATHS["sroie"]

    def iter_tasks(self):
        with open(self.paths.test_qa_json, "r") as f:
            qa_data = json.load(f)

        tasks = []
        for item in qa_data:
            metadata = item.get("metadata", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata or "{}")
            sample_name = metadata.get("sample_name", "unknown")
            image_path = os.path.join(self.paths.test_image_dir, f"{sample_name}.jpg")
            tasks.append(Task(
                sample_name=sample_name,
                image_path=image_path,
                question=item["question"],
                answer=item["answer"],
                task_type=_task_type_from_question(item["question"]),
            ))
        return tasks

    def load_bboxes(self, source: str = "gt"):
        if source == "gt":
            bbox_json = self.paths.test_bbox_json
        elif source == "retrieval":
            bbox_json = self.retrieval_bbox_path()
            if not os.path.exists(bbox_json):
                raise FileNotFoundError(
                    f"未找到定位结果 {bbox_json}，请先运行 scripts/localize.py --dataset sroie")
        else:
            raise ValueError(f"未知 bbox source: {source}")

        with open(bbox_json, "r") as f:
            bbox_data = json.load(f)

        bbox_dict = {}
        for item in bbox_data:
            key = (item["sample_name"], item["question_type"])
            boxes = [mb["box"] for mb in item.get("matching_bboxes", [])]
            if not boxes:
                boxes = [gt["box"] for gt in item.get("all_gt_items", [])
                         if gt["entity"] == item["question_type"]]
            if boxes:
                bbox_dict[key] = boxes
        return bbox_dict

    def load_confidences(self, source: str = "gt"):
        """(sample_name, task_type) -> 定位置信度。gt 源/缺失字段恒 1.0。"""
        if source != "retrieval":
            return {}
        with open(self.retrieval_bbox_path(), "r") as f:
            bbox_data = json.load(f)
        return {(item["sample_name"], item["question_type"]): item["confidence"]
                for item in bbox_data if "confidence" in item}

    def load_weight_maps(self, source: str = "gt"):
        """(sample_name, task_type) -> 空间置信场（得票框+权重列表）。"""
        if source != "retrieval":
            return {}
        with open(self.retrieval_bbox_path(), "r") as f:
            bbox_data = json.load(f)
        return {(item["sample_name"], item["question_type"]): item["weight_map"]
                for item in bbox_data if item.get("weight_map")}

    def load_retrieved(self, source: str = "retrieval"):
        """(sample_name, task_type) -> 检索邻居样本 id 列表（top-k）。"""
        if source != "retrieval":
            return {}
        with open(self.retrieval_bbox_path(), "r") as f:
            bbox_data = json.load(f)
        return {(item["sample_name"], item["question_type"]): item["retrieved"]
                for item in bbox_data if item.get("retrieved")}

    def load_layout_items(self, split: str):
        """从 answer_bboxes.json 的 all_gt_items 还原每个样本的完整布局。"""
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
                continue  # 同一样本的多条 QA 共享同一份 all_gt_items
            gt_items = item.get("all_gt_items")
            if not gt_items:
                continue
            image_path = os.path.join(image_dir, f"{sample_name}.jpg")
            if not os.path.exists(image_path):
                continue
            layouts[sample_name] = {
                "image_path": image_path,
                "items": [LayoutItem(text=g["text"], box=g["box"], entity=g["entity"])
                          for g in gt_items],
            }
        return layouts

    def build_prompt(self, task: Task) -> str:
        answer_prompt = ANSWER_PROMPTS.get(
            task.task_type, "Please only answer with the required information.")
        return f"{task.question} {answer_prompt}"
