"""推理主循环：正常 vs score_delta 干预双推理，断点续跑 + 增量保存。"""
import os
import gc
import json

import torch
from PIL import Image
from tqdm import tqdm

from .config import RunConfig, parse_target_layers
from .datasets import get_dataset
from .model_loader import setup_seeds
from .adapters import get_adapter
from .evaluation.metrics import is_correct_prediction, summarize
from .visualization import save_attention_comparison


def _load_json_list(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"警告: {path} 损坏，忽略并重新开始。")
    return []


def _save_json(path, data):
    # tmp + replace 原子写: 进程被杀时不会留下截断的 JSON
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)


class ExperimentRunner:

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        setup_seeds()
        self.dataset = get_dataset(cfg.dataset)
        if cfg.bbox_json:
            self.dataset.set_retrieval_bbox_path(cfg.bbox_json)
        self.bbox_map = self.dataset.load_bboxes(cfg.bbox_source)
        # 定位置信度（gt 源/无标定时恒 1.0，行为退化为固定 δ）
        self.conf_map = (self.dataset.load_confidences(cfg.bbox_source)
                         if cfg.intervention.use_confidence else {})
        # normal 基线复用: 贪心生成确定性已验证（三次独立运行 100% 一致）
        self.normal_cache = {}
        if cfg.normal_from:
            with open(cfg.normal_from, "r", encoding="utf-8") as f:
                self.normal_cache = {(r["sample_name"], r["question"]): r["prediction"]
                                     for r in json.load(f)}
            print(f"normal 基线复用: {len(self.normal_cache)} 条 ({cfg.normal_from})")
        self.adapter = get_adapter(cfg.model)
        self.adapter.load()
        self.adapter.bind_dataset(self.dataset)
        self.model = self.adapter.model
        self.processor = self.adapter.processor
        self.probe = self.adapter.create_probe(
            capture_layer=cfg.intervention.capture_layer)
        self.target_layers = parse_target_layers(
            cfg.intervention.target_layers, self.adapter.n_layers)

    # ── 单任务 ────────────────────────────────────────────────────
    def process_task(self, task, boxes_px):
        cfg = self.cfg
        raw_image = Image.open(task.image_path).convert("RGB")
        orig_size = raw_image.size
        image = self.adapter.prepare_image(raw_image)

        prompt = self.dataset.build_prompt(task)
        inputs = self.adapter.build_inputs(image, prompt,
                                           sample_name=task.sample_name)
        self.adapter.set_intervention_target(
            self.probe, inputs, boxes_px, orig_size)

        # 正常推理（可视化时挂 capture-only probe；命中基线缓存时直接复用）
        normal_maps = None
        cache_key = (task.sample_name, task.question)
        if cfg.visualize:
            with self.probe.attach(delta=0.0, capture=True):
                pred_normal = self.adapter.generate(inputs, cfg.max_new_tokens)
            normal_maps = self.probe.get_attention_maps()
        elif cache_key in self.normal_cache:
            pred_normal = self.normal_cache[cache_key]
        else:
            pred_normal = self.adapter.generate(inputs, cfg.max_new_tokens)

        # 干预推理：δ_eff = δ × c^γ（置信度加权，低于 cutoff 关断）
        conf = 1.0
        if cfg.intervention.use_confidence:
            conf = float(self.conf_map.get(task.key, 1.0))
        if cfg.intervention.use_confidence and conf < cfg.intervention.conf_cutoff:
            delta_eff = 0.0
        else:
            delta_eff = cfg.intervention.delta * conf ** cfg.intervention.conf_gamma
        with self.probe.attach(delta=delta_eff,
                               target_layers=self.target_layers,
                               capture=cfg.visualize,
                               persistent=cfg.intervention.persistent):
            pred_steered = self.adapter.generate(inputs, cfg.max_new_tokens)

        if cfg.visualize:
            steered_maps = self.probe.get_attention_maps()
            vis_dir = os.path.join(
                cfg.output_dir, "visualizations", task.task_type, task.sample_name)
            save_attention_comparison(normal_maps, steered_maps, image, vis_dir)

        def _result(pred):
            return {
                "image": os.path.basename(task.image_path),
                "task": task.task_type,
                "question": task.question,
                "prediction": pred,
                "ground_truth": task.answer,
                "is_correct": is_correct_prediction(pred, task.answer),
                "sample_name": task.sample_name,
            }

        del inputs
        gc.collect()
        torch.cuda.empty_cache()
        normal_r, steered_r = _result(pred_normal), _result(pred_steered)
        if cfg.intervention.use_confidence:
            steered_r["confidence"] = round(conf, 4)
            steered_r["delta_eff"] = round(delta_eff, 4)
        return normal_r, steered_r

    # ── 主循环 ────────────────────────────────────────────────────
    def run(self):
        cfg = self.cfg
        os.makedirs(cfg.output_dir, exist_ok=True)
        normal_path = os.path.join(cfg.output_dir, "normal_results.json")
        steered_path = os.path.join(cfg.output_dir, "intervened_results.json")

        normal_results = _load_json_list(normal_path)
        steered_results = _load_json_list(steered_path)
        done = {(r["sample_name"], r["question"]) for r in normal_results}
        if done:
            print(f"断点续跑: 已完成 {len(done)} 个任务。")

        tasks = self.dataset.iter_tasks()
        if cfg.n_tasks > 0:
            tasks = tasks[:cfg.n_tasks]
        if cfg.shard:
            i, n = map(int, cfg.shard.split("/"))
            tasks = tasks[i::n]
            print(f"分片 {i}/{n}: 本进程负责 {len(tasks)} 个任务")

        with tqdm(total=len(tasks), desc=f"{cfg.dataset}/{cfg.bbox_source}") as pbar:
            pbar.update(len([t for t in tasks
                             if (t.sample_name, t.question) in done]))
            for task in tasks:
                if (task.sample_name, task.question) in done:
                    continue
                boxes_px = self.bbox_map.get(task.key)
                if not boxes_px:
                    print(f"跳过 {task.key}: 无可用 bbox")
                    pbar.update(1)
                    continue
                if not os.path.exists(task.image_path):
                    print(f"跳过 {task.key}: 图像不存在 {task.image_path}")
                    pbar.update(1)
                    continue
                try:
                    normal_r, steered_r = self.process_task(task, boxes_px)
                    normal_results.append(normal_r)
                    steered_results.append(steered_r)
                    _save_json(normal_path, normal_results)
                    _save_json(steered_path, steered_results)
                except Exception as e:
                    print(f"处理 {task.key} 出错: {e}")
                    import traceback
                    traceback.print_exc()
                pbar.update(1)

        print(f"\n全部完成。结果目录: {cfg.output_dir}")
        return self.evaluate(normal_results, steered_results)

    def evaluate(self, normal_results, steered_results):
        summary = summarize(normal_results, steered_results)
        summary["config"] = {
            "dataset": self.cfg.dataset,
            "bbox_source": self.cfg.bbox_source,
            "delta": self.cfg.intervention.delta,
            "target_layers": self.cfg.intervention.target_layers,
        }
        eval_path = os.path.join(self.cfg.output_dir, "evaluation_results.json")
        _save_json(eval_path, summary)

        n, s = summary["normal"], summary["intervened"]
        print(f"正常推理准确率:  {n['correct_samples']}/{n['total_samples']}"
              f" = {n['per_sample_accuracy']:.2%}  F1={n['fscore_metrics']['micro_f1_score']:.4f}")
        print(f"干预推理准确率:  {s['correct_samples']}/{s['total_samples']}"
              f" = {s['per_sample_accuracy']:.2%}  F1={s['fscore_metrics']['micro_f1_score']:.4f}")
        print(f"干预效果: 准确率 {summary['comparison']['accuracy_difference']:+.2%}, "
              f"F1 {summary['comparison']['f1_difference']:+.4f}")
        print("按任务类型变化:")
        for t, d in sorted(summary["comparison"]["task_wise_differences"].items()):
            print(f"  {t}: {d:+.2%}")
        print(f"评估结果已保存: {eval_path}")
        return summary
