"""评估指标：与 LayTextLLM 项目一致的文本归一化、准确率与 micro F-score。"""


def normalize_text(text):
    if isinstance(text, list):
        text = text[0] if text else ""
    return text.lower().translate(str.maketrans("", "", " ().,\n-"))


def is_correct_prediction(pred, gt) -> bool:
    return normalize_text(pred) == normalize_text(gt)


def format_for_evaluation(results):
    """[{sample_name, task, prediction, ground_truth}, ...] -> (gts, preds)。

    按 sample_name 分组、task_type 为字段键，符合 LayTextLLM evalFscore 输入格式。
    """
    gts, preds = {}, {}
    for r in results:
        sample_name = r["sample_name"]
        key = r["task"]
        gts.setdefault(sample_name, {}).setdefault(key, []).append(r["ground_truth"])
        preds.setdefault(sample_name, {}).setdefault(key, []).append(r["prediction"])
    return gts, preds


def eval_fscore(gts, preds):
    """LayTextLLM 一致的 micro P/R/F1（子串包含判定）。"""
    total_tp = total_fp = total_fn = 0

    for key in gts:
        gt_set = {k.strip(): {normalize_text(v) for v in vs}
                  for k, vs in gts[key].items()}
        pred_set = {k.strip(): {normalize_text(v) for v in vs}
                    for k, vs in preds.get(key, {}).items()}

        for label in gt_set:
            pred_items = pred_set.get(label, set())
            total_tp += sum(1 for g in gt_set[label]
                            if any(g in p for p in pred_items))
            total_fp += sum(1 for p in pred_items
                            if all(p not in g for g in gt_set[label]))
            total_fn += sum(1 for g in gt_set[label]
                            if all(g not in p for p in pred_items))

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0)
    return {"micro_precision": precision, "micro_recall": recall,
            "micro_f1_score": f1}


def task_wise_accuracy(results):
    """按 task_type 统计正确率。"""
    stats = {}
    for r in results:
        s = stats.setdefault(r["task"], {"correct": 0, "total": 0})
        s["total"] += 1
        if r["is_correct"]:
            s["correct"] += 1
    for s in stats.values():
        s["accuracy"] = s["correct"] / s["total"] if s["total"] else 0
    return stats


def summarize(normal_results, intervened_results):
    """生成 normal vs intervened 的完整评估摘要。"""
    def _side(results):
        correct = sum(1 for r in results if r["is_correct"])
        total = len(results)
        gts, preds = format_for_evaluation(results)
        return {
            "per_sample_accuracy": correct / total if total else 0,
            "correct_samples": correct,
            "total_samples": total,
            "task_wise_accuracy": task_wise_accuracy(results),
            "fscore_metrics": eval_fscore(gts, preds),
        }

    normal = _side(normal_results)
    intervened = _side(intervened_results)
    comparison = {
        "accuracy_difference": (intervened["per_sample_accuracy"]
                                - normal["per_sample_accuracy"]),
        "f1_difference": (intervened["fscore_metrics"]["micro_f1_score"]
                          - normal["fscore_metrics"]["micro_f1_score"]),
        "task_wise_differences": {
            t: (intervened["task_wise_accuracy"].get(t, {}).get("accuracy", 0)
                - normal["task_wise_accuracy"].get(t, {}).get("accuracy", 0))
            for t in set(normal["task_wise_accuracy"]) | set(intervened["task_wise_accuracy"])
        },
    }
    return {"normal": normal, "intervened": intervened, "comparison": comparison}
