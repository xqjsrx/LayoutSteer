"""独立评估入口：对已有推理结果重新计算指标。

用法:
  python scripts/evaluate.py --result-dir output/sroie/gt_delta5_layers-all
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layoutsteer.evaluation import summarize


def main():
    parser = argparse.ArgumentParser(description="LayoutSteer 结果评估")
    parser.add_argument("--result-dir", type=str, required=True,
                        help="包含 normal_results.json / intervened_results.json 的目录")
    args = parser.parse_args()

    normal_path = os.path.join(args.result_dir, "normal_results.json")
    steered_path = os.path.join(args.result_dir, "intervened_results.json")
    for p in (normal_path, steered_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    with open(normal_path, "r", encoding="utf-8") as f:
        normal_results = json.load(f)
    with open(steered_path, "r", encoding="utf-8") as f:
        steered_results = json.load(f)

    summary = summarize(normal_results, steered_results)

    n, s = summary["normal"], summary["intervened"]
    print(f"任务数: {n['total_samples']}")
    print(f"正常推理: acc={n['per_sample_accuracy']:.2%} "
          f"F1={n['fscore_metrics']['micro_f1_score']:.4f}")
    print(f"干预推理: acc={s['per_sample_accuracy']:.2%} "
          f"F1={s['fscore_metrics']['micro_f1_score']:.4f}")
    print(f"干预效果: 准确率 {summary['comparison']['accuracy_difference']:+.2%}, "
          f"F1 {summary['comparison']['f1_difference']:+.4f}")
    print("\n按任务类型:")
    for t in sorted(summary["comparison"]["task_wise_differences"]):
        na = n["task_wise_accuracy"].get(t, {}).get("accuracy", 0)
        sa = s["task_wise_accuracy"].get(t, {}).get("accuracy", 0)
        print(f"  {t}: normal={na:.2%} intervened={sa:.2%} "
              f"({sa - na:+.2%})")

    out_path = os.path.join(args.result_dir, "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    print(f"\n评估结果已保存: {out_path}")


if __name__ == "__main__":
    main()
