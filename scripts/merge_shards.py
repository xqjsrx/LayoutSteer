"""合并分片运行目录 -> 一个完整结果目录（含重新评估）。

用法: python scripts/merge_shards.py <shard_dir...> <out_dir>
每个分片目录须含 normal_results.json / intervened_results.json，
按 (sample_name, question) 去重成对合并。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layoutsteer.evaluation.metrics import summarize


def main():
    *shard_dirs, out_dir = sys.argv[1:]
    normal, steered = {}, {}
    for d in shard_dirs:
        for name, store in (("normal_results.json", normal),
                            ("intervened_results.json", steered)):
            p = os.path.join(d, name)
            if not os.path.exists(p):
                continue
            for r in json.load(open(p)):
                store[(r["sample_name"], r["question"])] = r
    assert normal and steered, "分片目录里没有可合并的结果"
    n_list = list(normal.values())
    s_list = list(steered.values())
    os.makedirs(out_dir, exist_ok=True)
    for name, data in (("normal_results.json", n_list),
                       ("intervened_results.json", s_list)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    summary = summarize(n_list, s_list)
    with open(os.path.join(out_dir, "evaluation_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    print(f"合并 {len(shard_dirs)} 片: n={len(n_list)}  "
          f"Δ={summary['comparison']['accuracy_difference']:+.2%}  -> {out_dir}")


if __name__ == "__main__":
    main()
