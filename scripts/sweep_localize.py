"""定位超参扫描（纯 numpy，embedding 内存复用，每组约 1 秒）。

用法:
  python scripts/sweep_localize.py --dataset sroie
  python scripts/sweep_localize.py --dataset sroie \
      --cluster-dist 30,50,80,120 --local-topk 3,5,8 --global-topk 10
"""
import os
import sys
import json
import time
import argparse
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layoutsteer.config import LocalizationConfig, OUTPUT_DIR
from layoutsteer.datasets import get_dataset
from layoutsteer.localization.pipeline import (
    EmbStore, run_localization, summarize_report, macro_avg)


def main():
    parser = argparse.ArgumentParser(description="定位超参网格扫描")
    parser.add_argument("--dataset", type=str, default="sroie")
    parser.add_argument("--cluster-dist", type=str, default="30,50,80,120")
    parser.add_argument("--local-topk", type=str, default="3,5,8")
    parser.add_argument("--global-topk", type=str, default="10")
    parser.add_argument("--merge-ratio", type=str, default="0",
                        help="多簇合并权重比例（0=关）")
    parser.add_argument("--trim-mult", type=str, default="0",
                        help="离群剔除倍数（0=关）")
    parser.add_argument("--region-pad", type=str, default="0",
                        help="区域框外扩比例（0=关）")
    parser.add_argument("--pos-prior", type=str, default="0",
                        help="位置先验惩罚系数 λ（0=关）")
    parser.add_argument("--gap", type=float, default=None,
                        help="wrap_gap_ratio（默认取 config；需对应 embedding 缓存存在）")
    parser.add_argument("--no-wrap", action="store_true",
                        help="使用无环面拼接的 embedding 缓存")
    parser.add_argument("--n-test", type=int, default=0)
    args = parser.parse_args()

    cds = [float(x) for x in args.cluster_dist.split(",")]
    lks = [int(x) for x in args.local_topk.split(",")]
    gks = [int(x) for x in args.global_topk.split(",")]
    mrs = [float(x) for x in args.merge_ratio.split(",")]
    tms = [float(x) for x in args.trim_mult.split(",")]
    pds = [float(x) for x in args.region_pad.split(",")]
    pps = [float(x) for x in args.pos_prior.split(",")]

    ds = get_dataset(args.dataset)
    train_layouts = ds.load_layout_items("train")
    test_layouts = ds.load_layout_items("test")
    train_names = sorted(train_layouts.keys())

    # EmbStore 与 local_tag 绑定（渲染超参固定），全部配置共享一份内存缓存
    base_cfg = LocalizationConfig()
    if args.gap is not None:
        base_cfg.wrap_gap_ratio = args.gap
    if args.no_wrap:
        base_cfg.use_wrap = False
    store = EmbStore(ds, base_cfg)

    grid = list(itertools.product(gks, lks, cds, mrs, tms, pds, pps))
    print(f"扫描 {len(grid)} 组配置 (tag={base_cfg.local_tag}) ...\n")
    print(f"{'gk':>3} {'lk':>3} {'cd':>5} {'mr':>4} {'tm':>4} {'pd':>5} {'pp':>5} | "
          f"{'top1':>7} {'r_cover':>7} {'area%':>6} | per-entity r_cover")

    rows = []
    for gk, lk, cd, mr, tm, pd, pp in grid:
        cfg = LocalizationConfig(global_topk=gk)
        cfg.local_topk = lk
        cfg.cluster_dist = cd
        cfg.merge_weight_ratio = mr
        cfg.trim_mult = tm
        cfg.region_pad = pd
        cfg.position_prior = pp
        t0 = time.time()
        _, report = run_localization(
            ds, cfg, store, test_layouts, train_names,
            train_layouts=train_layouts, n_test=args.n_test)
        summary = summarize_report(report)
        row = {
            "global_topk": gk, "local_topk": lk, "cluster_dist": cd,
            "merge_weight_ratio": mr, "trim_mult": tm, "region_pad": pd,
            "position_prior": pp,
            "macro_top1": macro_avg(summary, "top1_acc"),
            "macro_in_cluster": macro_avg(summary, "in_cluster_acc"),
            "macro_region_cover": macro_avg(summary, "region_cover_acc"),
            "macro_area_ratio": macro_avg(summary, "mean_area_ratio"),
            "macro_top3": macro_avg(summary, "top3_acc"),
            "macro_top5": macro_avg(summary, "top5_acc"),
            "per_entity": {e: {k: s[k] for k in
                               ("top1_acc", "in_cluster_acc", "region_cover_acc",
                                "mean_area_ratio", "top3_acc", "top5_acc")}
                           for e, s in summary.items()},
            "seconds": round(time.time() - t0, 2),
        }
        rows.append(row)
        ent_str = " ".join(f"{e[:3]}={s['region_cover_acc']:.0%}"
                           for e, s in sorted(summary.items()))
        print(f"{gk:>3} {lk:>3} {cd:>5.0f} {mr:>4.1f} {tm:>4.1f} {pd:>5.2f} {pp:>5.2f} | "
              f"{row['macro_top1']:>7.2%} {row['macro_region_cover']:>7.2%} "
              f"{row['macro_area_ratio']:>6.1%} | {ent_str}", flush=True)

    rows.sort(key=lambda r: r["macro_region_cover"], reverse=True)
    best = rows[0]
    print(f"\n最优 (macro region_cover): gk={best['global_topk']} "
          f"lk={best['local_topk']} cd={best['cluster_dist']} "
          f"mr={best['merge_weight_ratio']} tm={best['trim_mult']} "
          f"pd={best['region_pad']} "
          f"-> r_cover={best['macro_region_cover']:.2%} "
          f"top1={best['macro_top1']:.2%} area={best['macro_area_ratio']:.1%}")

    out_path = os.path.join(OUTPUT_DIR, ds.name, "sweeps",
                            f"sweep_localize_{base_cfg.local_tag}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"local_tag": base_cfg.local_tag, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"完整结果: {out_path}")


if __name__ == "__main__":
    main()
