"""离线定位（Stage 1 全局检索 + Stage 2 局部投票聚类）——纯 numpy，无需 GPU。

embedding 已由 scripts/embed.py 离线预计算，本脚本只做缓存读取与匹配。

用法:
  python scripts/embed.py --dataset sroie      # 先离线提取 embedding（仅一次）
  python scripts/localize.py --dataset sroie
  python scripts/localize.py --dataset sroie --cluster-dist 80 --local-topk 5
  # 记忆机制（两遍式: Pass1 纯训练库 -> 门控入库 -> Pass2 混合检索）:
  python scripts/localize.py --dataset sroie --memory
  python scripts/localize.py --dataset sroie --memory --gate-q 0.75 --mem-weight 0.7
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layoutsteer.config import LocalizationConfig, MemoryConfig, OUTPUT_DIR
from layoutsteer.datasets import get_dataset
from layoutsteer.localization.confidence import build_calibration, annotate_results
from layoutsteer.localization.memory import (
    build_memory_bank, MemoryBank, compare_passes)
from layoutsteer.localization.pipeline import (
    EmbStore, run_localization, summarize_report)
from layoutsteer.localization.key_match import KeyMatcher


def print_summary(summary, title):
    print(f"\n=== {title} ===")
    for entity, s in sorted(summary.items()):
        print(f"  {entity}: top1={s['top1_acc']:.2%} "
              f"in_cluster={s['in_cluster_acc']:.2%} "
              f"region_cover={s['region_cover_acc']:.2%} "
              f"top3={s['top3_acc']:.2%} top5={s['top5_acc']:.2%} (n={s['n']})")


def main():
    parser = argparse.ArgumentParser(description="LayoutSteer 离线定位（纯 numpy 匹配）")
    parser.add_argument("--dataset", type=str, default="sroie")
    parser.add_argument("--model", type=str, default="qwen25vl")
    parser.add_argument("--n-test", type=int, default=0, help="0 = 全部测试样本")
    parser.add_argument("--global-topk", type=int, default=None,
                        help="全局检索模板数（默认取 config）")
    parser.add_argument("--cluster-dist", type=float, default=None,
                        help="层次聚类距离阈值（默认取 config）")
    parser.add_argument("--local-topk", type=int, default=None,
                        help="每模板投票候选数（默认取 config）")
    parser.add_argument("--memory", action="store_true",
                        help="启用记忆机制（两遍式）")
    parser.add_argument("--online-bank", type=str, default="",
                        help="外部在线记忆库 json（三重门控产物），单遍混合检索")
    parser.add_argument("--out-tag", type=str, default="",
                        help="输出文件后缀（多轮实验不覆盖主文件）")
    parser.add_argument("--train-n", type=int, default=0,
                        help="训练模板库削减到 N 个样本（固定种子采样，"
                             "模板稀缺场景实验；0=全量）")
    parser.add_argument("--bootstrap", action="store_true",
                        help="用自举模板库（注意力 anchor 零标注）替代 GT 模板")
    parser.add_argument("--gate-q", type=float, default=None,
                        help="入库门控 best_weight 分位（默认取 config）")
    parser.add_argument("--gate-single-region", action="store_true",
                        help="入库附加要求单区域框")
    parser.add_argument("--mem-weight", type=float, default=None,
                        help="伪模板投票降权系数（默认取 config）")
    parser.add_argument("--mem-max-k", type=int, default=None,
                        help="Stage1 记忆模板检索上限（默认取 config）")
    parser.add_argument("--key-tau", type=float, default=0.0,
                        help="软键匹配阈值: 键从精确同键松弛为语义相近键"
                             "（开放词表如 funsd 用），0=关闭")
    args = parser.parse_args()
    _msfx = "" if args.model == "qwen25vl" else f"_{args.model}"

    loc_cfg = LocalizationConfig()
    if args.global_topk is not None:
        loc_cfg.global_topk = args.global_topk
    if args.cluster_dist is not None:
        loc_cfg.cluster_dist = args.cluster_dist
    if args.local_topk is not None:
        loc_cfg.local_topk = args.local_topk

    mem_cfg = MemoryConfig()
    if args.gate_q is not None:
        mem_cfg.gate_weight_quantile = args.gate_q
    if args.gate_single_region:
        mem_cfg.gate_single_region = True
    if args.mem_weight is not None:
        mem_cfg.mem_weight = args.mem_weight
    if args.mem_max_k is not None:
        mem_cfg.mem_max_k = args.mem_max_k

    ds = get_dataset(args.dataset)
    out_dir = os.path.join(OUTPUT_DIR, ds.name)
    os.makedirs(out_dir, exist_ok=True)

    print("加载布局数据 ...")
    train_layouts = ds.load_layout_items("train")
    test_layouts = ds.load_layout_items("test")
    train_names = sorted(train_layouts.keys())
    if args.train_n > 0:
        import numpy as np
        rng = np.random.RandomState(0)
        train_names = sorted(rng.choice(train_names, args.train_n,
                                        replace=False).tolist())
        print(f"模板稀缺模式: 训练库削减至 {len(train_names)} 样本")
    print(f"训练模板池: {len(train_names)}, 测试样本: {len(test_layouts)}")
    print(f"超参: global_topk={loc_cfg.global_topk} local_topk={loc_cfg.local_topk} "
          f"cluster_dist={loc_cfg.cluster_dist} tag={loc_cfg.local_tag}")

    store = EmbStore(ds, loc_cfg, model=args.model)
    bootstrap_meta = None
    if args.bootstrap:
        from layoutsteer.config import CACHE_DIR
        bs_dir = os.path.join(
            CACHE_DIR, f"{ds.name}_bootstrap_template_{loc_cfg.local_tag}{_msfx}")
        meta_path = os.path.join(OUTPUT_DIR, ds.name,
                                 f"bootstrap{_msfx}", "bootstrap_meta.json")
        bootstrap_meta = json.load(open(meta_path))
        store = EmbStore(ds, loc_cfg, template_dir=bs_dir, model=args.model)
        print(f"自举模板库: {bs_dir} ({len(bootstrap_meta)} 条)")
    key_matcher = None
    if args.key_tau > 0:
        key_matcher = KeyMatcher(ds.entity_types, tau=args.key_tau)
        print(f"软键匹配: tau={args.key_tau} 键池={len(ds.entity_types)}")
    t0 = time.time()
    if args.online_bank:
        # 单遍混合检索: (自举或GT)模板库 ∪ 外部在线记忆
        bank = MemoryBank(json.load(open(args.online_bank)))
        print(f"在线记忆库: {len(bank)} 条 {bank.stats()}")
        results, report = run_localization(
            ds, loc_cfg, store, test_layouts, train_names,
            train_layouts=train_layouts, n_test=args.n_test, verbose=True,
            memory_bank=bank, mem_cfg=mem_cfg, bootstrap_meta=bootstrap_meta,
            key_matcher=key_matcher)
        summary = summarize_report(report)
        print_summary(summary, "train ∪ online memory")
    elif args.memory:
        # Pass 1: 纯训练库定位
        results, report = run_localization(
            ds, loc_cfg, store, test_layouts, train_names,
            train_layouts=train_layouts, n_test=args.n_test, verbose=True,
            key_matcher=key_matcher)
        summary = summarize_report(report)
        print_summary(summary, "Pass 1 (纯训练库)")
        # 门控入库（仅 Pass 1 结果，单轮，防级联污染）
        bank_dict = build_memory_bank(report, mem_cfg, ds.entity_types)
        bank = MemoryBank(bank_dict)
        print(f"\n记忆库: {len(bank)} 条 {bank.stats()} "
              f"(gate_q={mem_cfg.gate_weight_quantile} "
              f"single_region={mem_cfg.gate_single_region})")
        bank_path = os.path.join(out_dir, "memory_bank.json")
        with open(bank_path, "w", encoding="utf-8") as f:
            json.dump(bank_dict, f, ensure_ascii=False, indent=2)

        # Pass 2: train ∪ memory 混合检索
        report1 = report
        results, report = run_localization(
            ds, loc_cfg, store, test_layouts, train_names,
            train_layouts=train_layouts, n_test=args.n_test, verbose=True,
            memory_bank=bank, mem_cfg=mem_cfg, key_matcher=key_matcher)
        summary = summarize_report(report)
        print_summary(summary, "Pass 2 (train ∪ memory)")

        diff = compare_passes(report1, report)
        print("\n=== 修复/退化分解 (Pass1 -> Pass2) ===")
        for ent, d in sorted(diff.items()):
            print(f"  {ent}: top1 +{d['top1_fixed']}/-{d['top1_regressed']} "
                  f"cover +{d['cover_fixed']}/-{d['cover_regressed']} (n={d['n']})")
        with open(os.path.join(out_dir, "memory_comparison.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"mem_config": vars(mem_cfg),
                       "pass1_summary": summarize_report(report1),
                       "pass2_summary": summary, "diff": diff},
                      f, ensure_ascii=False, indent=2)
    else:
        results, report = run_localization(
            ds, loc_cfg, store, test_layouts, train_names,
            train_layouts=train_layouts, n_test=args.n_test, verbose=True,
            bootstrap_meta=bootstrap_meta, key_matcher=key_matcher)
        summary = summarize_report(report)
        print_summary(summary, "定位质量汇总")

    if args.out_tag:
        base = ds.retrieval_bbox_path().replace(".json", f"{args.out_tag}.json")
        bbox_out_path = base
        report_path = os.path.join(out_dir,
                                   f"localization_report{args.out_tag}.json")
    else:
        bbox_out_path = ds.retrieval_bbox_path()
        report_path = os.path.join(out_dir, "localization_report.json")
    # 置信度标定（best_weight 分位 -> 区域覆盖率）并注入定位结果，
    # 供推理侧 δ_eff = δ·c^γ 加权使用
    calib = build_calibration(report)
    annotate_results(results, report, calib)
    calib_path = os.path.join(out_dir, "confidence_calibration.json")
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print(f"置信度标定表: {calib_path}")
    with open(bbox_out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"config": {"global_topk": loc_cfg.global_topk,
                              "local_topk": loc_cfg.local_topk,
                              "cluster_dist": loc_cfg.cluster_dist,
                              "local_tag": loc_cfg.local_tag,
                              "memory": vars(mem_cfg) if args.memory else None},
                   "summary": summary, "per_sample": report},
                  f, ensure_ascii=False, indent=2)

    print(f"\n定位完成: {len(report)} 个样本, 耗时 {time.time() - t0:.1f}s")
    print(f"bbox 结果: {bbox_out_path}")
    print(f"定位报告: {report_path}")


if __name__ == "__main__":
    main()
