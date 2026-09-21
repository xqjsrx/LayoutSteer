"""推理入口：正常 vs score_delta 干预双推理。

每次运行输出到带时间戳的新目录 output/{dataset}/{tag}_{YYYYmmdd_HHMMSS}/；
续跑历史结果用 --resume 指定已有目录。

用法:
  conda activate qwt-layoutsteer
  python scripts/run_infer.py --dataset sroie --bbox-source gt --delta 5.0
  python scripts/run_infer.py --dataset sroie --bbox-source retrieval --delta 5.0 --layers all
  python scripts/run_infer.py --dataset sroie --bbox-source gt --delta 2.0 --n-tasks 20 --visualize
  python scripts/run_infer.py --dataset sroie --resume output/sroie/gt_delta5_layers-all_20260721_150000
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layoutsteer.config import RunConfig, InterventionConfig, make_model_config
from layoutsteer.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="LayoutSteer 干预推理")
    parser.add_argument("--dataset", type=str, default="sroie")
    parser.add_argument("--model", type=str, default="qwen25vl",
                        help="模型适配器名（config.MODEL_PRESETS）")
    parser.add_argument("--bbox-source", type=str, default="gt",
                        choices=["gt", "retrieval"])
    parser.add_argument("--delta", type=float, default=5.0,
                        help="score_delta 强度 δ（>0 增强目标区域注意力）")
    parser.add_argument("--layers", type=str, default="all",
                        help='干预层: "all" 或逗号分隔层索引，如 "22" / "20,21,22"')
    parser.add_argument("--n-tasks", type=int, default=0, help="0 = 全部任务")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--visualize", action="store_true",
                        help="保存注意力热图对比（较慢）")
    parser.add_argument("--resume", type=str, default="",
                        help="续跑已有输出目录（不新建时间戳目录）")
    parser.add_argument("--shard", type=str, default="",
                        help='任务分片 "i/n"（多卡并行，各分片需用 --resume 指定独立目录）')
    parser.add_argument("--conf-weight", action="store_true",
                        help="启用定位置信度加权 δ_eff = δ·c^γ（需定位结果含 confidence）")
    parser.add_argument("--conf-gamma", type=float, default=1.0,
                        help="置信度锐度 γ")
    parser.add_argument("--conf-cutoff", type=float, default=0.0,
                        help="c 低于此值时关断干预")
    parser.add_argument("--normal-from", type=str, default="",
                        help="从已有 normal_results.json 复用基线（扫参省一半生成）")
    parser.add_argument("--persistent", action="store_true",
                        help="一次性 KV cache 编辑模式（decode 零干预）")
    parser.add_argument("--bbox-json", type=str, default="",
                        help="指定定位产物路径（默认 localized_bboxes_{ds}.json）")
    parser.add_argument("--run-tag", type=str, default="",
                        help="输出目录附加标记（区分不同定位源的实验）")
    args = parser.parse_args()

    if args.resume and not os.path.isdir(args.resume):
        parser.error(f"--resume 目录不存在: {args.resume}")

    cfg = RunConfig(
        dataset=args.dataset,
        bbox_source=args.bbox_source,
        max_new_tokens=args.max_new_tokens,
        n_tasks=args.n_tasks,
        visualize=args.visualize,
        resume_dir=args.resume,
        shard=args.shard,
        normal_from=args.normal_from,
        bbox_json=args.bbox_json,
        run_tag=args.run_tag,
        model=make_model_config(args.model),
        intervention=InterventionConfig(
            delta=args.delta, target_layers=args.layers,
            use_confidence=args.conf_weight,
            conf_gamma=args.conf_gamma, conf_cutoff=args.conf_cutoff,
            persistent=args.persistent),
    )
    print(f"输出目录: {cfg.output_dir}")
    ExperimentRunner(cfg).run()


if __name__ == "__main__":
    main()
