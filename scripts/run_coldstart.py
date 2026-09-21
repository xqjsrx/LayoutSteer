"""任务C: 空库冷启动流式自举——系统从零开始边跑边长经验。

设定: 无训练集、无标注、库=∅。测试文档按随机顺序流式到达:
  1. 用当前记忆池检索定位（池内某实体无条目 -> 本任务不干预，
     直接采用预采集的 normal 预测）
  2. 有定位 -> structured 掩码 + 池内邻居 dK 干预（δ=2 固定，
     冷启动无标定表，不用置信度加权）
  3. 入库门控: G1 路径一致（干预与 normal 预测相同; 未干预任务仅 G2）
     + G2 normal logprob >= -0.05 -> 注意力主峰 anchor 入池
  资产全部复用: anchors_test.json(normal pred/lp/anchor)、
  test_global/test_local embedding 缓存、dk_bank_test。

产出: 每任务 {顺序, 是否干预, 正确性, 池大小} -> 生长曲线。

用法: CUDA_VISIBLE_DEVICES=3 python scripts/run_coldstart.py --seed 0
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from layoutsteer.config import RunConfig, LocalizationConfig, \
    parse_target_layers, bootstrap_dir, qtemplate_dir, morph_tag, \
    make_model_config
from layoutsteer.datasets import get_dataset
from layoutsteer.evaluation.metrics import is_correct_prediction
from layoutsteer.model_loader import (
    setup_seeds, load_model_and_processor, resize_image_by_pixel_limit,
    build_inputs)
from layoutsteer.intervention import (
    ScoreDeltaProbe, structured_mask_grid, grid_to_indices_weights)
from layoutsteer.adapters import get_adapter
from layoutsteer.localization.pipeline import EmbStore, build_regions, \
    union_box, _norm_y
from layoutsteer.localization.local_match import vote_and_cluster


def cache_keys(pkv, layer):
    if hasattr(pkv, "layers"):
        return pkv.layers[layer].keys
    return pkv.key_cache[layer]


def intervene(model, processor, inputs, dk_map, target_layers, seq_idx,
              weights, delta, max_new_tokens=128, live_dk=False):
    """zero-hook: 裸 prefill 前 L-1 -> 编辑 -> generate。

    live_dk=True: prefill 时 QRecorder 捕获末位置 q 现场伪逆解 dK
    （dk_map 参数被忽略, 完全不依赖 dk_bank）。
    """
    with torch.no_grad():
        if live_dk:
            # 完整 prefill 捕获真末位 q（首个生成步的 query）现解 dK,
            # 再裁剪 cache 至 L-1 回到裸 prefill 状态
            from build_dk_bank import QRecorder, solve_dk_all
            with QRecorder(set(target_layers)) as rec:
                out = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            pixel_values=inputs["pixel_values"],
                            image_grid_thw=inputs["image_grid_thw"],
                            use_cache=True)
            # KV 头数按模型取（Qwen2.5-VL=4, Qwen3-VL=8），勿写死
            _tc = getattr(model.config, "text_config", model.config)
            _n_kv = getattr(_tc, "num_key_value_heads", 4)
            dk_map = {l: v.float() for l, v in
                      solve_dk_all(rec.q_last, num_kv=_n_kv).items()}
            out.past_key_values.crop(inputs["input_ids"].shape[1] - 1)
        else:
            out = model(input_ids=inputs["input_ids"][:, :-1],
                        attention_mask=inputs["attention_mask"][:, :-1],
                        pixel_values=inputs["pixel_values"],
                        image_grid_thw=inputs["image_grid_thw"], use_cache=True)
        pkv = out.past_key_values
        for layer in target_layers:
            keys = cache_keys(pkv, layer)
            dk = dk_map[layer].to(keys.device)
            w = weights.to(keys.device, torch.float32)
            for h in range(keys.shape[1]):
                keys[0, h, seq_idx, :] += (
                    delta * w.unsqueeze(1) * dk[h].unsqueeze(0)).to(keys.dtype)
        gen = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            past_key_values=pkv, max_new_tokens=max_new_tokens,
            do_sample=False)
    pred = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()
    del out, pkv, gen
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sroie")
    parser.add_argument("--model", type=str, default="qwen25vl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", action="store_true",
                        help="统一文档流: 训练文档作热身段先入池（流式与批式"
                             "等价, 直接初始化）, 测试段从此池继续流式")
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--layers", type=str, default="19-27")
    parser.add_argument("--lp-thr", type=float, default=-0.05)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--anchor-k", type=int, default=3,
                        help="入池 anchor 峰数（3=与训练自举段一致，1=旧单峰）")
    parser.add_argument("--dilate-ratio", type=float, default=0.04,
                        help="morph 提框膨胀比（须与 embedding/anchors 阶段一致）")
    parser.add_argument("--morph", action="store_true",
                        help="OCR-free: 全部布局资产用形态学框（零标注零OCR）")
    parser.add_argument("--strict", action="store_true",
                        help="配合 --morph: 用严格版 anchors（注意力峰直接对齐 "
                             "morph 框, 无 GT 几何中转）")
    parser.add_argument("--own-dk", action="store_true",
                        help="dK 用当前样本自己的 q 求解（bank_test 预采集, "
                             "同一实体 prompt 末位置, 等价流式 prefill 现解）, "
                             "而非池内邻居 dK 均值")
    parser.add_argument("--live-dk", action="store_true",
                        help="dK 在裸 prefill 时现场从当前 q 伪逆求解, "
                             "完全不加载 dk_bank（最终简化架构）")
    args = parser.parse_args()

    cfg = RunConfig(model=make_model_config(args.model))
    loc_cfg = LocalizationConfig()
    setup_seeds()
    ds = get_dataset(args.dataset)
    _msfx = "" if args.model == "qwen25vl" else f"_{args.model}"
    boot_dir = bootstrap_dir(ds.name, args.model)
    anchors_test_path = os.path.join(boot_dir, "anchors_test.json")
    anchors_train_path = os.path.join(boot_dir, "anchors.json")
    bank_test_path = os.path.join(qtemplate_dir(ds.name, args.model),
                                  "dk_bank_test.pt")
    bank_train_path = os.path.join(qtemplate_dir(ds.name, args.model),
                                   "dk_bank_train.pt")
    store = EmbStore(ds, loc_cfg, model=args.model)
    if args.morph:
        # OCR-free: 布局/embedding/anchors 全部换 morph 资产（dK bank 与框无关）
        from layoutsteer.config import CACHE_DIR
        mtag = morph_tag(args.dilate_ratio)
        anchors_suffix = "_morphstrict.json" if args.strict else "_morph.json"
        store.dirs = dict(
            store.dirs,
            test_global=os.path.join(CACHE_DIR, f"{ds.name}_test_global_{mtag}"),
            test_local=os.path.join(
                CACHE_DIR, f"{ds.name}_test_local_{loc_cfg.local_tag}_{mtag}"),
            train_global=os.path.join(
                CACHE_DIR, f"{ds.name}_train_global_{mtag}"))
        anchors = json.load(open(
            anchors_test_path.replace(".json", anchors_suffix)))
        from run_ocrfree_localize import build_morph_layouts
        gt_layouts = ds.load_layout_items("test")
        layouts = build_morph_layouts(ds, sorted(gt_layouts.keys()),
                                      gt_layouts, args.dilate_ratio)
    else:
        anchors = json.load(open(anchors_test_path))
        layouts = ds.load_layout_items("test")
    bank_test = {} if args.live_dk else torch.load(bank_test_path)
    qa = {(t.sample_name, t.task_type): t for t in ds.iter_tasks()}

    names = sorted(layouts.keys())
    rng = np.random.RandomState(args.seed)
    stream = [names[i] for i in rng.permutation(len(names))]

    tag = "warm_" if args.warmup else ""
    if args.anchor_k != 1:
        tag += f"a{args.anchor_k}_"
    if args.morph:
        tag += "morphstrict_" if args.strict else "morph_"
    if args.own_dk:
        tag += "owndk_"
    if args.live_dk:
        tag += "livedk_"
    out_path = os.path.join(boot_dir, f"coldstart_{tag}seed{args.seed}.json")
    adapter = get_adapter(cfg.model)
    adapter.load()
    model, processor = adapter.model, adapter.processor
    probe = adapter.create_probe(capture_layer=cfg.intervention.capture_layer)
    target_layers = parse_target_layers(args.layers, probe.n_layers)

    # 记忆池: (source, entity) -> {embs: [K,D], ys: [K]}，检索键 pool_gmat
    pool_entry = {}
    pool_sources, pool_gmat = [], []
    bank_dk = (dict(torch.load(bank_train_path))
               if args.warmup and not args.live_dk else {})
    bank_dk.update(bank_test)

    if args.warmup:
        # 热身段: 训练文档流式入池≡批式（入库门控 logprob-only，
        # 与库状态无关），直接用自举模板缓存初始化
        from layoutsteer.config import CACHE_DIR
        if args.morph:
            mt = morph_tag(args.dilate_ratio)
            bs_suffix = ("_" + mt.replace("morph", "morphstrict")
                         if args.strict else "_" + mt)
        else:
            bs_suffix = ""
        bs_dir = os.path.join(
            CACHE_DIR,
            f"{ds.name}_bootstrap_template_{loc_cfg.local_tag}{bs_suffix}{_msfx}")
        anchors_train = json.load(open(
            anchors_train_path.replace(".json", anchors_suffix) if args.morph
            else anchors_train_path))
        tg_dir = store.dirs["train_global"]
        n_warm = 0
        for tname, ents in anchors_train.items():
            added = False
            for ent, a in ents.items():
                if a["mean_logprob"] < args.lp_thr:
                    continue
                p = os.path.join(bs_dir, f"{tname}__{ent}.npy")
                if not os.path.exists(p):
                    continue
                embs = np.load(p)
                if embs.size == 0:
                    continue
                pool_entry[(tname, ent)] = {
                    "embs": embs, "ys": a["anchor_ys"][:embs.shape[0]]}
                added = True
                n_warm += 1
            if added:
                pool_sources.append(tname)
                pool_gmat.append(np.load(os.path.join(tg_dir, f"{tname}.npy")))
        print(f"热身池: {len(pool_sources)} 文档 / {n_warm} 条", flush=True)

    results = []

    for oi, name in enumerate(stream):
        info = layouts[name]
        boxes = [it.box for it in info["items"]]
        extent = union_box(boxes)
        candidate_ys = np.array([_norm_y(b, extent) for b in boxes])
        test_emb = store.test_global(name)
        cand_local = store.test_local(name)
        order = (np.argsort(np.stack(pool_gmat) @ test_emb)[::-1]
                 if pool_sources else [])

        raw_size = None
        for ent in ds.sample_entities(name):
            a = anchors.get(name, {}).get(ent)
            task = qa.get((name, ent))
            if a is None or task is None:
                continue
            # 沿相似度凑 topk 个有该实体条目的池内源
            tpls, ys, nbs = [], [], []
            for pi in order:
                src = pool_sources[pi]
                e = pool_entry.get((src, ent))
                if e is None:
                    continue
                for k in range(len(e["ys"])):
                    tpls.append(np.asarray(e["embs"][k]))
                    ys.append(e["ys"][k])
                nbs.append(src)
                if len(nbs) >= args.topk:
                    break

            if tpls:
                votes, clusters = vote_and_cluster(
                    np.stack(tpls), cand_local, boxes, loc_cfg,
                    template_anchor_ys=np.array(ys),
                    candidate_ys=candidate_ys)
            else:
                clusters = []
            if clusters:
                regions = build_regions(clusters, boxes, extent, loc_cfg)
                region_boxes = [r for r, _ in regions]
                member_set = {m for _, ms in regions for m in ms}
                max_w = max(votes[0]["total_weight"], 1e-6)
                wmap = [{"box": [int(v) for v in boxes[vt["bbox_idx"]]],
                         "weight": round(vt["total_weight"] / max_w, 4),
                         "in_region": vt["bbox_idx"] in member_set}
                        for vt in votes]
                # dK: 现解（--live-dk）/ 当前样本预采集（--own-dk）/ 邻居均值
                if args.live_dk:
                    dk_map = "live"      # 哨兵: intervene 内 prefill 时现解
                elif args.own_dk:
                    e_own = bank_test.get(name, {}).get(ent)
                    dk_map = ({l: e_own[l].float() for l in target_layers}
                              if e_own is not None else None)
                else:
                    per_layer = {l: [] for l in target_layers}
                    for nb in nbs:
                        e2 = bank_dk.get(nb, {}).get(ent)
                        if e2 is None:
                            continue
                        for l in target_layers:
                            per_layer[l].append(e2[l].float())
                    dk_map = ({l: torch.stack(v).mean(0)
                               for l, v in per_layer.items()}
                              if per_layer[next(iter(target_layers))] else None)
            else:
                dk_map = None

            if dk_map is not None:
                if raw_size is None:
                    raw = Image.open(task.image_path).convert("RGB")
                    raw_size = raw.size
                    image = resize_image_by_pixel_limit(
                        raw, cfg.model.max_image_pixels)
                inputs = build_inputs(processor, image, ds.build_prompt(task),
                                      model.device)
                probe.setup_image_range(inputs["input_ids"],
                                        inputs["image_grid_thw"])
                grid = structured_mask_grid(
                    region_boxes, wmap, boxes, raw_size, probe.spatial_shape)
                grid_idx, weights = grid_to_indices_weights(grid)
                seq_idx = (probe._img_range[0] + grid_idx).to(model.device)
                pred = intervene(model, processor, inputs, dk_map,
                                 target_layers, seq_idx, weights, args.delta,
                                 live_dk=args.live_dk)
                intervened = True
                del inputs
            else:
                pred = a["pred"]          # 复用预采集 normal 预测（同 greedy）
                intervened = False

            correct = is_correct_prediction(pred, task.answer)
            results.append({
                "order": oi, "sample_name": name, "task": ent,
                "prediction": pred, "is_correct": bool(correct),
                "intervened": intervened,
                "pool_entries": len(pool_entry),
            })
            # 入库门控: G1（有干预时）+ G2；anchor 取 top3 峰，
            # 与训练自举段资产质量一致（单峰精度仅~69%，top3 达 94%，
            # 噪声峰靠跨样本投票消化）
            g1 = (not intervened) or (pred.strip() == a["pred"].strip())
            if g1 and a["mean_logprob"] >= args.lp_thr:
                idxs = a["anchor_idxs"][:args.anchor_k]
                pool_entry[(name, ent)] = {
                    "embs": np.stack([cand_local[int(i)] for i in idxs]),
                    "ys": [float(y) for y in a["anchor_ys"][:len(idxs)]]}

        # 任一实体入库则挂全局检索键
        if any((name, e) in pool_entry for e in ds.entity_types):
            pool_sources.append(name)
            pool_gmat.append(test_emb)

        if (oi + 1) % 20 == 0:
            json.dump(results, open(out_path + ".tmp", "w"),
                      ensure_ascii=False)
            os.replace(out_path + ".tmp", out_path)
            torch.cuda.empty_cache()
            n_int = sum(r["intervened"] for r in results)
            acc = np.mean([r["is_correct"] for r in results])
            print(f"[seed{args.seed}] {oi+1}/{len(stream)} 累积acc={acc:.2%} "
                  f"干预率={n_int/len(results):.0%} 池={len(pool_entry)}",
                  flush=True)

    json.dump(results, open(out_path + ".tmp", "w"), ensure_ascii=False)
    os.replace(out_path + ".tmp", out_path)
    acc = np.mean([r["is_correct"] for r in results])
    print(f"[seed{args.seed}] 完成: 总acc={acc:.2%} -> {out_path}")


if __name__ == "__main__":
    main()
