# LayoutSteer

**Self-Bootstrapping Layout Steering for Document Visual Information Extraction**

LayoutSteer is a plug-and-play, **training-free** framework that steers a frozen
vision-language model (VLM) with a **layout prior** at inference time. It needs
neither parameter updates nor OCR, and it accumulates extraction experience as
documents stream in.

Given a document image and a query, LayoutSteer

1. **Layout Mask Generation** — encodes the document purely by its spatial
   layout (text-line boxes rasterized into a binary layout image, embedded by the
   model's own frozen vision encoder), retrieves layout-similar experience from
   memory, votes over candidate boxes, and turns the matched region into a
   *graded* layout saliency mask;
2. **Layout-Guided Inference** — injects the mask into the image tokens' **key
   representations**: a shared perturbation direction `Δk = δ√d·Q⁺1` shifts the
   pre-softmax score of token `i` by exactly `δ·w_i`, so softmax reallocates
   attention in proportion to the mask;
3. **Self-Bootstrapping Memory** — grows from the model's **own** high-confidence
   inferences (confident answer *and* unchanged under intervention), storing the
   local layout around each attention peak under its field key. The memory starts
   empty; documents whose field is not yet covered fall back to the base
   prediction instead of being misled by a wrong prior.

The result: off-the-shelf VLMs of different architectures and scales improve on
four VIE benchmarks, with the frozen model and no extra network.

## Installation

```bash
pip install -r requirements.txt
```

`flash-attn` can be slow to build; prebuilt wheels are available at
<https://github.com/Dao-AILab/flash-attention/releases>.

Models are resolved from `LAYOUTSTEER_MODEL_ROOT` (default `./models`) and
datasets from `LAYOUTSTEER_DATASET_ROOT` (default `./dataset`); both may be
overridden by environment variables:

```bash
export LAYOUTSTEER_MODEL_ROOT=/path/to/models
export LAYOUTSTEER_DATASET_ROOT=/path/to/dataset
```

## Model Preparation

Download the backbone(s) you need under `LAYOUTSTEER_MODEL_ROOT` using the
directory names below (registered in `layoutsteer/config.py`, `MODEL_PRESETS`):

| Registry name | Directory | Role in the paper |
|---|---|---|
| `qwen3vl` | `Qwen3-VL-8B-Instruct` | primary backbone |
| `qwen25vl` | `Qwen2.5-VL-7B-Instruct` | generic VLM (also the localizer) |
| `llavaov15` | `LLaVA-OneVision-1.5-8B-Instruct` | generic VLM |
| `internvl35` | `InternVL3_5-8B` | generic VLM |
| `deepseekocr` | `DeepSeek-OCR` | specialized OCR-free |
| `docowl2` | `DocOwl2` | specialized OCR-free |
| `doclayllm` | `DocLayLLM_sft` | OCR-based |

Layout localization is backbone-agnostic: it always reuses the layout products of
Qwen2.5-VL, whose boxes are in original-image pixel coordinates.

## Dataset Preparation

We evaluate on four standard VIE benchmarks — **SROIE**, **CORD**, **FUNSD** and
**POIE** — on their official test splits. Each dataset is subject to its own
license; please download it from the official source. Since the memory is
bootstrapped from the processed documents, no training split and no field
annotation is required.

Place them under the unified convention:

```
dataset/{sroie,cord,funsd,poie}/
├── train/                      # only needed for template-retrieval / warmup routes
│   ├── images/
│   └── answer_bboxes.json
└── test/
    ├── images/
    ├── answer_bboxes.json
    └── qa_test.json
```

`answer_bboxes.json` is a list of per-question records:

```json
[
  {
    "sample_name": "X51005230605",
    "question": "What is the total amount?",
    "question_type": "total",
    "answer": "25.00",
    "matching_bboxes": [{"box": [211, 368, 405, 386]}],
    "all_gt_items": [
      {"text": "TOTAL 25.00", "box": [211, 368, 405, 386], "entity": "total"}
    ]
  }
]
```

* `box` is `[x1, y1, x2, y2]` in the pixel coordinates of `images/`.
* `all_gt_items` lists the page's layout elements (text lines with an entity
  label); it is what the retrieval indexes as layout.
* `matching_bboxes` is the answer region used as the intervention target, and it
  is ground truth — used **for evaluation only** once the geometric/automatic
  routes are in play.
* Entity-code datasets (CORD, POIE) should add `target_entity`
  (e.g. `"TOTAL.TOTAL_PRICE"`); for the open-vocabulary dataset (FUNSD),
  `question_type` is the normalized field name used as the retrieval key.

`qa_test.json` is a list of `{"question", "answer", "metadata": {"sample_name": ...}}`.

## Quickstart

```bash
# 0. Smoke test: 3 samples, verifies the whole intervention chain
python scripts/smoke_test.py
```

### 1. Layout mask generation

```bash
# Layout embeddings (global document layout + local neighbourhoods), cached once
python scripts/embed.py --dataset sroie

# Stage 1 global retrieval + Stage 2 voting/clustering -> localized_bboxes_sroie.json
python scripts/localize.py --dataset sroie
```

`localize.py` is pure numpy and runs in seconds once the embeddings are cached;
`--global-topk` (K experiences per query) and `--local-topk` (J votes per
experience) expose the two retrieval hyper-parameters, defaulting to K=5, J=2.

### 2. Layout-guided inference

```bash
# Ground-truth boxes: upper-bound reference for the intervention itself
python scripts/run_infer.py --dataset sroie --bbox-source gt --delta 2.0 --layers 19-27

# Localized boxes: the fully automatic setting
python scripts/run_infer.py --dataset sroie --bbox-source retrieval --delta 2.0 --layers 19-27
```

Each run writes `normal_results.json`, `intervened_results.json` and
`evaluation_results.json` into a fresh timestamped directory under
`output/{dataset}/runs/`, so runs never overwrite each other. Useful flags:
`--shard i/n` (multi-GPU), `--resume DIR`, `--normal-from FILE` (reuse the
baseline half of a sweep).

The intervention is applied to late decoder layers only. `delta` is tuned per
dataset by grid search:

```bash
python scripts/sweep_delta.py  --dataset sroie --bbox-source gt \
    --deltas 0.5,1,2,3,5 --layers 19-27 --normal-from <normal_results.json>
python scripts/sweep_layers.py --dataset sroie --bbox-source gt --delta 2 \
    --layer-specs "all|0-9|10-18|19-27|22" --normal-from <normal_results.json>
```

### 3. Self-bootstrapping memory (no annotation)

The memory starts empty and grows online from the test stream:

```bash
# Attention peaks of the model's own predictions -> pseudo field positions
python scripts/bootstrap_anchors.py --dataset sroie --split test --shard 0/3   # then --merge

# Stream the test set: intervene only when a field is already covered,
# otherwise fall back to the base prediction; accepted documents grow the memory
python scripts/run_coldstart.py --dataset sroie --seed 0 --live-dk --anchor-k 3
```

`--live-dk` derives the perturbation direction from the current sample's own
query via `Δk = δ√d·Q⁺1`, exactly as in the paper; no precomputed bank is needed.
Small test sets should be run with several `--seed` values.

### 4. Geometric (OCR-free) text-line boxes

The text-line boxes `B` used as the layout representation come from a purely
geometric morphological procedure (Otsu binarization → horizontal dilation to
join a line → connected components, with noise/vertical-rule filtering), so
neither OCR output nor field annotations are used:

```bash
# Inspect the pseudo text-line boxes against GT boxes
python scripts/visualize_ocrfree.py --n 4

# Localization on both sides with morphological boxes (recommended: symmetric)
python scripts/run_ocrfree_localize.py  --dataset sroie --embed-only   # test-side embeddings
python scripts/run_ocrfree_symmetric.py --dataset sroie                # -> localized_bboxes_sroie_morphsym.json

# End-to-end with a graded mask over the morphological layout
python scripts/run_mask.py --dataset sroie --mask-mode structured --core-gap 1.0 \
    --delta 2.0 --layers 19-27 --layout-source morph \
    --bbox-json output/sroie/localized_bboxes_sroie_morphsym.json --out-tag morphsym

# Memory bootstrapped from morphological boxes only (fully zero-annotation, zero-OCR)
python scripts/bootstrap_anchors.py     --dataset sroie --split train --morph --shard 0/3  # then --merge
python scripts/prep_morph_bootstrap.py  --dataset sroie --strict --shard 0/3               # then --merge
python scripts/run_coldstart.py         --dataset sroie --morph --strict --live-dk
```

## Verifying the Intervention

`scripts/verify_delta_injection.py` asserts that the pre-softmax score of every
targeted image token changes by exactly `δ` (± tolerance) while non-target tokens
receive zero leakage. Run it before adapting a new backbone:

```bash
python scripts/verify_delta_injection.py --model qwen3vl --dataset sroie \
    --delta 5.0 --layer 22 --n-tasks 5
```

## Repository Layout

```
layoutsteer/
├── config.py              all paths / hyper-parameters; MODEL_PRESETS
├── intervention/          ScoreDeltaProbe + ΔK hooks + bbox→token grid mapping
├── localization/          layout embedding, global retrieval, local voting/clustering
├── adapters/              per-model load / input / token mapping / hook tier
├── datasets/              unified KIE dataset interface (sroie / standard)
├── evaluation/            metrics (LayTextLLM-compatible)
├── model_loader.py        Qwen2.5-VL loading + input construction
├── runner.py              dual-inference main loop (normal vs. steered)
└── visualization.py       attention heat-map overlays
scripts/                   pipeline entry points (see Quickstart)
dataset/                   datasets, in the convention above
```

## Main Results

Reported in the paper. Accuracy (%) for OCR-free models, averaged over the
datasets each model is evaluated on; F1 (%) for the OCR-based model.

| Model | base | + LayoutSteer |
|---|---|---|
| Qwen2.5-VL | 70.6 | **75.0** |
| LLaVA-OV-1.5 | 63.4 | **66.2** |
| InternVL3 | 66.5 | **68.0** |
| Qwen3-VL | 79.9 | **82.7** |
| DeepSeek-OCR | 45.9 | **50.5** |
| DocOwl2 | 52.1 | **55.3** |
| DocLayLLM (F1) | 81.5 | **87.1** |

Two observations from the ablation: the mask is only useful when it is both
*layout-aligned* and *graded* (a random region or a uniform global boost even
falls below the base model), and the fallback matters as much as the prior —
documents whose layout is not yet covered are left untouched.

## Citation

```bibtex
@misc{layoutsteer,
  title  = {LayoutSteer: Self-Bootstrapping Layout Steering for
            Document Visual Information Extraction},
  author = {Qian, Wentao and Zheng, Xiaohan and Zhuang, Liansheng},
  note   = {Manuscript under review},
  year   = {2026}
}
```

## License

Released under the Apache License 2.0 — see [LICENSE](LICENSE).

Third-party model code under `layoutsteer/adapters/vendor/` (DocOwl2, DocLayLLM,
DeepSeek-OCR) is redistributed under its original Apache-2.0 headers, which are
retained in the respective files. The datasets are not redistributed.
