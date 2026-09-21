"""InternVL3.5 适配器（Tier A, LLM=原生 Qwen3ForCausalLM）。

已核实（remote code + README, tf 4.57.1）:
  - LLM 走 ALL_ATTENTION_FUNCTIONS 分发 -> ScoreDeltaProbe 原样可用；
    视觉塔私有 FlashAttention 不经分发，不受影响
  - 图像 token: 每瓦 256 token（16x16 行主序, 28px/格），瓦片行主序排列，
    多瓦时末尾追加整图缩略图；IMG_CONTEXT(151671) 单段连续，
    前后 <img>(151669) </img>(151670)
  - 瓦片网格 = find_closest_aspect_ratio（各向异性拉伸，无 padding）
  - 需自建 input_ids 驱动 model.generate（chat() 会内部建构造）；
    须先设 model.img_context_token_id
  - bbox->token: 绝对序列索引路径（probe.set_target_indices）
"""
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

from .base import ModelAdapter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START, IMG_END, IMG_CONTEXT = "<img>", "</img>", "<IMG_CONTEXT>"
TILE = 448          # force_image_size
CELL = 28           # 448 / 16: 每 token 覆盖的瓦内像素
GRID = 16           # 每瓦 16x16 token
TOK_PER_TILE = GRID * GRID


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    """逐行复刻官方 README（tie-break 分支必须一致）。"""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def pick_grid(image, min_num=1, max_num=12, image_size=TILE):
    """(cols, rows) 瓦片网格（与官方 dynamic_preprocess 一致）。"""
    w, h = image.size
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1])
    return find_closest_aspect_ratio(w / h, target_ratios, w, h, image_size)


def tile_images(image, grid, image_size=TILE, use_thumbnail=True):
    cols, rows = grid
    resized = image.resize((image_size * cols, image_size * rows))
    tiles = []
    for i in range(cols * rows):
        c, r = i % cols, i // cols
        tiles.append(resized.crop((c * image_size, r * image_size,
                                   (c + 1) * image_size, (r + 1) * image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


class InternVL35Adapter(ModelAdapter):
    name = "internvl35"
    tier = "A"

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.tokenizer = None
        self._transform = T.Compose([
            T.Lambda(lambda im: im.convert("RGB") if im.mode != "RGB" else im),
            T.Resize((TILE, TILE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def load(self):
        from transformers import AutoModel
        print(f"加载模型: {self.cfg.model_path} (use_flash_attn, remote code)")
        self.model = AutoModel.from_pretrained(
            self.cfg.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).to("cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_path, trust_remote_code=True, use_fast=False)
        self.processor = self.tokenizer     # runner 兼容占位
        self.model.img_context_token_id = \
            self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT)
        print("模型加载完成。")

    @property
    def n_layers(self):
        return self.model.language_model.config.num_hidden_layers

    def create_probe(self, capture_layer: int = 22):
        from ..intervention import ScoreDeltaProbe
        probe = ScoreDeltaProbe(self.model, None, capture_layer=capture_layer)
        probe.n_layers = self.n_layers
        return probe

    # ── 输入构建 ────────────────────────────────────────────────
    def build_inputs(self, image, prompt: str, sample_name: str = None):
        grid = pick_grid(image)
        tiles = tile_images(image, grid)
        pixel_values = torch.stack(
            [self._transform(t) for t in tiles]).to(torch.bfloat16).cuda()

        # internvl2_5 模板（conversation.py, MPT 风格）
        n_tok = TOK_PER_TILE * len(tiles)
        image_block = IMG_START + IMG_CONTEXT * n_tok + IMG_END
        text = ("<|im_start|>system\n你是书生·万象，英文名是InternVL，是由上海"
                "人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
                "<|im_end|>\n<|im_start|>user\n"
                f"{image_block}\n{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n")
        enc = self.tokenizer(text, return_tensors="pt")
        return {
            "pixel_values": pixel_values,
            "input_ids": enc["input_ids"].cuda(),
            "attention_mask": enc["attention_mask"].cuda(),
            "_grid": grid,
            "_n_tiles": len(tiles),
        }

    # ── bbox -> 绝对序列索引 ─────────────────────────────────────
    def set_intervention_target(self, probe, inputs, boxes_px, orig_size,
                                weights=None):
        assert weights is None, "internvl35 暂不支持布局掩码权重"
        ids = inputs["input_ids"][0].tolist()
        ctx_id = self.model.img_context_token_id
        img_base = ids.index(ctx_id)
        cols, rows = inputs["_grid"]
        n_tiles = inputs["_n_tiles"]
        has_thumb = n_tiles > cols * rows
        W, H = orig_size

        offsets = set()
        for x1, y1, x2, y2 in boxes_px:
            # 瓦片视图: 原图各向异性拉伸到 (448*cols, 448*rows)
            sx, sy = TILE * cols / W, TILE * rows / H
            gx1, gy1, gx2, gy2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
            c1 = max(0, min(int(gx1 // CELL), GRID * cols - 1))
            r1 = max(0, min(int(gy1 // CELL), GRID * rows - 1))
            c2 = max(0, min(int(gx2 // CELL), GRID * cols - 1))
            r2 = max(0, min(int(gy2 // CELL), GRID * rows - 1))
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    tile = (r // GRID) * cols + (c // GRID)
                    offsets.add(tile * TOK_PER_TILE
                                + (r % GRID) * GRID + (c % GRID))
            # 缩略图视图（末位瓦）: 整图直接拉伸到 448x448
            if has_thumb:
                tx1 = max(0, min(int(x1 / W * TILE // CELL), GRID - 1))
                ty1 = max(0, min(int(y1 / H * TILE // CELL), GRID - 1))
                tx2 = max(0, min(int(x2 / W * TILE // CELL), GRID - 1))
                ty2 = max(0, min(int(y2 / H * TILE // CELL), GRID - 1))
                base = cols * rows * TOK_PER_TILE
                for r in range(ty1, ty2 + 1):
                    for c in range(tx1, tx2 + 1):
                        offsets.add(base + r * GRID + c)

        seq_idx = torch.tensor(sorted(img_base + o for o in offsets),
                               dtype=torch.long)
        probe.set_target_indices(seq_idx)

    def generate(self, inputs, max_new_tokens: int = 128) -> str:
        eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        with torch.no_grad():
            out = self.model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos,
            )
        # InternVLChatModel.generate 走 inputs_embeds，输出只含新 token
        text = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        return text.split("<|im_end|>")[0].strip()
