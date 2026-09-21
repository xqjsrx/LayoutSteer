"""DeepSeek-OCR 适配器（Tier B/C, 仅限 transformers==4.46.3 env: qwt-dsocr）。

已核实（remote code modeling_deepseekocr.py / modeling_deepseekv2.py）:
  - 语言侧 12 层 MHA(10 头, head_dim 128, kv_lora_rank=None)；
    ATTENTION_CLASSES 取 transformers 4.46 stock LlamaAttention（eager）
    -> 对 stock LlamaAttention.forward 打补丁注入 _LS_PROBE 回调
  - 图像 token 单段连续（id 128815）, 但 masked_scatter_ 的特征序为
    [局部瓦片网格, 全局视图, view_seperator]，与 token 计数分组无关:
      局部段: R=10*h_num 行 x (C=10*w_num) 列, 行末 1 个 newline 槽
              (stride C+1)，坐标系 = 原图各向异性拉伸到 (640*w, 640*h)
      全局段: 16x16 + 行末 newline (stride 17)，坐标系 = ImageOps.pad
              保比例居中到 1024x1024
      小图(<=640x640)无局部段
  - prompt 为 plain 拼接: bos + "<image>\n{question}"
"""
import re
import sys
import math

import torch

from ..intervention.hooks import VendorHookProbe
from .base import ModelAdapter

IMAGE_TOKEN_ID = 128815
BASE_SIZE = 1024      # 全局视图
TILE_SIZE = 640       # 局部瓦片
CELL = 64             # 16px patch x4 下采样
G_GRID = BASE_SIZE // CELL          # 16
T_GRID = TILE_SIZE // CELL          # 10


class DeepSeekOCRAdapter(ModelAdapter):
    name = "deepseekocr"
    tier = "C"

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.tokenizer = None
        # 逐行存在 image_newline 槽, 行 stride = C+1（token 构造与 bbox 映射同步使用）
        self.has_newline = True

    def load(self):
        import transformers
        assert transformers.__version__.startswith("4.46"), \
            f"deepseekocr 需 transformers 4.46.x（qwt-dsocr env），当前 {transformers.__version__}"
        from transformers import AutoModel, AutoTokenizer
        print(f"加载模型: {self.cfg.model_path} (eager, trust_remote_code)")
        self.model = AutoModel.from_pretrained(
            self.cfg.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            trust_remote_code=True,
            use_safetensors=True,
        ).to("cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_path, trust_remote_code=True)
        self.processor = self.tokenizer
        self._remote = sys.modules[type(self.model).__module__]
        from .vendor.dsocr_llama_eager import patch_llama_eager_forward
        self._hook_host = patch_llama_eager_forward()
        print("模型加载完成。")

    @property
    def n_layers(self):
        return self.model.config.num_hidden_layers

    def create_probe(self, capture_layer: int = 6):
        return VendorHookProbe(self.model, self.tokenizer,
                               host_module=self._hook_host,
                               capture_layer=capture_layer)

    # ── 输入构建（infer() 逐行复刻, 不落盘不 streamer）─────────────
    def build_inputs(self, image, prompt: str, sample_name: str = None):
        from PIL import ImageOps
        m = self._remote
        tok = self.tokenizer
        transform = m.BasicImageTransform(mean=(0.5, 0.5, 0.5),
                                          std=(0.5, 0.5, 0.5), normalize=True)
        image = image.convert("RGB")
        w, h = image.size

        crops, crop_ratio = [], [1, 1]
        if w > TILE_SIZE or h > TILE_SIZE:
            crops, crop_ratio = m.dynamic_preprocess(image)
        w_num, h_num = crop_ratio
        global_view = ImageOps.pad(
            image, (BASE_SIZE, BASE_SIZE),
            color=tuple(int(x * 255) for x in transform.mean))
        images_ori = torch.stack([transform(global_view).to(torch.bfloat16)])
        if crops:
            images_crop = torch.stack(
                [transform(c).to(torch.bfloat16) for c in crops])
        else:
            images_crop = torch.zeros((1, 3, BASE_SIZE, BASE_SIZE),
                                      dtype=torch.bfloat16)

        # token 序列: bos + "" + [image tokens] + "\n{prompt}"
        nq_base, nq = G_GRID, T_GRID
        nl = [IMAGE_TOKEN_ID] if self.has_newline else []
        tokenized_image = ([IMAGE_TOKEN_ID] * nq_base + nl) * nq_base \
            + [IMAGE_TOKEN_ID]
        if w_num > 1 or h_num > 1:
            tokenized_image += ([IMAGE_TOKEN_ID] * (nq * w_num) + nl) \
                * (nq * h_num)
        tail_ids = m.text_encode(tok, "\n" + prompt, bos=False, eos=False)
        ids = [0] + tokenized_image + tail_ids
        seq_mask = ([False] + [True] * len(tokenized_image)
                    + [False] * len(tail_ids))

        dev = "cuda"
        return {
            "input_ids": torch.LongTensor([ids]).to(dev),
            "images": [(images_crop.to(dev), images_ori.to(dev))],
            "images_seq_mask": torch.tensor([seq_mask], dtype=torch.bool).to(dev),
            "images_spatial_crop": torch.tensor([[w_num, h_num]],
                                                dtype=torch.long),
            "_img_base": 1,                # bos 之后第一个 image token
            "_crop_ratio": (w_num, h_num),
            "_img_size": (w, h),
        }

    # ── bbox -> 绝对序列索引（局部段 + 全局段）───────────────────
    def set_intervention_target(self, probe, inputs, boxes_px, orig_size,
                                weights=None):
        assert weights is None, "deepseekocr 暂不支持布局掩码权重"
        W, H = orig_size
        w_num, h_num = inputs["_crop_ratio"]
        P = inputs["_img_base"]
        has_local = w_num > 1 or h_num > 1
        nl = 1 if self.has_newline else 0   # 行末 newline 槽
        # 特征填充序: [局部, 全局, sep]；token 槽位按此序占用
        local_len = (T_GRID * w_num + nl) * (T_GRID * h_num) if has_local else 0
        G = P + local_len

        # 全局视图 pad 保比例居中
        s = BASE_SIZE / max(W, H)
        pad_x = (BASE_SIZE - W * s) / 2
        pad_y = (BASE_SIZE - H * s) / 2

        offsets = set()
        for x1, y1, x2, y2 in boxes_px:
            if has_local:
                C = T_GRID * w_num
                R = T_GRID * h_num
                sx = TILE_SIZE * w_num / W
                sy = TILE_SIZE * h_num / H
                c1 = max(0, min(int(x1 * sx // CELL), C - 1))
                r1 = max(0, min(int(y1 * sy // CELL), R - 1))
                c2 = max(0, min(int(x2 * sx // CELL), C - 1))
                r2 = max(0, min(int(y2 * sy // CELL), R - 1))
                for r in range(r1, r2 + 1):
                    for c in range(c1, c2 + 1):
                        offsets.add(P + r * (C + nl) + c)
            gx1 = max(0, min(int((x1 * s + pad_x) // CELL), G_GRID - 1))
            gy1 = max(0, min(int((y1 * s + pad_y) // CELL), G_GRID - 1))
            gx2 = max(0, min(int((x2 * s + pad_x) // CELL), G_GRID - 1))
            gy2 = max(0, min(int((y2 * s + pad_y) // CELL), G_GRID - 1))
            for r in range(gy1, gy2 + 1):
                for c in range(gx1, gx2 + 1):
                    offsets.add(G + r * (G_GRID + nl) + c)

        seq_idx = torch.tensor(sorted(offsets), dtype=torch.long)
        probe.set_target_indices(seq_idx)

    def generate(self, inputs, max_new_tokens: int = 128) -> str:
        feed = {k: v for k, v in inputs.items() if not k.startswith("_")}
        n_in = feed["input_ids"].shape[1]
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            out = self.model.generate(
                feed["input_ids"],
                images=feed["images"],
                images_seq_mask=feed["images_seq_mask"],
                images_spatial_crop=feed["images_spatial_crop"],
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                use_cache=True)
        text = self.tokenizer.decode(out[0, n_in:])
        stop = "<｜end▁of▁sentence｜>"
        if text.endswith(stop):
            text = text[:-len(stop)]
        return _unwrap_sentence(text.strip())


def _unwrap_sentence(text: str) -> str:
    """去句子外壳: 该模型习惯输出 "The date ... is X."（normal 与干预
    两路同规则处理，Δ 口径公平；指令模型输出裸值不受影响）。"""
    m = re.match(r"^[Tt]he\s.{0,60}?\bis\s*:?\s*(.+)$", text, re.S)
    if m:
        text = m.group(1).strip()
    if text.endswith("."):
        text = text[:-1].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text
