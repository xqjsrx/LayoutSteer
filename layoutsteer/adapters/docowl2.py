"""DocOwl2 适配器（Tier C: softmax 前对目标 score 列加 δ）。

已核实（vendor/docowl2 远程代码）:
  - LLM 解码注意力为手写 eager（MAM 多路 K/V, modeling_llama2_mam.py:947
    attn_weights = matmul(q, k^T)/√d），无 layer_idx 无 Cache.update
    -> Tier C: 在 attn_weights 上给目标图像 token 列加 δ（与 ΔK 等价），
       层号由 adapter 在加载后逐层写入 self_attn._ls_layer_idx
  - 视觉: HRDocCompressor 以全局视图 token 为 query, 输出保持全局视图的
    行主序网格 + 1 个 compressor_eos, 每页单段连续, 插在 IMAGE_TOKEN_INDEX
    (-200) 位置。网格形状与 token 数由运行时探测确定（不硬编码）。
  - 官方入口 model.chat 内部走 processor + generate; 此处复用 processor
    自行驱动 generate 以插入干预。
"""

import sys

import torch
from transformers import AutoTokenizer

from ..intervention.hooks import ScoreAddProbe
from .base import ModelAdapter

IMAGE_TOKEN_INDEX = -200
BASIC_SIZE = 504


class DocOwl2Adapter(ModelAdapter):
    name = "docowl2"
    tier = "C"

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.tokenizer = None
        self._grid = None          # (rows, cols) 运行时探测

    def load(self):
        from .vendor.docowl2 import modeling_mplug_docowl as m
        self._vendor_mod = sys.modules[
            m.MPLUGDocOwl2.__module__.rsplit(".", 1)[0] + ".modeling_llama2_mam"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_path, use_fast=False)
        print(f"加载模型: {self.cfg.model_path} (vendor, bf16, eager)")
        self.model = m.MPLUGDocOwl2.from_pretrained(
            self.cfg.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        self.model.init_processor(tokenizer=self.tokenizer,
                                 basic_image_size=BASIC_SIZE,
                                 crop_anchors="grid_12")
        self.processor = self.tokenizer
        # 手写注意力无 layer_idx，逐层写入供 Tier C 回调识别
        for i, layer in enumerate(self.model.model.layers):
            layer.self_attn._ls_layer_idx = i
        print("模型加载完成。")

    @property
    def n_layers(self):
        return self.model.config.num_hidden_layers

    def create_probe(self, capture_layer: int = 22):
        return ScoreAddProbe(self.model, self.tokenizer,
                             host_module=self._vendor_mod,
                             capture_layer=capture_layer)

    # ── 输入构建（复刻 chat 前半段）──────────────────────────────
    def build_inputs(self, image, prompt: str, sample_name: str = None):
        messages = [{"role": "USER", "content": "<|image|>" + prompt}]
        image_tensor, patch_positions, input_ids = self.model.processor(
            images=[image], messages=messages)
        dev = self.model.device
        return {
            "input_ids": input_ids.unsqueeze(0).to(dev),
            "images": image_tensor.to(dev, dtype=torch.bfloat16),
            "patch_positions": patch_positions.to(dev),
            "_img_size": image.size,
        }

    # ── bbox -> 绝对序列索引（全局视图行主序单段）─────────────────
    def set_intervention_target(self, probe, inputs, boxes_px, orig_size,
                                weights=None):
        assert weights is None, "docowl2 暂不支持布局掩码权重"
        ids = inputs["input_ids"][0].tolist()
        ph = ids.index(IMAGE_TOKEN_INDEX)      # 占位符位置（展开前）
        n_tok, rows, cols = self._probe_grid(inputs)
        # 展开后图像块起点 = 占位符之前的文本 token 数（占位符本身被替换）
        base = ph
        W, H = orig_size
        cell_h = H / rows
        cell_w = W / cols

        offsets = set()
        for x1, y1, x2, y2 in boxes_px:
            r1 = max(0, min(int(y1 / cell_h), rows - 1))
            r2 = max(0, min(int(y2 / cell_h), rows - 1))
            c1 = max(0, min(int(x1 / cell_w), cols - 1))
            c2 = max(0, min(int(x2 / cell_w), cols - 1))
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    offsets.add(base + r * cols + c)
        probe.set_target_indices(
            torch.tensor(sorted(offsets), dtype=torch.long))

    def _probe_grid(self, inputs):
        """运行时探测每页图像 token 数与网格形状（不硬编码 324/36x9）。"""
        if self._grid is not None:
            return self._grid
        with torch.no_grad():
            _, _, _, _, embeds, _ = \
                self.model.prepare_inputs_labels_for_multimodal(
                    inputs["input_ids"], None, None, None,
                    inputs["images"], inputs["patch_positions"])
        n_total = embeds.shape[1]
        n_text = inputs["input_ids"].shape[1] - 1        # 占位符替换
        n_img = n_total - n_text
        # 网格: 全局视图 504 上按 ViT patch 14 / H-Reducer 列压缩 4 ->
        # rows = 504/14 = 36, cols = n_img_without_eos / rows
        rows = BASIC_SIZE // 14
        cols = max(1, (n_img - 1) // rows)               # -1 去 compressor_eos
        assert rows * cols <= n_img, (n_img, rows, cols)
        self._grid = (n_img, rows, cols)
        print(f"docowl2 图像 token: {n_img}（网格 {rows}x{cols} + eos）")
        return self._grid

    def generate(self, inputs, max_new_tokens: int = 128) -> str:
        # 该模型全链路是旧式 legacy-tuple cache，tf>=4.50 的 generate 预建
        # DynamicCache 无法兼容 -> 自写贪心循环，显式传 legacy tuple。
        # 干预 hooks 在注意力 forward 内部，照常触发。
        ids = inputs["input_ids"]
        dev = ids.device
        eos = self.tokenizer.eos_token_id
        past = None
        cur = ids
        generated = []
        with torch.inference_mode():
            for step in range(max_new_tokens):
                past_len = past[0][0].shape[2] if past is not None else 0
                attn_mask = torch.ones((1, past_len + cur.shape[1]),
                                       dtype=torch.long, device=dev)
                out = self.model(
                    input_ids=cur,
                    attention_mask=attn_mask,
                    past_key_values=past,
                    images=inputs["images"] if step == 0 else None,
                    patch_positions=(inputs["patch_positions"]
                                     if step == 0 else None),
                    use_cache=True,
                    return_dict=True)
                next_id = int(out.logits[:, -1, :].argmax(-1).item())
                generated.append(next_id)
                past = out.past_key_values
                cur = torch.tensor([[next_id]], device=dev)
                if next_id == eos:
                    break
        text = self.tokenizer.decode(generated).strip()
        return text.replace("</s>", "").strip()
