"""DocLayLLM 适配器（Tier B/C, vendor 副本 + 显式回调探针）。

已核实（vendor/doclayllm/modelling_llama3.py, tf 4.57.1 实测端到端可跑）:
  - 视觉: 196 patch token(id 150001, 14x14 行主序), 224x224 直接拉伸
  - 布局: 每条 OCR 行文本后跟 1 个 spatial token(id 150000)，其嵌入被替换为
    0-1000 归一化框的 LayoutLMv2 式 2D 嵌入（文本 token 本身不带 2D 位置）
  - 三个注意力类均为文件内手写, cache update 后已插 _LS_PROBE 回调
  - 必须 use_cache=True（cache-less 多步生成有 in-place 置零 bug）
  - checkpoint fp32 8.04B -> bf16 加载
干预目标: 答案 bbox 覆盖的 patch token；可选(--含文本) bbox 内 OCR 行的
文本 token span（构造 prompt 时记录）。
"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from ..intervention.hooks import VendorHookProbe
from .base import ModelAdapter

SPATIAL_ID = 150000
PATCH_ID = 150001
N_PATCH = 196            # 14x14
PATCH_GRID = 14


def _preprocess_image(image: Image.Image) -> torch.Tensor:
    """LayoutLMv3 图像预处理复刻: 224 拉伸 + 1/255 + mean=std=0.5。"""
    img = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1)


class DocLayLLMAdapter(ModelAdapter):
    name = "doclayllm"
    tier = "B"

    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self.tokenizer = None
        self._ocr_cache = None
        # 干预只作用于图像 token: OCR 文本与其坐标输入保持不变（见论文）
        self.include_text_tokens = False

    def load(self):
        from .vendor.doclayllm import modelling_llama3 as m
        self._vendor_mod = m
        print(f"加载模型: {self.cfg.model_path} (vendor, bf16, sdpa)")
        self.model = m.LlamaForCausalLM.from_pretrained(
            self.cfg.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to("cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path)
        self.processor = self.tokenizer
        print("模型加载完成。")

    @property
    def n_layers(self):
        return self.model.config.num_hidden_layers

    def create_probe(self, capture_layer: int = 22):
        return VendorHookProbe(self.model, self.tokenizer,
                               host_module=self._vendor_mod,
                               capture_layer=capture_layer)

    def _ocr_lines(self, sample_name):
        if self._ocr_cache is None:
            self._ocr_cache = self.dataset.load_layout_items("test")
        info = self._ocr_cache.get(sample_name)
        return info["items"] if info else []

    # ── 输入构建（官方 infer_demo 逐行复刻 + 行 span 记录）────────
    def build_inputs(self, image, prompt: str, sample_name: str = None):
        tok = self.tokenizer
        W, H = image.size
        items = self._ocr_lines(sample_name)

        fore = ("<|start_header_id|>system<|end_header_id|>\n\n"
                "You are a helpful assistant.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "Giving the document image patches,")
        ids = [tok.bos_token_id] + tok.encode(fore, add_special_tokens=False)
        patch_start = len(ids)
        ids += [PATCH_ID] * N_PATCH
        ids += tok.encode(', and text content and its location in form of '
                          '"text, [left, top, right, bottom]":\n',
                          add_special_tokens=False)

        bboxes, line_spans, line_boxes = [], [], []
        for it in items:
            text = it.text.strip()
            if not text:
                continue
            t_ids = tok.encode(text, add_special_tokens=False)
            start = len(ids)
            ids += t_ids + [SPATIAL_ID] + tok.encode("\n",
                                                     add_special_tokens=False)
            line_spans.append((start, start + len(t_ids) + 1))  # 含 spatial
            line_boxes.append(it.box)
            x1, y1, x2, y2 = it.box
            bboxes.append([max(0, min(int(x1 / W * 1000), 1000)),
                           max(0, min(int(y1 / H * 1000), 1000)),
                           max(0, min(int(x2 / W * 1000), 1000)),
                           max(0, min(int(y2 / H * 1000), 1000))])

        ids += tok.encode("\n" + prompt + "<|eot_id|>"
                          "<|start_header_id|>assistant<|end_header_id|>\n\n",
                          add_special_tokens=False)
        if not bboxes:   # 无 OCR 行时给占位行（forward 按 -100 过滤）
            bboxes = [[-100, -100, -100, -100]]

        dev = self.model.device
        return {
            "input_ids": torch.LongTensor([ids]).to(dev),
            "position_ids": torch.arange(len(ids)).unsqueeze(0).to(dev),
            "bbox": torch.LongTensor([bboxes]).to(dev),
            "pixel_values": _preprocess_image(image).unsqueeze(0)
                .to(dev, torch.bfloat16),
            "_patch_start": patch_start,
            "_line_spans": line_spans,
            "_line_boxes": line_boxes,
            "_img_size": (W, H),
        }

    # ── bbox -> patch token（+可选行文本 span）──────────────────
    def set_intervention_target(self, probe, inputs, boxes_px, orig_size,
                                weights=None):
        assert weights is None, "doclayllm 暂不支持布局掩码权重"
        W, H = orig_size
        offsets = set()
        for x1, y1, x2, y2 in boxes_px:
            c1 = max(0, min(int(x1 / W * PATCH_GRID), PATCH_GRID - 1))
            r1 = max(0, min(int(y1 / H * PATCH_GRID), PATCH_GRID - 1))
            c2 = max(0, min(int(x2 / W * PATCH_GRID), PATCH_GRID - 1))
            r2 = max(0, min(int(y2 / H * PATCH_GRID), PATCH_GRID - 1))
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    offsets.add(inputs["_patch_start"] + r * PATCH_GRID + c)
        if self.include_text_tokens:
            for (s, e), box in zip(inputs["_line_spans"],
                                   inputs["_line_boxes"]):
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                if any(bx1 <= cx <= bx2 and by1 <= cy <= by2
                       for bx1, by1, bx2, by2 in boxes_px):
                    offsets.update(range(s, e))
        seq_idx = torch.tensor(sorted(offsets), dtype=torch.long)
        probe.set_target_indices(seq_idx)

    def generate(self, inputs, max_new_tokens: int = 128) -> str:
        feed = {k: v for k, v in inputs.items() if not k.startswith("_")}
        # vendor forward 会就地置零 150000/150001 占位 id，
        # 同一 inputs 要跑 normal+steered 两次生成，必须克隆
        feed["input_ids"] = feed["input_ids"].clone()
        n_in = feed["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **feed, max_new_tokens=max_new_tokens, do_sample=False,
                use_cache=True, bos_token_id=128000,
                eos_token_id=[128001, 128009], pad_token_id=128001)
        return self.tokenizer.decode(out[0][n_in:],
                                     skip_special_tokens=True).strip()
