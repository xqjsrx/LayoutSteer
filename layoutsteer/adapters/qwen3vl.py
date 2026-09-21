"""Qwen3-VL 适配器（Tier A）。

与 Qwen2.5-VL 机制同构（已核实 transformers 4.57 源码）:
  - 文本解码注意力走 ALL_ATTENTION_FUNCTIONS 分发 -> ScoreDeltaProbe 零改动
  - vision_start/end/image_pad token id 与 Qwen2.5-VL 相同，image token
    连续块，网格 = grid_thw // spatial_merge_size(2)，像素步长 32（patch16）
  - DeepStack 仅在解码层 0-2 对图像位置做原位加法，不改序列长度/位置
  - 视觉塔注意力也走同一分发，但其 module 无 layer_idx，wrapper 自动跳过
"""
import torch
from transformers import AutoProcessor

from .qwen25vl import Qwen25VLAdapter


class Qwen3VLAdapter(Qwen25VLAdapter):
    name = "qwen3vl"
    tier = "A"

    def load(self):
        from transformers import Qwen3VLForConditionalGeneration
        print(f"加载模型: {self.cfg.model_path} "
              f"(attn={self.cfg.attn_implementation})")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.cfg.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=self.cfg.attn_implementation,
        ).to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(self.cfg.model_path)
        print("模型加载完成。")

    def load_for_localization(self):
        """embedding 提取用: sdpa + 低分辨率。"""
        from transformers import Qwen3VLForConditionalGeneration
        print(f"加载模型: {self.cfg.model_path} (attn=sdpa, 定位低分辨率)")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.cfg.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_path, max_pixels=self.cfg.localize_max_pixels)
        self.processor.image_processor.size["longest_edge"] = \
            self.cfg.localize_max_pixels
        print("模型加载完成。")
