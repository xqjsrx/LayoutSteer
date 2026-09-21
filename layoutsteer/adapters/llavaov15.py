"""LLaVA-OneVision-1.5 适配器（Tier B）。

已核实（remote code modeling_llavaonevision1_5.py, tf 4.57.1）:
  - 解码器为自带 LLaVAOneVision1_5_ATTENTION_CLASSES（:791-795），
    不走 ALL_ATTENTION_FUNCTIONS -> 需 ClassPatchProbe 符号替换
  - FA2 forward 在 past_key_value.update 后 repeat_kv 并转置为
    [B, seq, 32, 128] 调 `_flash_attention_forward`（:683-693）
  - 图像 token: Qwen2VLImageProcessor（patch14 merge2, 28px 步长），
    vision_start/image_pad/vision_end 同 Qwen 系 id，连续块无缩略图
    -> bbox 映射与 Qwen2.5-VL 完全同式，直接继承
  - 4.57.1 阻塞点: remote code 从 transformers.modeling_flash_attention_utils
    导入已删除的 flash_attn_varlen_func -> 加载前属性注入（零改源码）
"""
import sys

import torch
from transformers import AutoProcessor

from ..model_loader import build_inputs, generate_and_decode
from ..intervention.hooks import ClassPatchProbe
from .qwen25vl import Qwen25VLAdapter


def _inject_flash_symbols():
    """把 flash_attn 的函数补进 transformers.modeling_flash_attention_utils。"""
    import transformers.modeling_flash_attention_utils as m
    if not hasattr(m, "flash_attn_varlen_func"):
        from flash_attn import flash_attn_varlen_func
        m.flash_attn_varlen_func = flash_attn_varlen_func


class LlavaOV15Adapter(Qwen25VLAdapter):
    name = "llavaov15"
    tier = "B"

    def load(self):
        from transformers import AutoModelForCausalLM
        _inject_flash_symbols()
        print(f"加载模型: {self.cfg.model_path} "
              f"(attn={self.cfg.attn_implementation}, trust_remote_code)")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=self.cfg.attn_implementation,
            trust_remote_code=True,
        ).to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_path, trust_remote_code=True)
        print("模型加载完成。")

    def create_probe(self, capture_layer: int = 22):
        remote_mod = sys.modules[type(self.model).__module__]
        attn_modules = [
            layer.self_attn
            for layer in self.model.model.language_model.layers]
        return ClassPatchProbe(
            self.model, self.processor,
            attn_modules=attn_modules,
            patch_host=remote_mod,
            symbol="_flash_attention_forward",
            capture_layer=capture_layer)

    def build_inputs(self, image, prompt: str, sample_name: str = None):
        return build_inputs(self.processor, image, prompt, self.model.device)

    def generate(self, inputs, max_new_tokens: int = 128) -> str:
        return generate_and_decode(self.model, self.processor, inputs,
                                   max_new_tokens)
