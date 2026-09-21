"""模型加载与输入构建（Qwen2.5-VL + Flash Attention 2）。"""
import math
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from .config import ModelConfig


def setup_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_model_and_processor(cfg: ModelConfig, for_localization: bool = False):
    """加载 Qwen2.5-VL。

    for_localization=True 时使用 sdpa + 低分辨率（embedding 提取，不需要干预）。
    """
    # 延迟导入: transformers<4.49（DeepSeek-OCR env）无此类，勿在模块级失败
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    attn_impl = "sdpa" if for_localization else cfg.attn_implementation
    print(f"加载模型: {cfg.model_path} (attn={attn_impl})")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    ).to("cuda").eval()

    if for_localization:
        processor = AutoProcessor.from_pretrained(
            cfg.model_path, max_pixels=cfg.localize_max_pixels)
        processor.image_processor.size["longest_edge"] = cfg.localize_max_pixels
    else:
        processor = AutoProcessor.from_pretrained(cfg.model_path)

    print("模型加载完成。")
    return model, processor


def resize_image_by_pixel_limit(image: Image.Image, max_pixels: int) -> Image.Image:
    """超过像素上限时按比例缩小（防显存溢出，沿用旧版阈值策略）。"""
    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (width * height))
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def build_inputs(processor, image: Image.Image, prompt_text: str, device):
    """构建 Qwen2.5-VL 推理输入。"""
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt_text},
        {"type": "image", "image": image},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True,
                       return_tensors="pt").to(device)
    return inputs


def generate_and_decode(model, processor, inputs, max_new_tokens: int = 128) -> str:
    """贪心生成并解码新增部分。"""
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    return processor.batch_decode(
        generated_ids[:, input_len:], skip_special_tokens=True)[0].strip()
