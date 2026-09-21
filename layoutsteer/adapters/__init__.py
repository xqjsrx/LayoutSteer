"""模型适配层: 把"模型加载/输入构建/图像token定位/干预钩子"从 runner 解耦。

每个 adapter 负责一个模型家族:
  - load(): 加载模型与 processor
  - build_inputs(image, prompt): 构建推理输入
  - set_intervention_target(probe, inputs, boxes_px, orig_size):
      把原图像素 bbox 换算为该模型图像 token 的序列索引并设置到 probe
  - create_probe(): 返回带 attach(delta, target_layers, ...) 的干预探针
      （Tier A: ALL_ATTENTION_FUNCTIONS 替换 / Tier B: 类 forward 补丁 /
       Tier C: eager score 列加 δ）
  - generate(inputs, max_new_tokens): 贪心生成并解码

定位（布局检索）不在此层: 干预框一律复用 Qwen2.5-VL 的定位产物
（原图像素坐标, 模型无关）。
"""
from .base import ModelAdapter


def get_adapter(model_cfg) -> ModelAdapter:
    """按 ModelConfig.name 实例化 adapter（延迟导入, 避免装载无关依赖）。"""
    name = getattr(model_cfg, "name", "qwen25vl")
    if name == "qwen25vl":
        from .qwen25vl import Qwen25VLAdapter
        return Qwen25VLAdapter(model_cfg)
    if name == "qwen3vl":
        from .qwen3vl import Qwen3VLAdapter
        return Qwen3VLAdapter(model_cfg)
    if name == "llavaov15":
        from .llavaov15 import LlavaOV15Adapter
        return LlavaOV15Adapter(model_cfg)
    if name == "internvl35":
        from .internvl35 import InternVL35Adapter
        return InternVL35Adapter(model_cfg)
    if name == "doclayllm":
        from .doclayllm import DocLayLLMAdapter
        return DocLayLLMAdapter(model_cfg)
    if name == "deepseekocr":
        from .deepseekocr import DeepSeekOCRAdapter
        return DeepSeekOCRAdapter(model_cfg)
    if name == "docowl2":
        from .docowl2 import DocOwl2Adapter
        return DocOwl2Adapter(model_cfg)
    raise ValueError(f"未知模型适配器: {name}")
