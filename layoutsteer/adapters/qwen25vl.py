"""Qwen2.5-VL 默认适配器（Tier A）。

对既有 model_loader / regions / ScoreDeltaProbe 的薄包装——
所有调用与 runner 原实现逐行等价, 保证历史实验可复现。
"""
from ..model_loader import (
    load_model_and_processor, resize_image_by_pixel_limit,
    build_inputs, generate_and_decode)
from ..intervention import (
    ScoreDeltaProbe, bboxes_px_to_rel, bboxes_to_grid_indices)
from .base import ModelAdapter


class Qwen25VLAdapter(ModelAdapter):
    name = "qwen25vl"
    tier = "A"

    def load(self):
        self.model, self.processor = load_model_and_processor(self.cfg)

    def load_for_localization(self):
        """embedding 提取用: sdpa + 低分辨率（不需要干预, 省显存加速）。"""
        self.model, self.processor = load_model_and_processor(
            self.cfg, for_localization=True)

    @property
    def n_layers(self):
        text_config = getattr(self.model.config, "text_config",
                              self.model.config)
        return text_config.num_hidden_layers

    def create_probe(self, capture_layer: int = 22):
        return ScoreDeltaProbe(self.model, self.processor,
                               capture_layer=capture_layer)

    def set_intervention_target(self, probe, inputs, boxes_px, orig_size,
                                weights=None):
        probe.setup_image_range(inputs["input_ids"], inputs["image_grid_thw"])
        boxes_rel = bboxes_px_to_rel(boxes_px, orig_size)
        grid_idx = bboxes_to_grid_indices(boxes_rel, probe.spatial_shape)
        probe.set_region(grid_idx, weights)

    def prepare_image(self, image):
        return resize_image_by_pixel_limit(image, self.cfg.max_image_pixels)

    def build_inputs(self, image, prompt: str, sample_name: str = None):
        return build_inputs(self.processor, image, prompt, self.model.device)

    def generate(self, inputs, max_new_tokens: int = 128) -> str:
        return generate_and_decode(self.model, self.processor, inputs,
                                   max_new_tokens)
