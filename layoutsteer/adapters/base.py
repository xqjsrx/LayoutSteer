"""ModelAdapter 接口: 干预实验的模型无关抽象。

约定:
  - boxes_px 一律为原图像素坐标（定位产物坐标系, 模型无关）
  - probe 由 create_probe() 提供, 须支持
        attach(delta, target_layers=..., capture=..., persistent=...)
    上下文管理器协议（与 ScoreDeltaProbe 相同）
  - set_intervention_target() 负责本模型的 bbox -> token 序列索引换算
    （网格/瓦片/newline 槽等模型差异全部封装在此, probe 不感知）
"""
from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    name: str = ""
    #: 干预钩子层级: "A" ALL_ATTENTION_FUNCTIONS 分发 /
    #: "B" 自定义注意力类补丁 / "C" eager score 列加 δ
    tier: str = ""

    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self.model = None
        self.processor = None

    # ── 加载 ────────────────────────────────────────────────────
    @abstractmethod
    def load(self):
        """加载模型与 processor（就地填充 self.model / self.processor）。"""

    @property
    @abstractmethod
    def n_layers(self) -> int:
        """语言侧解码层数（target_layers 解析用）。"""

    # ── 干预 ────────────────────────────────────────────────────
    @abstractmethod
    def create_probe(self, capture_layer: int = 0):
        """返回本模型的干预探针。"""

    @abstractmethod
    def set_intervention_target(self, probe, inputs, boxes_px, orig_size,
                                weights=None):
        """把原图像素 bbox 换算为图像 token 目标并设置到 probe。

        weights: 可选逐框/逐格干预权重（布局掩码模式）。
        """

    # ── 推理 ────────────────────────────────────────────────────
    def bind_dataset(self, dataset):
        """需要数据集上下文的模型（如 DocLayLLM 的 OCR 行）在此接收句柄。"""
        self.dataset = dataset

    @abstractmethod
    def build_inputs(self, image, prompt: str, sample_name: str = None):
        """构建单图单轮推理输入（返回可直接喂 generate 的 dict/对象）。

        sample_name: 需要 OCR 等样本级上下文的模型使用（其余忽略）。
        """

    @abstractmethod
    def generate(self, inputs, max_new_tokens: int = 128) -> str:
        """贪心生成并解码新增部分。"""

    def prepare_image(self, image):
        """推理前的图像预处理（默认原样返回; Qwen 系按像素上限缩放）。"""
        return image
