"""全局配置：模型、数据集路径、干预与定位参数。

所有硬编码路径集中于此，其余模块一律通过 dataclass 读取。
"""
import os
from dataclasses import dataclass, field
from datetime import datetime

# ── 项目根目录 ────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# 模型与数据集的根目录，可用环境变量覆盖（默认放在项目内）
MODEL_ROOT = os.environ.get("LAYOUTSTEER_MODEL_ROOT",
                            os.path.join(PROJECT_ROOT, "models"))
DATASET_ROOT = os.environ.get("LAYOUTSTEER_DATASET_ROOT",
                              os.path.join(PROJECT_ROOT, "dataset"))


# ── 模型配置 ──────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    name: str = "qwen25vl"             # 适配器名（layoutsteer.adapters）
    model_path: str = os.path.join(MODEL_ROOT, "Qwen2.5-VL-7B-Instruct")
    attn_implementation: str = "flash_attention_2"
    # 推理侧图像像素上限（与旧版 refactor_fix 的干预安全阈值一致）
    max_image_pixels: int = 1_400_000
    # 定位侧（embedding 提取）使用低分辨率，加速且省显存
    localize_max_pixels: int = 256 * 28 * 28


# 论文所用模型的注册表（定位一律复用 Qwen2.5-VL 的布局产物, 与模型无关）
MODEL_PRESETS = {
    # ── 通用 VLM（OCR-free）──
    "qwen25vl": dict(model_path=os.path.join(
        MODEL_ROOT, "Qwen2.5-VL-7B-Instruct")),
    "qwen3vl": dict(model_path=os.path.join(
        MODEL_ROOT, "Qwen3-VL-8B-Instruct")),
    "llavaov15": dict(model_path=os.path.join(
        MODEL_ROOT, "LLaVA-OneVision-1.5-8B-Instruct")),
    "internvl35": dict(model_path=os.path.join(MODEL_ROOT, "InternVL3_5-8B")),
    # ── 专用文档模型（OCR-free）──
    "deepseekocr": dict(model_path=os.path.join(MODEL_ROOT, "DeepSeek-OCR")),
    "docowl2": dict(model_path=os.path.join(MODEL_ROOT, "DocOwl2")),
    # ── 基于 OCR 的文档模型 ──
    "doclayllm": dict(model_path=os.path.join(MODEL_ROOT, "DocLayLLM_sft")),
}


def make_model_config(name: str) -> ModelConfig:
    if name not in MODEL_PRESETS:
        raise ValueError(f"未知模型: {name}（可选 {sorted(MODEL_PRESETS)}）")
    return ModelConfig(name=name, **MODEL_PRESETS[name])


# ── 干预配置 ──────────────────────────────────────────────────────
@dataclass
class InterventionConfig:
    # 默认值 = SROIE 全量扫描最优单配置: δ=2 @ 晚层 19-27 + 置信度加权
    #   retrieval 框 93.30%（基线 91.86%）；浅层干预伤转写（全层 δ=5 崩坏 -10.6pt，
    #   晚层 δ=5 仅 -1.1pt），层消融见 runs/layersweep_* 与 runs/deltasweep_*_L19-27
    delta: float = 2.0                 # score_delta 的 δ（对应旧版 strength）
    target_layers: str = "19-27"       # "all" / "22" / "20,21,22" / "19-27"区间
    capture_layer: int = 22            # 可视化捕获层
    persistent: bool = False           # 一次性 KV cache 编辑（prefill 写入，
                                       # decode 零干预；全量验证与 per-step 持平）
    # ── 定位置信度加权: δ_eff = δ × c^γ（c = 区域覆盖概率标定值，gt 源恒 1）──
    use_confidence: bool = False       # 启用置信度加权
    conf_gamma: float = 1.0            # 锐度：>1 对低置信更保守
    conf_cutoff: float = 0.0           # c < cutoff 时关断干预（δ_eff=0）


# ── 定位配置 ──────────────────────────────────────────────────────
@dataclass
class LocalizationConfig:
    # 默认值 = sweep_localize 在 SROIE (tag=n15_g0.05) 上的均衡最优组合:
    #   macro top1=61.6% region_cover=89.6% area=10.2%（多区域 + 位置先验）
    global_topk: int = 5               # Stage1 全局检索模板数（少而精优于多而杂）
    n_local_bboxes: int = 15           # 局部布局图包含的邻近 bbox 数
    local_topk: int = 2                # 每个模板投票的候选数（票少而精）
    cluster_dist: float = 240.0        # 层次聚类距离阈值（像素，大阈值合并相邻投票）
    position_prior: float = 0.2        # >0: 投票相似度减 λ*|候选归一化y - 模板锚点y|，
                                       # 压制环面对称性带来的首尾混淆
    wrap_gap_ratio: float = 0.05       # 环形拼接处的空隙（占图宽/高比例），区分 wrap 内容与真实邻居
    use_wrap: bool = True              # False: 局部邻域不做环面拼接（边缘锚点只看真实邻居）
    # ── 区域构造层（从选中簇到最终区域框的后处理，均不影响 embedding）──
    max_regions: int = 2               # 最终输出的区域框上限（每入选簇一框）
    merge_weight_ratio: float = 0.1    # >0: 权重 >= ratio*最优簇 的次簇也输出区域框
                                       # （有位置先验后第二簇质量高，宽松接纳最优）
    trim_mult: float = 2.5             # >0: 剔除距成员中心 > mult*中位距离 的离群成员
    region_pad: float = 0.02           # 区域框按文档内容尺寸比例四向外扩
    global_prompt: str = "Represent the layout structure of this document image."
    local_prompt: str = "Represent the layout structure of this region."

    @property
    def local_tag(self) -> str:
        """局部布局 embedding 缓存目录后缀（携带影响渲染的超参，参数变则自动重建）。"""
        if not self.use_wrap:
            return f"n{self.n_local_bboxes}_nowrap"
        return f"n{self.n_local_bboxes}_g{self.wrap_gap_ratio:g}"


# ── 记忆机制配置 ─────────────────────────────────────────────────
@dataclass
class MemoryConfig:
    """高置信定位结果升格为伪模板的入库与检索参数。

    门控标定 (SROIE, 1367 条): best_weight >= 全局 P75 时伪锚点 top1 精度 91.5%。
    分位按实体各自计算，避免只有好定位的 address/company 入库。
    """
    gate_weight_quantile: float = 0.75  # 按实体取 best_weight 分位阈值入库
    gate_single_region: bool = False    # 附加要求单区域框（投票共识）
    mem_weight: float = 0.7             # 伪模板投票降权系数
    mem_max_k: int = 2                  # Stage1 记忆模板检索上限（保 GT 模板多数）


# ── 数据集路径 ───────────────────────────────────────────────────
@dataclass
class DatasetPaths:
    name: str
    test_qa_json: str                  # LayTextLLM 格式 QA 数据
    test_image_dir: str                # 推理用图片目录（裁剪图；POIE 为原图）
    test_bbox_json: str                # GT 答案框（与图片同坐标系）
    train_bbox_json: str               # 训练集答案框（定位模板用）
    train_image_dir: str               # 训练集图片目录


def _standard_paths(name: str) -> DatasetPaths:
    """统一目录约定: dataset/{name}/{train,test}/{images, answer_bboxes.json, qa_test.json}"""
    root = os.path.join(DATASET_ROOT, name)
    return DatasetPaths(
        name=name,
        test_qa_json=os.path.join(root, "test", "qa_test.json"),
        test_image_dir=os.path.join(root, "test", "images"),
        test_bbox_json=os.path.join(root, "test", "answer_bboxes.json"),
        train_bbox_json=os.path.join(root, "train", "answer_bboxes.json"),
        train_image_dir=os.path.join(root, "train", "images"),
    )


DATASET_PATHS = {
    name: _standard_paths(name)
    for name in ("sroie", "cord", "funsd", "poie")
}


def bootstrap_dir(ds_name: str, model: str = "qwen25vl") -> str:
    """自举产物目录: anchors / bootstrap_meta / coldstart 结果。

    自举产物（注意力锚点、模板库、冷启动池）是模型相关的,
    非默认模型按 _{model} 后缀分目录, qwen25vl 保持历史路径不变。
    """
    suffix = "" if model == "qwen25vl" else f"_{model}"
    return os.path.join(OUTPUT_DIR, ds_name, f"bootstrap{suffix}")


def qtemplate_dir(ds_name: str, model: str = "qwen25vl") -> str:
    """dK 记忆库目录（模型相关, 命名规则同 bootstrap_dir）。"""
    suffix = "" if model == "qwen25vl" else f"_{model}"
    return os.path.join(OUTPUT_DIR, ds_name, f"qtemplate{suffix}")


def morph_tag(dilate_ratio: float) -> str:
    """morph 缓存后缀（0.04 -> "morph04"）。

    anchor_idxs 是 extract_text_boxes 输出的位置索引，test_local 各行须与
    重算的框列表一一对齐，因此全链路（提框/锚点/模板/冷启动）必须共用本函数。
    """
    return f"morph{int(round(dilate_ratio * 100)):02d}"


# ── 运行配置 ──────────────────────────────────────────────────────
@dataclass
class RunConfig:
    dataset: str = "sroie"
    bbox_source: str = "gt"            # "gt" / "retrieval"
    max_new_tokens: int = 128
    n_tasks: int = 0                   # 0 = 全部
    visualize: bool = False
    resume_dir: str = ""               # 非空: 续跑指定目录（不新建时间戳目录）
    shard: str = ""                    # "i/n": 任务轮转分片（多卡并行，配合 resume_dir 使用）
    normal_from: str = ""              # 非空: 从已有 normal_results.json 复用基线
                                       # （贪心确定性已验证，扫参时省掉一半生成）
    bbox_json: str = ""                # 非空: 指定定位产物（如自举定位版），
                                       # 覆盖默认 localized_bboxes_{ds}.json
    run_tag: str = ""                  # 输出目录附加标记（区分同超参不同定位源）
    model: ModelConfig = field(default_factory=ModelConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)

    def __post_init__(self):
        # 每次运行新建带时间戳的输出目录，避免不同次运行的结果混在一起
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def output_dir(self) -> str:
        if self.resume_dir:
            return self.resume_dir
        tag = f"{self.bbox_source}_delta{self.intervention.delta:g}_layers-{self.intervention.target_layers}"
        if self.model.name != "qwen25vl":
            tag = f"{self.model.name}_{tag}"
        if self.intervention.use_confidence:
            tag += f"_confg{self.intervention.conf_gamma:g}"
            if self.intervention.conf_cutoff > 0:
                tag += f"c{self.intervention.conf_cutoff:g}"
        if self.run_tag:
            tag += f"_{self.run_tag}"
        return os.path.join(OUTPUT_DIR, self.dataset, "runs", f"{tag}_{self.run_id}")


def shard_tasks(tasks, i: int, n: int):
    """按样本分组轮转分片（同样本任务同片）。

    任务列表每样本连续 4 个实体，若用 tasks[i::n] 且 n 与实体数共振，
    每片会聚集单一实体（如 address 片生成最长、耗时远高于 total 片），
    造成多卡负载不均。按样本轮转可保证每片实体构成均匀。
    """
    groups, order = {}, []
    for t in tasks:
        if t.sample_name not in groups:
            groups[t.sample_name] = []
            order.append(t.sample_name)
        groups[t.sample_name].append(t)
    return [t for name in order[i::n] for t in groups[name]]


def parse_target_layers(spec: str, n_layers: int):
    """解析 target_layers 配置为层索引集合。

    支持: "all" / "22" / "20,21,22" / "19-27"（闭区间）/ 混合 "3,10-12"。
    """
    if spec == "all":
        return set(range(n_layers))
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            layers.update(range(int(lo), int(hi) + 1))
        else:
            layers.add(int(part))
    return layers
