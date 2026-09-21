"""ScoreDeltaProbe: Flash Attention 2 下的 score_delta K 向量干预。

原理（继承自原型 prototypes/k_intervention.py，仅保留 score_delta 模式）:
  1. 替换 ALL_ATTENTION_FUNCTIONS["flash_attention_2"] 为 wrapper，
     在 FA2 分发入口拦截 post-RoPE / post-cache 的 Q/K。
  2. 对目标层的 K[target] += ΔK，其中 ΔK = δ·√d·Q⁺·𝟙 由伪逆求解，
     使目标 token 的 pre-softmax score 精确变化 δ（对生成位置精确成立）。
     δ > 0 增强注意力，δ < 0 抑制。
  3. 可选：在 capture_layer 捕获 Q/K 用于注意力热图可视化。

兼容模型: Qwen2.5-VL（ALL_ATTENTION_FUNCTIONS 分发机制）。
"""
import torch
import torch.nn.functional as F
import numpy as np
from contextlib import contextmanager
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:   # transformers<4.48（DeepSeek-OCR env）无分发注册表
    ALL_ATTENTION_FUNCTIONS = None

VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
SPATIAL_MERGE_SIZE = 2


class ScoreDeltaProbe:
    """score_delta K 向量干预探针（FA2 兼容）。

    用法:
        probe = ScoreDeltaProbe(model, processor)
        probe.setup_image_range(input_ids, image_grid_thw)
        probe.set_region(grid_indices)          # None = 全部图像 token
        with probe.attach(delta=5.0, target_layers={...}):
            out = model.generate(**inputs)
    """

    def __init__(self, model, processor, capture_layer: int = 22):
        self.model = model
        self.processor = processor
        self.capture_layer = capture_layer

        cfg = model.config
        text_config = getattr(cfg, "text_config", None) \
            or getattr(cfg, "llm_config", None) or cfg
        self.n_layers = getattr(text_config, "num_hidden_layers", 0)

        self._original_fa2 = None
        self._installed = False

        self._img_range = None       # (img_start, img_end) prefill 坐标
        self._spatial_shape = None   # (h, w)
        self._region_grid_idx = None # 网格索引 (None = 全部图像 token)
        self._region_weights = None  # 逐 token 干预权重 (None = 均匀 1.0)
        self._abs_idx = None         # 绝对序列索引（adapter 直接指定时）
        self._abs_weights = None

        self._delta = 0.0
        self._delta_active = False
        self._target_layers = set()
        self._capture = False
        self._reset_state()

    def _reset_state(self):
        self._k_list = []
        self._k_lengths = []
        self._captured_q = []
        self._step = 0
        self._decode_step = 0

    # ── 图像范围与干预区域 ────────────────────────────────────────
    def setup_image_range(self, input_ids, image_grid_thw,
                          spatial_merge_size: int = SPATIAL_MERGE_SIZE):
        """定位图像 token 序列范围与空间网格。attach 前必须调用。

        spatial_merge_size: 视觉塔空间合并倍率（Qwen2.5/3-VL 为 2；
        MonkeyOCRv2 等无合并模型传 1，此时 token 数 = grid_thw h*w）。
        """
        ids = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
        tokenizer = self.processor.tokenizer
        vs_id = tokenizer.convert_tokens_to_ids(VISION_START_TOKEN)
        ve_id = tokenizer.convert_tokens_to_ids(VISION_END_TOKEN)
        self._img_range = (ids.index(vs_id) + 1, ids.index(ve_id))
        h = int(image_grid_thw[0, 1].item()) // spatial_merge_size
        w = int(image_grid_thw[0, 2].item()) // spatial_merge_size
        self._spatial_shape = (h, w)
        n_img = self._img_range[1] - self._img_range[0]
        assert n_img == h * w, f"图像 token 数 {n_img} 与网格 {h}x{w} 不符"

    def set_region(self, grid_indices, weights=None):
        """设置干预区域（行主序网格索引 LongTensor），None 表示全部图像 token。

        weights: 与 grid_indices 等长的 FloatTensor，逐 token 干预强度系数
        （布局掩码模式：δ_i = δ·w_i）；None = 区域内均匀 δ。
        """
        self._abs_idx = None
        self._abs_weights = None
        self._region_grid_idx = grid_indices
        self._region_weights = weights

    def set_target_indices(self, seq_indices, weights=None):
        """直接设置干预目标的绝对序列索引（多段/非连续图像 token 模型用）。

        与 setup_image_range + set_region 互斥: 走此路径时 bbox->token
        换算完全由 adapter 完成，probe 不再做网格偏移。
        """
        self._abs_idx = seq_indices
        self._abs_weights = weights
        self._region_grid_idx = None
        self._region_weights = None

    @property
    def spatial_shape(self):
        return self._spatial_shape

    # ── 安装 / 卸载 ──────────────────────────────────────────────
    def attach(self, delta, target_layers=None, capture: bool = False,
               target_heads=None, steer_prefill=True, max_steer_steps=None,
               persistent: bool = False):
        """安装 wrapper，返回 context manager。

        Args:
            delta: score 增量（0 表示仅捕获不干预）。
                标量 = 全 batch 同一强度；序列 = 逐 batch 元素独立强度
                （同一任务一次生成内并行扫多个 δ）。
            target_layers: 干预层索引集合（None = 全部层）
            capture: 是否在 capture_layer 记录 Q/K 用于可视化
            target_heads: (layer, KV head) 级掩码。dict{layer: set(kv_head)}
                全 batch 共享，或 list[dict] 逐 batch 元素（需 delta 等长）。
                提供时干预层自动取掩码中出现的层，target_layers 被忽略。
            steer_prefill: prefill 阶段是否干预（bool 或逐元素 list[bool]）。
            max_steer_steps: 仅干预前 k 个 decode 步（None=不限；0=仅 prefill；
                int 或逐元素 list）。机理依据: 修复分岔在答案开头，退化分岔
                在中段，早撤除可保留翻转、削减转写期扰动。
            persistent: 一次性持久干预（KV cache 编辑）。prefill 时 ΔK
                直接就地写入 cache 存储（DynamicCache.update 返回的就是
                存储本体，后续 cat 按值拷贝传播编辑），decode 全程 no-op。
                与 per-step 模式的区别仅在于 clone 与否。
        """
        self._reset_state()
        self._parse_attach_args(delta, target_layers, target_heads,
                                steer_prefill, max_steer_steps, persistent)
        self._capture = capture

        assert ALL_ATTENTION_FUNCTIONS is not None, \
            "transformers<4.48 无 FA2 分发注册表，须用 Tier B/C 探针"
        self._original_fa2 = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = self._wrapper
        self._installed = True
        return self._ctx()

    def _parse_attach_args(self, delta, target_layers, target_heads,
                           steer_prefill, max_steer_steps, persistent):
        """attach 参数解析（Tier A/B 驱动共享）。"""
        if isinstance(delta, (list, tuple, np.ndarray)):
            self._delta = [float(d) for d in delta]
            self._delta_active = any(d != 0.0 for d in self._delta)
        else:
            self._delta = float(delta)
            self._delta_active = self._delta != 0.0
        # target_layers: 层索引集合（全 batch 共享），或 list[集合]（逐 batch 元素，
        # 需 delta 也为等长序列，用于一次生成内并行扫多组层配置）
        self._layer_sets = None
        if target_layers is None:
            self._target_layers = set(range(self.n_layers))
        elif isinstance(target_layers, (list, tuple)) and target_layers \
                and isinstance(target_layers[0], (set, frozenset, list, tuple)):
            self._layer_sets = [set(s) for s in target_layers]
            self._target_layers = set().union(*self._layer_sets)
            assert isinstance(self._delta, list) and \
                len(self._delta) == len(self._layer_sets), \
                "逐元素层配置需 delta 与 target_layers 等长"
        else:
            self._target_layers = set(target_layers)
        # 混合注意力模型（如 Qwen3.5: 24 层线性注意力无可编辑 KV）:
        # 与可干预层求交, 交集为空则显式报错而非静默失效
        allowed = getattr(self, "intervenable_layers", None)
        if allowed is not None:
            self._target_layers &= set(allowed)
            if self._layer_sets is not None:
                self._layer_sets = [s & set(allowed) for s in self._layer_sets]
            assert self._target_layers or not self._delta_active, \
                "指定层带与可干预层（full_attention）无交集"
        # head 掩码: 提供时干预层 = 掩码层集合，逐 KV head 精确控制
        self._head_masks = None
        if target_heads is not None:
            if isinstance(target_heads, dict):
                masks = [target_heads] * (len(self._delta)
                                          if isinstance(self._delta, list) else 1)
            else:
                masks = list(target_heads)
                assert isinstance(self._delta, list) and \
                    len(self._delta) == len(masks), \
                    "逐元素 head 掩码需 delta 等长"
            self._head_masks = masks
            self._layer_sets = [set(m.keys()) for m in masks]
            self._target_layers = set().union(*self._layer_sets)
        self._steer_prefill = steer_prefill
        self._max_steps = max_steer_steps
        self._persistent = persistent

    @contextmanager
    def _ctx(self):
        try:
            yield self
        finally:
            self.detach()

    def detach(self):
        if self._installed and self._original_fa2 is not None:
            ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = self._original_fa2
            self._original_fa2 = None
            self._installed = False

    # ── 核心 wrapper ─────────────────────────────────────────────
    def _wrapper(self, module, query_states, key_states, value_states,
                 attention_mask=None, **kwargs):
        layer_idx = getattr(module, "layer_idx", None)

        # decode 步计数（每个 forward 在层 0 更新一次）
        if layer_idx == 0:
            if query_states.shape[2] == 1:
                self._decode_step += 1
            else:
                self._decode_step = 0

        if self._delta_active and layer_idx in self._target_layers:
            key_states = self._apply_score_delta(
                query_states, key_states, layer_idx)

        if self._capture and layer_idx == self.capture_layer:
            self._record(query_states, key_states)

        return self._original_fa2(
            module, query_states, key_states, value_states,
            attention_mask=attention_mask, **kwargs)

    # ── score_delta 干预 ─────────────────────────────────────────
    def _target_seq_indices(self, kv_len, device):
        """干预目标在 KV 序列中的索引及逐 token 权重 (idx, w|None)。"""
        if self._abs_idx is not None:
            idx = self._abs_idx.to(device)
            keep = idx < kv_len
            idx = idx[keep]
            if len(idx) == 0:
                return None, None
            w = None
            if self._abs_weights is not None:
                w = self._abs_weights.to(device=device,
                                         dtype=torch.float32)[keep]
            return idx, w
        if self._img_range is None:
            return None, None
        img_start, img_end = self._img_range
        if kv_len <= img_start:
            return None, None
        if self._region_grid_idx is None:
            end = min(img_end, kv_len)
            return torch.arange(img_start, end, device=device,
                                dtype=torch.long), None
        seq_idx = img_start + self._region_grid_idx.to(device)
        keep = seq_idx < min(img_end, kv_len)
        seq_idx = seq_idx[keep]
        if len(seq_idx) == 0:
            return None, None
        w = None
        if self._region_weights is not None:
            w = self._region_weights.to(device=device,
                                        dtype=torch.float32)[keep]
        return seq_idx, w

    def _apply_score_delta(self, query_states, key_states, layer_idx=None):
        """K[target] += ΔK，ΔK = δ·√d·Q⁺·𝟙（逐 batch 元素、逐 KV head 伪逆求解）。

        每步都基于当前生成位置的 Q 重新求解，K cache 本身不被污染
        （在 clone 上修改）。δ 为序列时逐 batch 元素独立取值；
        配置了逐元素层集合时，不包含当前层的元素 δ 视为 0。
        """
        target_idx, region_w = self._target_seq_indices(
            key_states.shape[2], key_states.device)
        if target_idx is None:
            return key_states

        # 持久模式: ΔK 已在 prefill 时写入 cache，decode 无需任何操作
        if self._persistent and query_states.shape[2] == 1:
            return key_states

        B = key_states.shape[0]
        if isinstance(self._delta, list):
            deltas = list(self._delta)
            assert len(deltas) == B, f"delta 向量长度 {len(deltas)} != batch {B}"
        else:
            deltas = [self._delta] * B
        if self._layer_sets is not None:
            deltas = [d if layer_idx in ls else 0.0
                      for d, ls in zip(deltas, self._layer_sets)]
        # 相位/步数门控: prefill 开关 + decode 步数上限（均支持逐元素）
        if query_states.shape[2] > 1:  # prefill
            sp = self._steer_prefill
            sp = sp if isinstance(sp, list) else [sp] * B
            deltas = [d if f else 0.0 for d, f in zip(deltas, sp)]
        else:                          # decode 第 self._decode_step 步（从 1 计）
            ms = self._max_steps
            ms = ms if isinstance(ms, list) else [ms] * B
            deltas = [d if (m is None or self._decode_step <= m) else 0.0
                      for d, m in zip(deltas, ms)]
        if not any(deltas):
            return key_states
        head_masks = self._head_masks
        if head_masks is not None and len(head_masks) == 1 and B > 1:
            head_masks = head_masks * B

        if not self._persistent:
            key_states = key_states.clone()
        # persistent: 不 clone，就地写——update() 返回的就是 cache 存储本体，
        # 编辑随后续 decode 步的 cat 按值传播，实现一次干预全程生效
        num_heads = query_states.shape[1]
        num_kv = key_states.shape[1]
        groups = num_heads // num_kv
        head_dim = query_states.shape[-1]
        sqrt_d = head_dim ** 0.5

        q = query_states.float()
        # autocast 下 matmul 会被降精度（DeepSeek-OCR 的 generate 整体包在
        # autocast(bf16) 里），而 linalg.inv 不支持低精度 -> 局部关断
        with torch.autocast("cuda", enabled=False):
            for b in range(B):
                if deltas[b] == 0.0:
                    continue
                for h in range(num_kv):
                    # head 掩码: 仅干预选中的 (layer, KV head)
                    if head_masks is not None and \
                            h not in head_masks[b].get(layer_idx, ()):
                        continue
                    # 该 KV head 对应的 Q heads，取最后一个 position（生成位置）
                    q_h = q[b, h * groups:(h + 1) * groups, -1, :]     # [groups, head_dim]
                    # 解 Q_h · ΔK^T = δ·√d·𝟙: 伪逆 Q_h⁺ = Q_h^T·(Q_h·Q_h^T)^{-1}
                    QQt = q_h @ q_h.T
                    QQt += 1e-6 * torch.eye(groups, device=q.device)
                    Q_pinv = q_h.T @ torch.linalg.inv(QQt)             # [head_dim, groups]
                    dK = deltas[b] * sqrt_d * (Q_pinv @ torch.ones(groups, device=q.device))
                    dK = dK.to(key_states.dtype)
                    if region_w is not None:
                        # 布局掩码: 逐 token 强度 δ_i = δ·w_i（外积）
                        key_states[b, h, target_idx, :] += \
                            region_w.to(key_states.dtype).unsqueeze(1) * dK.unsqueeze(0)
                    else:
                        key_states[b, h, target_idx, :] += dK

        return key_states

    # ── 捕获与注意力图计算（可视化用） ────────────────────────────
    def _record(self, query_states, key_states):
        # 注意: 必须 clone。decode 阶段 wrapper 收到的 key_states 是完整
        # KV cache，切片 + detach 仍是视图，会引用整段 cache storage，
        # 导致 DynamicCache 每步 cat 产生的旧 storage 无法释放（线性泄漏）。
        if self._step == 0:
            self._k_list.append(key_states.detach().clone())
            self._k_lengths.append(key_states.shape[2])
            self._captured_q.append(
                query_states[:, :, -1:, :].detach().clone())
        else:
            self._k_list.append(key_states[:, :, -1:, :].detach().clone())
            self._k_lengths.append(self._k_list[0].shape[2] + self._step)
            self._captured_q.append(query_states.detach().clone())
        self._step += 1

    def get_attention_maps(self):
        """由捕获的 Q/K 手动计算逐 token 图像注意力图 list[np.ndarray(h, w)]。"""
        if not self._captured_q or self._img_range is None:
            return []

        img_start, img_end = self._img_range
        h, w = self._spatial_shape

        k_full = torch.cat(self._k_list, dim=2).to(torch.float32)
        num_heads = self._captured_q[0].shape[1]
        num_kv_groups = num_heads // k_full.shape[1]
        if num_kv_groups > 1:
            k_full = k_full.repeat_interleave(num_kv_groups, dim=1)
        scale = k_full.shape[-1] ** -0.5

        att_maps = []
        for q, k_len in zip(self._captured_q, self._k_lengths):
            scores = torch.matmul(q.to(torch.float32),
                                  k_full[:, :, :k_len, :].transpose(-2, -1)) * scale
            attn = F.softmax(scores, dim=-1)
            img_attn = attn[0, :, 0, img_start:img_end].mean(0)
            att_maps.append(img_attn.cpu().numpy().reshape(h, w))

        del k_full
        torch.cuda.empty_cache()
        return att_maps

    def image_attention_stats(self):
        """图像注意力统计（total/max/entropy 逐 token）。"""
        stats = []
        for m in self.get_attention_maps():
            total = float(m.sum())
            p = m / (total + 1e-8)
            stats.append({
                "total": total,
                "max": float(m.max()),
                "entropy": float(-np.sum(p * np.log(p + 1e-8))),
            })
        return stats
