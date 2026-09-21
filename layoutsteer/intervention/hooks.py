"""Tier B 钩子: 自定义注意力类（不走 ALL_ATTENTION_FUNCTIONS 分发）的 ΔK 干预。

机制（以 LLaVA-OV-1.5 为代表的旧式 *_ATTENTION_CLASSES 模型）:
  1. 每个解码层 self_attn 挂 forward_pre_hook 记录当前层号与相位
  2. 替换远程模块命名空间里的 `_flash_attention_forward` 符号——
     该函数收到的 K 是 post-RoPE / post-cache / repeat_kv 后的
     [B, seq, num_heads, head_dim]，转置到规范布局后复用
     ScoreDeltaProbe._apply_score_delta 的伪逆求解（num_kv==num_heads,
     groups=1 退化为 dK = δ·√d·q/|q|²）。

与 Tier A 的 ScoreDeltaProbe 共享全部干预语义（δ 序列/层集合/相位门控/
persistent 不支持——Tier B 模型 cache 编辑语义不同，暂不提供）。
"""
from contextlib import contextmanager

import torch

from .probe import ScoreDeltaProbe


class ClassPatchProbe(ScoreDeltaProbe):
    """通过模块级符号替换注入 ΔK 的探针。

    Args:
        model, processor: 同 ScoreDeltaProbe
        attn_modules: 解码层注意力模块列表（须带 layer_idx 属性）
        patch_host: 持有 `_flash_attention_forward` 符号的远程模块对象
        symbol: 被替换的符号名
    """

    def __init__(self, model, processor, attn_modules, patch_host,
                 symbol: str = "_flash_attention_forward",
                 capture_layer: int = 22):
        super().__init__(model, processor, capture_layer=capture_layer)
        self._attn_modules = attn_modules
        self._patch_host = patch_host
        self._symbol = symbol
        self._orig_fn = None
        self._cur_layer = None
        self._pre_hooks = []

    # ── 安装 / 卸载 ──────────────────────────────────────────────
    def attach(self, delta, target_layers=None, capture: bool = False,
               target_heads=None, steer_prefill=True, max_steer_steps=None,
               persistent: bool = False):
        assert not persistent, "Tier B 暂不支持 persistent 模式"
        # 复用父类的 δ/层集合/head 掩码解析，但不安装 FA2 分发钩子
        self._reset_state()
        self._parse_attach_args(delta, target_layers, target_heads,
                                steer_prefill, max_steer_steps, persistent)
        self._capture = capture

        for m in self._attn_modules:
            self._pre_hooks.append(
                m.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        self._orig_fn = getattr(self._patch_host, self._symbol)
        setattr(self._patch_host, self._symbol, self._patched_fn)
        self._installed = True
        return self._ctx()

    def detach(self):
        if self._installed:
            for h in self._pre_hooks:
                h.remove()
            self._pre_hooks = []
            if self._orig_fn is not None:
                setattr(self._patch_host, self._symbol, self._orig_fn)
                self._orig_fn = None
            self._installed = False

    # ── 层号与相位跟踪 ───────────────────────────────────────────
    def _pre_hook(self, module, args, kwargs):
        self._cur_layer = module.layer_idx
        if module.layer_idx == 0:
            hs = kwargs.get("hidden_states",
                            args[0] if args else None)
            if hs is not None:
                if hs.shape[1] == 1:
                    self._decode_step += 1
                else:
                    self._decode_step = 0
        return None

    # ── 被替换的 _flash_attention_forward ────────────────────────
    def _patched_fn(self, query_states, key_states, value_states,
                    attention_mask, query_length, *args, **kwargs):
        layer_idx = self._cur_layer
        # [B, seq, H, d] -> 规范布局 [B, H, seq, d]
        if self._delta_active and layer_idx in self._target_layers:
            q = query_states.transpose(1, 2)
            k = key_states.transpose(1, 2)
            k = self._apply_score_delta(q, k, layer_idx)
            key_states = k.transpose(1, 2)

        if self._capture and layer_idx == self.capture_layer:
            self._record(query_states.transpose(1, 2),
                         key_states.transpose(1, 2))

        return self._orig_fn(query_states, key_states, value_states,
                             attention_mask, query_length, *args, **kwargs)


class VendorHookProbe(ScoreDeltaProbe):
    """vendor 副本的显式回调探针。

    vendor 代码在 attention forward 的 cache update 后调用
    `_LS_PROBE.maybe_edit(q, k, layer_idx)`（规范布局 [B, KV, seq, d]），
    attach/detach 仅设置/清除宿主模块的 _LS_PROBE 全局。
    """

    def __init__(self, model, processor, host_module, capture_layer: int = 22):
        super().__init__(model, processor, capture_layer=capture_layer)
        self._host = host_module

    def attach(self, delta, target_layers=None, capture: bool = False,
               target_heads=None, steer_prefill=True, max_steer_steps=None,
               persistent: bool = False):
        assert not persistent, "vendor 探针暂不支持 persistent 模式"
        self._reset_state()
        self._parse_attach_args(delta, target_layers, target_heads,
                                steer_prefill, max_steer_steps, persistent)
        self._capture = capture
        setattr(self._host, "_LS_PROBE", self)
        self._installed = True
        return self._ctx()

    def detach(self):
        if self._installed:
            setattr(self._host, "_LS_PROBE", None)
            self._installed = False

    def maybe_edit(self, query_states, key_states, layer_idx):
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
        return key_states


class ScoreAddProbe(VendorHookProbe):
    """Tier C: eager 手写注意力上直接给目标 token 的 score 列加 δ。

    与在 K 上加 ΔK（令 pre-softmax score 精确变化 δ）在生成位置等价，
    但无需伪逆求解，也适用于没有独立可改 K 张量的手写实现
    （DocOwl2 的 MAM 多路 K/V）。远程代码在 softmax 前调用
    maybe_add_scores(attn_weights, layer_idx)。
    """

    def _reset_state(self):
        super()._reset_state()
        self._captured_scores = []

    def maybe_add_scores(self, attn_weights, layer_idx):
        if layer_idx == 0:
            if attn_weights.shape[2] == 1:
                self._decode_step += 1
            else:
                self._decode_step = 0
        need_capture = self._capture and layer_idx == self.capture_layer
        need_edit = self._delta_active and layer_idx in self._target_layers
        if not need_edit:
            if need_capture:
                self._captured_scores.append(
                    attn_weights[:, :, -1:, :].detach().clone())
            return attn_weights
        kv_len = attn_weights.shape[-1]
        idx, w = self._target_seq_indices(kv_len, attn_weights.device)
        if idx is None:
            return attn_weights
        B = attn_weights.shape[0]
        deltas = self._delta if isinstance(self._delta, list) \
            else [self._delta] * B
        if self._layer_sets is not None:
            deltas = [d if layer_idx in ls else 0.0
                      for d, ls in zip(deltas, self._layer_sets)]
        if attn_weights.shape[2] > 1:          # prefill
            sp = self._steer_prefill
            sp = sp if isinstance(sp, list) else [sp] * B
            deltas = [d if f else 0.0 for d, f in zip(deltas, sp)]
        else:                                   # decode
            ms = self._max_steps
            ms = ms if isinstance(ms, list) else [ms] * B
            deltas = [d if (m is None or self._decode_step <= m) else 0.0
                      for d, m in zip(deltas, ms)]
        if not any(deltas):
            return attn_weights
        attn_weights = attn_weights.clone()
        for b in range(B):
            if deltas[b] == 0.0:
                continue
            add = deltas[b] if w is None else deltas[b] * w
            attn_weights[b, :, :, idx] += torch.as_tensor(
                add, dtype=attn_weights.dtype, device=attn_weights.device)
        if need_capture:
            self._captured_scores.append(
                attn_weights[:, :, -1:, :].detach().clone())
        return attn_weights
