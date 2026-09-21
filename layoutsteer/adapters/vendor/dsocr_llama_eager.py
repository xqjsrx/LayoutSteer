"""DeepSeek-OCR 语言侧 eager 注意力补丁（transformers==4.46.x 专用）。

对 stock LlamaAttention.forward 取源插桩: 在 past_key_value.update 之后
插入 _LS_PROBE.maybe_edit 回调（post-RoPE / post-cache, [B, KV, seq, d]，
MHA 下 KV==heads, 伪逆退化为 dK = δ·√d·q/|q|²），再整类替换。

返回 hook host（VendorHookProbe.attach 对其 setattr("_LS_PROBE", probe)
即写入补丁函数的 globals，实现零侵入的动态开关）。
"""
import inspect
import textwrap


class _HookHost:
    """把 setattr 转写进补丁函数 globals 的桥接对象。"""

    def __init__(self, g):
        object.__setattr__(self, "_g", g)

    def __setattr__(self, key, value):
        self._g[key] = value

    def __getattr__(self, key):
        try:
            return self._g[key]
        except KeyError:
            raise AttributeError(key)


ANCHOR = ("        key_states, value_states = past_key_value.update("
          "key_states, value_states, self.layer_idx, cache_kwargs)\n")
HOOK = ANCHOR + (
    "    if _LS_PROBE is not None:\n"
    "        key_states = _LS_PROBE.maybe_edit("
    "query_states, key_states, self.layer_idx)\n")


def patch_llama_eager_forward():
    import transformers
    assert transformers.__version__.startswith("4.46"), \
        f"补丁按 4.46 源码锚定，当前 {transformers.__version__}"
    from transformers.models.llama import modeling_llama as ml

    src = textwrap.dedent(inspect.getsource(ml.LlamaAttention.forward))
    assert ANCHOR in src, "4.46 LlamaAttention.forward 源码锚点未命中"
    src = src.replace(ANCHOR, HOOK, 1)

    g = dict(vars(ml))
    g["_LS_PROBE"] = None
    exec(compile(src, "<dsocr_patched_llama_forward>", "exec"), g)
    ml.LlamaAttention.forward = g["forward"]
    return _HookHost(g)
