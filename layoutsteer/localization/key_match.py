"""软键匹配：开放词表下把"精确同键"松弛为"语义相近键"。

硬匹配（τ=1）是软匹配的特例——闭集实体名恰好总是精确命中，行为不变；
开放词表（funsd 表单字段名）下 FAX NOc.: / FAX: / Sender Fax Number 这类
同字段异写会被归并，模板池覆盖率从 57.6% 升到约 88%。

相似度用归一化串（仅 [a-z0-9]）的 SequenceMatcher ratio：它按长度归一，
天然拒绝"单个泛化词吞掉长键"的错误归并（Area: ↛ Glue Pree Area，
Confidentiality Note ↛ NOTE:），而保留真正的同字段异写。
匹配到的非精确键作为模板时票权按相似度打折，精确键始终权重 1。
"""
import re
from difflib import SequenceMatcher


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


class KeyMatcher:
    """键 -> [(候选键, 相似度)] 降序，精确匹配在首位。tau<=0 时退化为硬匹配。"""

    def __init__(self, pool_keys, tau=0.0, max_alt=5):
        self.tau = tau
        self.max_alt = max_alt
        self.pool = {k: _norm(k) for k in pool_keys}
        self._cache = {}

    def candidates(self, key):
        if self.tau <= 0:
            return [(key, 1.0)]
        if key not in self._cache:
            kn = _norm(key)
            alts = []
            for pk, pn in self.pool.items():
                if pk == key:
                    continue
                s = SequenceMatcher(None, kn, pn).ratio()
                if s >= self.tau:
                    alts.append((pk, s))
            alts.sort(key=lambda x: -x[1])
            self._cache[key] = [(key, 1.0)] + alts[:self.max_alt]
        return self._cache[key]
