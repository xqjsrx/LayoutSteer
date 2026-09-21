"""Post-merger embedding 提取器。

extract():       整模型前向 + merger hook（旧路径，保留兼容）
extract_batch(): 批量图像直接喂 visual tower（无 LLM 前向），按 image_grid_thw
                 切分 token 均值池化——大幅消除逐图开销。
"""
import torch


class PostMergerEmbedder:

    def __init__(self, model, processor, prompt: str):
        self.model = model
        self.processor = processor
        self.prompt = prompt
        self._outputs = []
        self._hook = model.model.visual.merger.register_forward_hook(
            lambda m, i, o: self._outputs.append(o.detach()))

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def extract(self, image):
        """单张图 -> L2 归一化 embedding (np.ndarray [D])，走整模型前向。"""
        self._outputs.clear()
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": self.prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True,
                                return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self.model(**inputs)
        raw = self._outputs[-1]
        if raw.dim() == 3:
            raw = raw[0]
        emb = raw.mean(dim=0)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.float().cpu().numpy().flatten()

    def extract_batch(self, images):
        """批量图像 -> [np.ndarray [D]]，只过 visual tower（数值与 extract 等价）。

        Qwen2.5-VL 视觉塔将多图 token 扁平拼接，按 image_grid_thw 计算
        各图 token 数切片后分别均值池化。
        """
        # 本路径不读 hook 输出，清空防止 merger hook 逐批累积输出
        # （hook 持有的引用会拦住 merger 输出 storage 的释放，全量跑时线性涨显存）
        self._outputs.clear()
        pvs, grids = [], []
        for im in images:
            out = self.processor.image_processor(images=[im])
            pvs.append(out["pixel_values"])
            grids.append(out["image_grid_thw"])
        pv = torch.cat(pvs).to(self.model.device, torch.bfloat16)
        grid = torch.cat(grids).to(self.model.device)

        with torch.no_grad():
            hidden = self.model.visual(pv, grid_thw=grid)  # (sum_tokens, D)
        if isinstance(hidden, tuple):   # Qwen3-VL: (hidden, deepstack_list)
            hidden = hidden[0]

        # spatial_merge_size^2 = 4: 每图 token 数 = H*W/4
        counts = [(int(g[1]) * int(g[2])) // 4 for g in grid]
        embs = []
        offset = 0
        for n in counts:
            e = hidden[offset:offset + n].mean(dim=0)
            e = torch.nn.functional.normalize(e, dim=-1)
            embs.append(e.float().cpu().numpy().flatten())
            offset += n
        return embs
