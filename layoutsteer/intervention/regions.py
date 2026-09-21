"""bbox -> 图像 token 网格索引映射。

Qwen2.5-VL 的图像 token 按行主序排列在 <|vision_start|> 与 <|vision_end|> 之间，
网格尺寸为 image_grid_thw // spatial_merge_size（精确值，不再手推 672 网格）。
"""
import torch


def grid_shape_from_inputs(inputs, spatial_merge_size: int = 2):
    """从 processor 输出的 image_grid_thw 得到 (h, w) 网格尺寸。"""
    thw = inputs["image_grid_thw"]
    h = int(thw[0, 1].item()) // spatial_merge_size
    w = int(thw[0, 2].item()) // spatial_merge_size
    return h, w


def bboxes_px_to_rel(bboxes_px, image_size):
    """像素坐标框 -> 相对坐标框（0~1）。image_size 为缩放前的原始 (W, H)。"""
    W, H = image_size
    return [[x1 / W, y1 / H, x2 / W, y2 / H] for x1, y1, x2, y2 in bboxes_px]


def bboxes_to_grid_indices(bboxes_rel, grid_hw, device="cpu"):
    """多个相对坐标框 -> 行主序网格索引并集 (LongTensor)。

    与旧版 FocusedSoftIntervener 的 bbox 奖励图语义一致：
    框内（含边界格）的所有 token 均被干预。
    """
    h, w = grid_hw
    mask = torch.zeros(h, w, dtype=torch.bool)
    for x1, y1, x2, y2 in bboxes_rel:
        c1 = max(0, min(int(x1 * w), w - 1))
        r1 = max(0, min(int(y1 * h), h - 1))
        c2 = max(0, min(int(x2 * w), w - 1))
        r2 = max(0, min(int(y2 * h), h - 1))
        mask[r1:r2 + 1, c1:c2 + 1] = True
    grid_idx = mask.flatten().nonzero(as_tuple=True)[0]
    return grid_idx.to(device)


def _gaussian_blur2d(grid, sigma):
    """可分离高斯模糊（纯 torch，掩码羽化用）。"""
    if sigma <= 0:
        return grid
    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    g = grid.unsqueeze(0).unsqueeze(0)
    g = torch.nn.functional.conv2d(
        g, kernel.view(1, 1, 1, -1), padding=(0, radius))
    g = torch.nn.functional.conv2d(
        g, kernel.view(1, 1, -1, 1), padding=(radius, 0))
    return g[0, 0]


def weight_map_to_grid(weight_map, image_size, grid_hw,
                       mode="votes", feather_sigma=1.0):
    """空间置信场（得票框+权重列表）-> [h, w] float 布局掩码（max=1）。

    mode="votes":   每个得票框按归一化投票权重填充所覆盖格子（max 聚合），
                    再羽化——干预强度逐点正比于定位置信的空间分布
    mode="feather": 仅 in_region 框按 1.0 填充 + 羽化（≈现有 bbox 软边版）
    """
    W, H = image_size
    h, w = grid_hw
    grid = torch.zeros(h, w, dtype=torch.float32)
    for entry in weight_map:
        if mode == "feather" and not entry.get("in_region", False):
            continue
        weight = 1.0 if mode == "feather" else float(entry["weight"])
        x1, y1, x2, y2 = entry["box"]
        c1 = max(0, min(int(x1 / W * w), w - 1))
        r1 = max(0, min(int(y1 / H * h), h - 1))
        c2 = max(0, min(int(x2 / W * w), w - 1))
        r2 = max(0, min(int(y2 / H * h), h - 1))
        grid[r1:r2 + 1, c1:c2 + 1] = torch.maximum(
            grid[r1:r2 + 1, c1:c2 + 1], torch.tensor(weight))
    # 羽化仅向外扩软边，不稀释核心强度（否则等效整体降 δ，干预剂量泄漏）
    grid = torch.maximum(grid, _gaussian_blur2d(grid, feather_sigma))
    if grid.max() > 0:
        grid = grid / grid.max()
    return grid


def layout_mask_grid(region_boxes_px, layout_boxes_px, image_size, grid_hw,
                     bg_weight=0.1, feather_sigma=0.6):
    """Layout 底纹掩码: 目标区域矩形满剂量 + 全文档文本行低权底纹。

    干预核心与 bbox 均匀模式完全一致（区域内 w=1，消融已证明满剂量矩形
    最优）；底纹以 bg_weight 描出版面结构，使掩码本身呈现文档 layout。
    """
    W, H = image_size
    h, w = grid_hw
    grid = torch.zeros(h, w, dtype=torch.float32)

    def fill(box, weight):
        x1, y1, x2, y2 = box
        c1 = max(0, min(int(x1 / W * w), w - 1))
        r1 = max(0, min(int(y1 / H * h), h - 1))
        c2 = max(0, min(int(x2 / W * w), w - 1))
        r2 = max(0, min(int(y2 / H * h), h - 1))
        grid[r1:r2 + 1, c1:c2 + 1] = torch.maximum(
            grid[r1:r2 + 1, c1:c2 + 1], torch.tensor(weight))

    for box in layout_boxes_px:
        fill(box, bg_weight)
    if feather_sigma > 0:  # 底纹轻度柔化，版面观感更连贯
        grid = torch.maximum(grid, _gaussian_blur2d(grid, feather_sigma))
    for box in region_boxes_px:
        fill(box, 1.0)   # 核心矩形最后填，确保满剂量不被任何处理稀释
    return grid


def structured_mask_grid(region_boxes_px, weight_map, layout_boxes_px,
                         image_size, grid_hw, bg_near=0.4, bg_far=0.03,
                         prox_sigma=5.0, halo_sigma=2.0, halo_gain=0.65,
                         vote_gain=0.5, core_gap_weight=1.0):
    """分层结构掩码（论文展示形态 D）: 四层 max 叠加。

    L1 核心           : 簇内文本框 w=1.0，外接矩形间隙 w=core_gap_weight
                         （剂量消融铁律: gap=1.0 最优 93.52%，稀释即掉分）
    L2 光晕           : 核心区高斯扩散×halo_gain，从矩形边界向外递减
    L3 票权热斑       : 得票框按归一化票权×vote_gain 填充+羽化
    L4 距离调制底纹   : 文本行强度 = bg_far + (bg_near-bg_far)·P，
                         P 为核心区宽尺度（prox_sigma）高斯邻近场——
                         版面底纹以目标为中心向外连续变淡（全局明暗梯度）

    矩形外剂量均围绕目标递减（已验证小增益）。
    """
    W, H = image_size
    h, w = grid_hw

    def rect(box):
        x1, y1, x2, y2 = box
        c1 = max(0, min(int(x1 / W * w), w - 1))
        r1 = max(0, min(int(y1 / H * h), h - 1))
        c2 = max(0, min(int(x2 / W * w), w - 1))
        r2 = max(0, min(int(y2 / H * h), h - 1))
        return r1, r2, c1, c2

    # L1 核心: 间隙打底 + 簇内文本框满剂量（结构轮廓）
    core = torch.zeros(h, w, dtype=torch.float32)
    for box in region_boxes_px:
        r1, r2, c1, c2 = rect(box)
        core[r1:r2 + 1, c1:c2 + 1] = core_gap_weight
    for entry in weight_map or []:
        if not entry.get("in_region", False):
            continue
        r1, r2, c1, c2 = rect(entry["box"])
        core[r1:r2 + 1, c1:c2 + 1] = 1.0
    # L2 光晕（仅取核心外的扩散部分）
    halo = _gaussian_blur2d(core, halo_sigma)
    halo = halo / max(float(halo.max()), 1e-6) * halo_gain
    # L3 票权热斑
    votes = torch.zeros(h, w, dtype=torch.float32)
    for entry in weight_map or []:
        r1, r2, c1, c2 = rect(entry["box"])
        votes[r1:r2 + 1, c1:c2 + 1] = torch.maximum(
            votes[r1:r2 + 1, c1:c2 + 1],
            torch.tensor(float(entry["weight"]) * vote_gain))
    votes = torch.maximum(votes, _gaussian_blur2d(votes, 1.0))
    # L4 距离调制底纹: 邻近场 P 基于到最近核心格的距离变换。
    # 不用高斯扩散：扩散场线性可加且强度随核心面积增长，多区域/大区域
    # 时光晕叠加合并成大平台，衰减不可见；距离变换天然 max 合成，
    # 衰减剖面与区域数量/面积无关。
    core_cells = core.nonzero().float()                      # [N, 2]
    if len(core_cells) > 0:
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w),
                                indexing="ij")
        cells = torch.stack([ys.flatten(), xs.flatten()], 1).float()
        d = torch.cdist(cells, core_cells).min(1).values.view(h, w)
        prox = torch.exp(-(d / prox_sigma) ** 2 / 2)
    else:
        prox = torch.zeros(h, w, dtype=torch.float32)
    lines = torch.zeros(h, w, dtype=torch.float32)
    for box in layout_boxes_px:
        r1, r2, c1, c2 = rect(box)
        lines[r1:r2 + 1, c1:c2 + 1] = 1.0
    bg = lines * (bg_far + (bg_near - bg_far) * prox)
    bg = torch.maximum(bg, _gaussian_blur2d(bg, 0.6))

    grid = torch.maximum(torch.maximum(halo, votes), bg)
    grid = torch.maximum(grid, core)   # 核心最后钉回，不被光晕/热斑稀释
    return grid


def grid_to_indices_weights(grid, threshold=0.05, device="cpu"):
    """[h, w] 掩码 -> (行主序索引 LongTensor, 对应权重 FloatTensor)。"""
    flat = grid.flatten()
    idx = (flat >= threshold).nonzero(as_tuple=True)[0]
    return idx.to(device), flat[idx].to(device)
