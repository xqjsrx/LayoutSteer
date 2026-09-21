"""布局图渲染：全局布局图 + 局部（环面邻域）布局图。"""
from PIL import Image, ImageDraw


def bbox_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def generate_layout(image_size, boxes):
    """全局布局图：黑底 + 白色矩形。"""
    w, h = image_size
    layout = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(layout)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], fill=255)
    return Image.merge("RGB", [layout, layout, layout])


def get_local_bboxes(boxes, center_idx, n_local, img_size, wrap_gap_ratio=0.0,
                     use_wrap=True):
    """取中心框周围 n_local 个最近框。

    use_wrap=True: 环面 wrap-around 距离（上下/左右循环相接），边缘框也有
    完整邻域，被 wrap 的框平移到虚拟位置（坐标可为负），wrap_gap_ratio
    控制拼接处的可见空隙。
    use_wrap=False: 普通欧氏距离，边缘锚点只看真实邻居（避免页首/页底
    邻域内容集合对称导致的首尾混淆）。
    返回 (local_bboxes: [(idx, box)], region: (x1, y1, x2, y2))。
    """
    cx, cy = bbox_center(boxes[center_idx])
    W, H = img_size
    gap_x, gap_y = W * wrap_gap_ratio, H * wrap_gap_ratio
    distances = []
    for i, b in enumerate(boxes):
        bx, by = bbox_center(b)
        dx, dy = abs(bx - cx), abs(by - cy)
        if use_wrap:
            if W > 0:
                dx = min(dx, W - dx)
            if H > 0:
                dy = min(dy, H - dy)
        distances.append(((dx ** 2 + dy ** 2) ** 0.5, i))
    distances.sort()
    indices = [idx for _, idx in distances[:n_local]]
    if center_idx in indices:
        indices.remove(center_idx)
    indices.insert(0, center_idx)

    local_bboxes = []
    for idx in indices:
        x1, y1, x2, y2 = boxes[idx]
        if use_wrap:
            if W > 0:
                dx = (x1 + x2) / 2 - cx
                if dx > W / 2:
                    x1 -= W + gap_x; x2 -= W + gap_x
                elif dx < -W / 2:
                    x1 += W + gap_x; x2 += W + gap_x
            if H > 0:
                dy = (y1 + y2) / 2 - cy
                if dy > H / 2:
                    y1 -= H + gap_y; y2 -= H + gap_y
                elif dy < -H / 2:
                    y1 += H + gap_y; y2 += H + gap_y
        local_bboxes.append((idx, (x1, y1, x2, y2)))

    region = (min(b[0] for _, b in local_bboxes),
              min(b[1] for _, b in local_bboxes),
              max(b[2] for _, b in local_bboxes),
              max(b[3] for _, b in local_bboxes))
    return local_bboxes, region


def generate_local_layout(local_bboxes, region):
    """局部布局图（支持 wrap 后的负坐标）。"""
    rx1, ry1, rx2, ry2 = region
    rw, rh = int(rx2 - rx1), int(ry2 - ry1)
    if rw < 1 or rh < 1:
        rw, rh = 10, 10
    layout = Image.new("L", (rw, rh), 0)
    draw = ImageDraw.Draw(layout)
    for _, (x1, y1, x2, y2) in local_bboxes:
        draw.rectangle([x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1], fill=255)
    return Image.merge("RGB", [layout, layout, layout])
