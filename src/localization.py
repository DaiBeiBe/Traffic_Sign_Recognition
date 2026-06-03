"""
localization.py - 交通标志区域定位模块
基于颜色空间（HSV）定位交通标志区域
支持：红色（禁止类）、黄色（警告类）、蓝色（指示类）、绿色（指路类）、红色（停车类）
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# ─────────────────────────────────────────
# HSV 颜色阈值配置
# ★ 三类标志均扩宽 S/V 下限，覆盖实景/夜间场景
# ─────────────────────────────────────────
COLOR_RANGES = {
    'prohibit': [
        # ★ V: 100→60, S: 100→90，覆盖实景照片中变暗的红色
        (np.array([0,   90, 60]),  np.array([10,  255, 255])),
        (np.array([160, 90, 60]),  np.array([180, 255, 255])),
    ],
    'warning': [
        # ★ H扩宽至10-40，S下限70，V下限60
        #   覆盖实景黄色（outdoor yellow S常低至70-80）
        (np.array([10,  70, 60]),  np.array([40,  255, 255])),
    ],
    'mandatory': [
        # ★ 继承上次改进：V下限30覆盖夜间，H宽到95-135
        (np.array([95,  60, 30]),  np.array([135, 255, 255])),
    ],
    'guide': [
        (np.array([40,  80, 60]),  np.array([80,  255, 255])),
    ],
    'stop': [
        (np.array([0,   90, 60]),  np.array([10,  255, 255])),
        (np.array([160, 90, 60]),  np.array([180, 255, 255])),
    ],
}

# 各类别期望轮廓形状
# ★ warning 0.4→0.25：实景三角形轮廓不完整，圆形度实测 0.3 左右
# ★ prohibit 0.6→0.45：tilted 照片轮廓略有缺损
SHAPE_CONFIG = {
    'prohibit':  {'min_circularity': 0.45, 'shape': 'circle'},
    'warning':   {'min_circularity': 0.25, 'shape': 'triangle'},
    'mandatory': {'min_circularity': 0.30, 'shape': 'circle'},
    'guide':     {'min_circularity': 0.0,  'shape': 'rectangle'},
    'stop':      {'min_circularity': 0.40, 'shape': 'octagon'},
}


def bgr_to_hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def build_color_mask(hsv: np.ndarray, sign_type: str) -> np.ndarray:
    ranges = COLOR_RANGES.get(sign_type)
    if ranges is None:
        raise ValueError(f"未知标志类型: {sign_type}")

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in ranges:
        mask |= cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def compute_circularity(contour) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


# ─────────────────────────────────────────
# ★ 新增：三角形轮廓验证（warning 专用）
# ─────────────────────────────────────────

def _is_triangle_like(contour) -> bool:
    """
    用 approxPolyDP 多 epsilon 尝试，检查轮廓是否近似三角形
    3-5 个顶点均视为三角形（实景轮廓噪声多，允许轻微偏差）
    """
    peri = cv2.arcLength(contour, True)
    if peri < 1:
        return False
    for eps in [0.04, 0.06, 0.08, 0.10, 0.12]:
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        n = len(approx)
        if 3 <= n <= 5:
            return True
    return False


def filter_contours(contours, sign_type: str,
                    min_area: int = 300) -> List:
    """
    按面积、形状筛选候选轮廓
    ★ warning 类额外做三角形形状验证，过滤背景杂色区域
    """
    cfg   = SHAPE_CONFIG.get(sign_type, {'min_circularity': 0.0})
    min_circ   = cfg['min_circularity']
    is_warning = (sign_type == 'warning')

    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        circ = compute_circularity(cnt)
        if circ < min_circ:
            continue
        # ★ 警告类：额外检查三角形轮廓（抑制背景杂色误判）
        if is_warning and not _is_triangle_like(cnt):
            continue
        valid.append((cnt, area, circ))

    valid.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in valid]


def find_roi(img: np.ndarray, mask: np.ndarray,
             sign_type: str = 'unknown') -> Optional[Tuple[np.ndarray, Tuple]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    filtered = filter_contours(contours, sign_type)
    if not filtered:
        # 退化：直接取最大轮廓（警告类退化时跳过三角检查）
        filtered = [max(contours, key=cv2.contourArea)]

    cnt = filtered[0]
    x, y, w, h = cv2.boundingRect(cnt)

    pad = int(max(w, h) * 0.08)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)

    roi = img[y:y + h, x:x + w]
    return roi, (x, y, w, h)


# ─────────────────────────────────────────
# ★ 改进：build_mask_auto 加入形状质量加权
#   防止 prohibit_tilted_02（蓝芯红叉）被误判为 mandatory
# ─────────────────────────────────────────

def build_mask_auto(hsv: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    自动检测颜色，返回最优掩码及对应类别
    评分 = 有效面积 × 形状奖励系数
    · 圆形的红色区域给予 1.4× 奖励（禁止标志特征鲜明）
    · 圆形的蓝色区域给予 1.2× 奖励
    · 其余按原始面积
    """
    best_mask  = None
    best_type  = 'unknown'
    best_score = 0

    for sign_type in COLOR_RANGES:
        mask = build_color_mask(hsv, sign_type)
        area = cv2.countNonZero(mask)
        if area < 200:
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt  = max(contours, key=cv2.contourArea)
        circ = compute_circularity(cnt)

        # 形状奖励：圆形红色 > 圆形蓝色 > 其余
        if sign_type == 'prohibit' and circ > 0.45:
            bonus = 1.4
        elif sign_type == 'mandatory' and circ > 0.45:
            bonus = 1.2
        else:
            bonus = 1.0

        score = area * bonus
        if score > best_score:
            best_score = score
            best_mask  = mask
            best_type  = sign_type

    return best_mask, best_type


# ─────────────────────────────────────────
# 倾斜矫正（mandatory 透视 + 通用旋转）
# ─────────────────────────────────────────

def _try_perspective_correct(roi: np.ndarray) -> np.ndarray:
    """
    椭圆拟合透视矫正（针对 mandatory 蓝色标志）
    长短轴比 < 0.82 时才触发，防止正常圆形误操作
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv,
                            np.array([90,  40, 20]),
                            np.array([140, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return roi

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return roi

    ellipse = cv2.fitEllipse(cnt)
    (cx, cy), (ax1, ax2), angle = ellipse
    minor_ax = min(ax1, ax2)
    major_ax = max(ax1, ax2)

    aspect = minor_ax / (major_ax + 1e-6)
    if aspect > 0.82:
        return roi

    h, w  = roi.shape[:2]
    scale_y = min(major_ax / (minor_ax + 1e-6), 2.5)

    M_rot   = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(roi, M_rot, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    new_h   = int(h * scale_y)
    corrected = cv2.resize(rotated, (w, new_h), interpolation=cv2.INTER_CUBIC)

    if new_h > h:
        start = (new_h - h) // 2
        corrected = corrected[start:start + h, :]
    return corrected


def correct_tilt(roi: np.ndarray) -> np.ndarray:
    """
    两步倾斜矫正：
    1. 透视矫正（椭圆→圆，针对 mandatory）
    2. 旋转矫正（minAreaRect，通用）
    """
    roi = _try_perspective_correct(roi)

    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return roi

    cnt   = max(contours, key=cv2.contourArea)
    rect  = cv2.minAreaRect(cnt)
    angle = rect[2]

    h, w = roi.shape[:2]
    if angle < -45:
        angle += 90
    if abs(angle) < 2:
        return roi

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(roi, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def localize(img: np.ndarray,
             sign_type: Optional[str] = None
             ) -> Tuple[Optional[np.ndarray], np.ndarray, str, Optional[Tuple]]:
    hsv = bgr_to_hsv(img)

    if sign_type and sign_type in COLOR_RANGES:
        mask          = build_color_mask(hsv, sign_type)
        detected_type = sign_type
    else:
        mask, detected_type = build_mask_auto(hsv)

    result = find_roi(img, mask, detected_type)
    if result is None:
        return None, mask, detected_type, None

    roi, bbox = result
    return roi, mask, detected_type, bbox


def draw_detection(img: np.ndarray, bbox: Tuple,
                   label: str, color: Tuple = (0, 255, 0)) -> np.ndarray:
    vis = img.copy()
    x, y, w, h = bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
    cv2.putText(vis, label, (x, max(y - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis