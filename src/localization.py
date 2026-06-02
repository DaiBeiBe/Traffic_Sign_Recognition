"""
localization.py - 交通标志区域定位模块
基于颜色空间（HSV）定位交通标志区域
支持：红色（禁止类）、黄色（警告类）、蓝色（指示类）、白色（停车类）、绿色（指路类）
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# ─────────────────────────────────────────
# HSV 颜色阈值配置（可按实际图像微调）
# ─────────────────────────────────────────
COLOR_RANGES = {
    'prohibit': [   # 禁止类 —— 红色（HSV中红色跨越0和180度，需两段）
        (np.array([0,   100, 100]), np.array([10,  255, 255])),
        (np.array([160, 100, 100]), np.array([180, 255, 255])),
    ],
    'warning': [    # 警告类 —— 黄色
        (np.array([15,  100, 100]), np.array([35,  255, 255])),
    ],
    'mandatory': [  # 指示类 —— 蓝色
        (np.array([100, 100, 100]), np.array([130, 255, 255])),
    ],
    'guide': [      # 指路类 —— 绿色
        (np.array([40,  80,  80]),  np.array([80,  255, 255])),
    ],
    'stop': [       # 停车类 —— 红色/白色组合（以红色为主）
        (np.array([0,   100, 100]), np.array([10,  255, 255])),
        (np.array([160, 100, 100]), np.array([180, 255, 255])),
    ],
}

# 各类别期望轮廓形状（用于形状筛选）
SHAPE_CONFIG = {
    'prohibit':  {'min_circularity': 0.6,  'shape': 'circle'},
    'warning':   {'min_circularity': 0.4,  'shape': 'triangle'},
    'mandatory': {'min_circularity': 0.6,  'shape': 'circle'},
    'guide':     {'min_circularity': 0.0,  'shape': 'rectangle'},
    'stop':      {'min_circularity': 0.5,  'shape': 'octagon'},
}


def bgr_to_hsv(img: np.ndarray) -> np.ndarray:
    """BGR 转 HSV"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def build_color_mask(hsv: np.ndarray, sign_type: str) -> np.ndarray:
    """
    根据标志类别生成颜色掩码
    :param hsv: HSV图像
    :param sign_type: 标志类型
    :return: 二值掩码
    """
    ranges = COLOR_RANGES.get(sign_type)
    if ranges is None:
        raise ValueError(f"未知标志类型: {sign_type}")

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in ranges:
        mask |= cv2.inRange(hsv, lower, upper)

    # 形态学去噪：先开运算去小噪点，再闭运算填孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def build_mask_auto(hsv: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    自动检测颜色，返回最优掩码及对应类别
    :return: (mask, sign_type)
    """
    best_mask = None
    best_type = 'unknown'
    best_area = 0

    for sign_type in COLOR_RANGES:
        mask = build_color_mask(hsv, sign_type)
        area = cv2.countNonZero(mask)
        if area > best_area:
            best_area = area
            best_mask = mask
            best_type = sign_type

    return best_mask, best_type


def compute_circularity(contour) -> float:
    """计算轮廓圆形度"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter ** 2)


def filter_contours(contours, sign_type: str,
                    min_area: int = 500) -> List:
    """
    按面积和形状筛选候选轮廓
    :param contours: 原始轮廓列表
    :param sign_type: 标志类型
    :param min_area: 最小面积阈值
    :return: 筛选后的轮廓列表
    """
    cfg = SHAPE_CONFIG.get(sign_type, {'min_circularity': 0.0})
    min_circ = cfg['min_circularity']

    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        circ = compute_circularity(cnt)
        if circ >= min_circ:
            valid.append((cnt, area, circ))

    # 按面积降序排列
    valid.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in valid]


def find_roi(img: np.ndarray, mask: np.ndarray,
             sign_type: str = 'unknown') -> Optional[Tuple[np.ndarray, Tuple]]:
    """
    在掩码中找到最大候选区域并裁剪
    :param img: 原始BGR图像
    :param mask: 颜色掩码
    :param sign_type: 标志类型（用于形状筛选）
    :return: (roi图像, bounding_box) 或 None
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    filtered = filter_contours(contours, sign_type)
    if not filtered:
        # 退化：直接取最大轮廓
        filtered = [max(contours, key=cv2.contourArea)]

    cnt = filtered[0]
    x, y, w, h = cv2.boundingRect(cnt)

    # 适当扩边（防止标志被截断）
    pad = int(max(w, h) * 0.05)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)

    roi = img[y:y + h, x:x + w]
    return roi, (x, y, w, h)


def correct_tilt(roi: np.ndarray) -> np.ndarray:
    """
    对倾斜的ROI进行透视变换矫正
    先转灰度+Canny找边缘，再用最小外接矩形计算旋转角度
    :param roi: 裁剪后的ROI
    :return: 矫正后的图像
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return roi

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)          # ((cx,cy), (w,h), angle)
    angle = rect[2]

    # OpenCV minAreaRect 角度范围 [-90, 0)，需要修正
    (h, w) = roi.shape[:2]
    if angle < -45:
        angle += 90

    # 只在倾斜明显时才矫正（避免误操作）
    if abs(angle) < 2:
        return roi

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    corrected = cv2.warpAffine(roi, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    return corrected


def localize(img: np.ndarray,
             sign_type: Optional[str] = None
             ) -> Tuple[Optional[np.ndarray], np.ndarray, str, Optional[Tuple]]:
    """
    完整定位流程
    :param img: 预处理后的BGR图像
    :param sign_type: 若已知类别则直接用该颜色；否则自动检测
    :return: (roi, mask, detected_type, bbox)
    """
    hsv = bgr_to_hsv(img)

    if sign_type and sign_type in COLOR_RANGES:
        mask = build_color_mask(hsv, sign_type)
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
    """在原图上绘制检测框和标签"""
    vis = img.copy()
    x, y, w, h = bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
    cv2.putText(vis, label, (x, max(y - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis