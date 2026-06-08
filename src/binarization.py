"""
binarization.py - 二值化模块
支持 Otsu / 自适应阈值 / 颜色掩码直接二值化
Otsu	    自适应阈值
单个全局值	 每个像素不同
并提供形态学后处理
"""

import cv2
import numpy as np

# 将图像转换为灰度图，便于后续二值化处理。
def to_gray(img: np.ndarray) -> np.ndarray:
    # 如果输入已经是灰度图，则直接返回。
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Otsu 二值化：自动计算全局阈值，适合 bimodal（双峰）图像，但对光照不均敏感。
# 类间方差最大化方法，适用于前景和背景灰度分布明显分离的情况。
def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu 算法用于自动决定阈值，然后用二进制阈值化规则来生成最终图像。
    return binary

# 自适应阈值：根据局部区域动态计算阈值，适合光照不均的图像。
# 计算每个像素的阈值，基于其邻域内的像素值，适用于光照条件变化较大的图像。
def adaptive_binarize(gray: np.ndarray,
                      block_size: int = 21,
                      C: int = 5) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )

# 颜色掩码直接二值化：针对已知类别，利用颜色信息生成二值图，适合颜色特征明显的标志。
def mask_binarize(roi: np.ndarray, sign_type: str) -> np.ndarray:
    from localization import build_color_mask
    hsv          = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_mask   = build_color_mask(hsv, sign_type)
    pattern_mask = cv2.bitwise_not(color_mask)
    return pattern_mask

# 形态学清理：开运算去噪点，闭运算填补孔洞，适合清理二值图中的小瑕疵。
def morphology_clean(binary: np.ndarray,
                     open_ksize: int = 3,
                     close_ksize: int = 5) -> np.ndarray:
    # 形态学开运算去除小噪点
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize,  open_ksize))
    cleaned = cv2.morphologyEx(binary,  cv2.MORPH_OPEN,  k_open)
    # 形态学闭运算填补小孔洞
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)
    return cleaned

# 完整二值化流水线：结合多种方法，增强鲁棒性，适合复杂场景。
def binarize_pipeline(roi: np.ndarray,
                      method: str = 'both') -> np.ndarray:
    """
    完整二值化流水线
    ★ 默认 'both'（Otsu ∩ 自适应），光照不均时更鲁棒
    """
    gray = to_gray(roi)

    if method == 'otsu':
        binary = otsu_binarize(gray)
    elif method == 'adaptive':
        binary = adaptive_binarize(gray)
    elif method == 'both':
        b1     = otsu_binarize(gray)
        b2     = adaptive_binarize(gray)
        binary = cv2.bitwise_and(b1, b2)
    else:
        raise ValueError(f"不支持的二值化方法: {method}")

    return morphology_clean(binary)