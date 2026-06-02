"""
binarization.py - 二值化模块
支持 Otsu / 自适应阈值 / 颜色掩码直接二值化
并提供形态学后处理
"""

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    """BGR → 灰度"""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    """
    Otsu 全局自动阈值二值化
    :param gray: 灰度图
    :return: 二值图（前景255，背景0）
    """
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def adaptive_binarize(gray: np.ndarray,
                      block_size: int = 21,
                      C: int = 5) -> np.ndarray:
    """
    自适应局部阈值二值化（适合光照不均图像）
    :param gray: 灰度图
    :param block_size: 局部窗口大小（奇数）
    :param C: 阈值偏移量
    :return: 二值图
    """
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )
    return binary


def mask_binarize(roi: np.ndarray, sign_type: str) -> np.ndarray:
    """
    基于HSV颜色掩码直接生成前景二值图
    适合颜色鲜明的交通标志内部图案提取
    :param roi: 裁剪后的ROI（BGR）
    :param sign_type: 标志类型
    :return: 二值图（图案区域为255）
    """
    from localization import build_color_mask
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 获取标志颜色区域（背景）
    color_mask = build_color_mask(hsv, sign_type)

    # 取反：标志内部图案（非颜色区域）作为前景
    pattern_mask = cv2.bitwise_not(color_mask)
    return pattern_mask


def morphology_clean(binary: np.ndarray,
                     open_ksize: int = 3,
                     close_ksize: int = 5) -> np.ndarray:
    """
    形态学后处理：去除噪点，填充孔洞
    :param binary: 二值图
    :param open_ksize: 开运算核大小（去噪）
    :param close_ksize: 闭运算核大小（填孔）
    :return: 处理后的二值图
    """
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (open_ksize, open_ksize))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (close_ksize, close_ksize))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)
    return cleaned


def binarize_pipeline(roi: np.ndarray,
                      method: str = 'otsu') -> np.ndarray:
    """
    完整二值化流水线
    :param roi: 输入ROI（BGR）
    :param method: 'otsu' | 'adaptive' | 'both'（融合两种结果）
    :return: 清洁后的二值图
    """
    gray = to_gray(roi)

    if method == 'otsu':
        binary = otsu_binarize(gray)
    elif method == 'adaptive':
        binary = adaptive_binarize(gray)
    elif method == 'both':
        b1 = otsu_binarize(gray)
        b2 = adaptive_binarize(gray)
        binary = cv2.bitwise_and(b1, b2)   # 取交集，更保守
    else:
        raise ValueError(f"不支持的二值化方法: {method}")

    binary = morphology_clean(binary)
    return binary