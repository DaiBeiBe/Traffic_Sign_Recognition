"""
preprocess.py - 图像预处理模块
功能：图像读取、缩放、去噪、夜间/暗光自适应增强
"""

import cv2
import numpy as np
import os


def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"图像文件不存在: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    return img


def resize_image(img: np.ndarray, size: tuple = (128, 128)) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def denoise(img: np.ndarray, method: str = 'gaussian') -> np.ndarray:
    if method == 'gaussian':
        return cv2.GaussianBlur(img, (5, 5), 0)
    elif method == 'median':
        return cv2.medianBlur(img, 5)
    elif method == 'bilateral':
        return cv2.bilateralFilter(img, 9, 75, 75)
    else:
        raise ValueError(f"不支持的去噪方法: {method}")


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    伽马校正
    gamma < 1 → 提亮暗部（夜间/逆光）
    gamma > 1 → 压暗亮部（过曝）
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    自适应对比度增强：
    · mean_V < 60  → 夜间：先伽马提亮，CLAHE clipLimit=4.0
    · mean_V < 120 → 普通户外：CLAHE clipLimit=3.0
    · 其余         → 标准室内：CLAHE clipLimit=2.0
    """
    hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_v = float(hsv[:, :, 2].mean())

    if mean_v < 60:
        # 夜间：gamma 按亮度动态计算（越暗提亮幅度越大）
        gamma = max(0.3, mean_v / 80.0)
        img   = gamma_correction(img, gamma)
        clip_limit = 4.0
    elif mean_v < 120:
        clip_limit = 3.0
    else:
        clip_limit = 2.0

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe       = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced  = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def preprocess_pipeline(path: str, target_size: tuple = (128, 128),
                         denoise_method: str = 'gaussian') -> np.ndarray:
    """
    完整预处理流水线（含暗光自适应增强）
    """
    img = load_image(path)
    img = enhance_contrast(img)
    img = denoise(img, method=denoise_method)
    img = resize_image(img, size=target_size)
    return img