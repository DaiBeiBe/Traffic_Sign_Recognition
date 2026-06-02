"""
preprocess.py - 图像预处理模块
功能：图像读取、缩放、去噪
"""

import cv2
import numpy as np
import os


def load_image(path: str) -> np.ndarray:
    """
    读取图像文件
    :param path: 图像路径
    :return: BGR格式图像
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"图像文件不存在: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    return img


def resize_image(img: np.ndarray, size: tuple = (128, 128)) -> np.ndarray:
    """
    缩放图像至指定尺寸
    :param img: 输入图像
    :param size: 目标尺寸 (width, height)
    :return: 缩放后的图像
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def denoise(img: np.ndarray, method: str = 'gaussian') -> np.ndarray:
    """
    图像去噪
    :param img: 输入图像
    :param method: 'gaussian' | 'median' | 'bilateral'
    :return: 去噪后的图像
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(img, (5, 5), 0)
    elif method == 'median':
        return cv2.medianBlur(img, 5)
    elif method == 'bilateral':
        return cv2.bilateralFilter(img, 9, 75, 75)
    else:
        raise ValueError(f"不支持的去噪方法: {method}")


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    对比度增强（CLAHE）
    :param img: BGR输入图像
    :return: 增强后的图像
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def preprocess_pipeline(path: str, target_size: tuple = (128, 128),
                         denoise_method: str = 'gaussian') -> np.ndarray:
    """
    完整预处理流水线
    :param path: 图像路径
    :param target_size: 目标尺寸
    :param denoise_method: 去噪方法
    :return: 预处理后的图像
    """
    img = load_image(path)
    img = enhance_contrast(img)
    img = denoise(img, method=denoise_method)
    img = resize_image(img, size=target_size)
    return img