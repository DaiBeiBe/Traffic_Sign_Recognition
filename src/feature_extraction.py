"""
feature_extraction.py - 特征提取模块
主要方法：胡矩（Hu Moments）7维特征向量
辅助方法：HOG 特征（可选）
"""

import cv2
import numpy as np
from typing import Tuple

# 胡矩特征：对平移/旋转/缩放不变，适合形状描述，尤其是交通标志的几何特征。
def compute_hu_moments(binary: np.ndarray) -> np.ndarray:
    """
    计算胡矩特征（7维，对平移/旋转/缩放不变）
    :param binary: 二值图（前景255）
    :return: shape=(7,) 的特征向量
    """
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()   # shape (7,)
    return hu

# 对胡矩做对数变换，缩小数量级差异，利于距离比较。
def log_transform_hu(hu: np.ndarray) -> np.ndarray:
    """
    对胡矩做对数变换，缩小数量级差异，利于距离比较
    :param hu: 原始胡矩向量
    :return: 对数变换后的向量
    """
    # 防止 log(0)；保留符号
    epsilon = 1e-10
    return -np.sign(hu) * np.log10(np.abs(hu) + epsilon)


def compute_moments_features(binary: np.ndarray) -> np.ndarray:
    """
    计算完整的矩特征（中心矩 + 归一化中心矩 + 胡矩）
    :param binary: 二值图
    :return: 特征向量（对数胡矩，shape=(7,)）
    """
    hu = compute_hu_moments(binary)
    return log_transform_hu(hu)


def compute_hog(gray: np.ndarray,
                win_size: Tuple = (64, 64),
                cell_size: Tuple = (8, 8),
                block_size: Tuple = (16, 16),
                nbins: int = 9) -> np.ndarray:
    """
    计算 HOG（方向梯度直方图）特征
    :param gray: 灰度图（会自动 resize 到 win_size）
    :param win_size: 检测窗口大小
    :param cell_size: 每个 cell 的像素大小
    :param block_size: 每个 block 的像素大小
    :param nbins: 方向 bin 的数量
    :return: HOG 特征向量
    """
    gray_resized = cv2.resize(gray, win_size)

    hog = cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=block_size,
        _blockStride=(cell_size[0] // 2, cell_size[1] // 2),
        _cellSize=cell_size,
        _nbins=nbins
    )
    descriptor = hog.compute(gray_resized)
    return descriptor.flatten()


def compute_shape_features(binary: np.ndarray) -> np.ndarray:
    """
    计算几何形状描述符：
    - 圆形度
    - 长宽比
    - 凸包面积比
    :param binary: 二值图
    :return: shape=(3,) 特征向量
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(3)

    # 选择最大轮廓作为目标，计算面积、周长等基本属性。
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    # 圆形度
    circularity = (4 * np.pi * area / (perimeter ** 2 + 1e-10))

    # 长宽比（最小外接矩形）
    _, (w, h), _ = cv2.minAreaRect(cnt)
    aspect_ratio = min(w, h) / (max(w, h) + 1e-10)

    # 凸包填充率
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convexity = area / (hull_area + 1e-10)

    return np.array([circularity, aspect_ratio, convexity])


def extract_features(binary: np.ndarray,
                     gray: np.ndarray = None,
                     use_hog: bool = False,
                     use_shape: bool = True) -> np.ndarray:
    """
    综合特征提取入口
    :param binary: 二值图
    :param gray:   灰度图（use_hog=True 时需要）
    :param use_hog: 是否附加 HOG 特征
    :param use_shape: 是否附加形状特征
    :return: 最终特征向量
    """
    hu_feat = compute_moments_features(binary)

    parts = [hu_feat]

    if use_shape:
        shape_feat = compute_shape_features(binary)
        parts.append(shape_feat)

    if use_hog and gray is not None:
        hog_feat = compute_hog(gray)
        parts.append(hog_feat)

    return np.concatenate(parts)