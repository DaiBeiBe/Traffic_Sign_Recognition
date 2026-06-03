"""
data_utils.py - 数据集工具
功能：
  1. 扫描 data/raw/ 目录，自动按子文件夹分类
  2. 生成合成测试图像（当真实数据不足时用于调试）
  3. 注册模板到识别器
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple


SIGN_CLASSES = ['prohibit', 'warning', 'mandatory', 'guide', 'stop']

# 各类标志的代表颜色（BGR）
CLASS_COLORS = {
    'prohibit':  (0,   0,   220),   # 红色
    'warning':   (0,   200, 220),   # 黄色
    'mandatory': (220, 80,  0),     # 蓝色
    'guide':     (0,   160, 0),     # 绿色
    'stop':      (60,  60,  180),   # 深红
}


def scan_dataset(data_dir: str) -> Dict[str, List[Tuple[str, bool]]]:
    """
    扫描数据集目录
    :param data_dir: data/raw/ 根目录
    :return: {label: [(image_path, is_tilted), ...]}
    """
    dataset = {cls: [] for cls in SIGN_CLASSES}

    ideal_dir  = os.path.join(data_dir, 'ideal')
    tilted_dir = os.path.join(data_dir, 'tilted')

    # 理想图像
    if os.path.exists(ideal_dir):
        for cls in SIGN_CLASSES:
            cls_dir = os.path.join(ideal_dir, cls)
            if os.path.exists(cls_dir):
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        dataset[cls].append(
                            (os.path.join(cls_dir, fname), False))

    # 倾斜图像（文件名包含类别前缀，如 prohibit_01.jpg）
    if os.path.exists(tilted_dir):
        for fname in sorted(os.listdir(tilted_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            for cls in SIGN_CLASSES:
                if fname.lower().startswith(cls):
                    dataset[cls].append(
                        (os.path.join(tilted_dir, fname), True))
                    break

    return dataset


def validate_dataset(dataset: Dict) -> bool:
    """检查数据集是否满足实验要求（每类≥3理想，≥2倾斜）"""
    ok = True
    for cls, items in dataset.items():
        ideal_n  = sum(1 for _, t in items if not t)
        tilted_n = sum(1 for _, t in items if t)
        status = "✓" if ideal_n >= 3 and tilted_n >= 2 else "✗"
        print(f"  [{status}] {cls:12s}: 理想={ideal_n}张  倾斜={tilted_n}张")
        if ideal_n < 3 or tilted_n < 2:
            ok = False
    return ok


# ─────────────────────────────────────────
# 合成图像生成器（调试用）
# ─────────────────────────────────────────

def _draw_prohibit(size=128) -> np.ndarray:
    """红色圆环 + 白色圆盘 + 红色横杠"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 200
    cx, cy, r = size // 2, size // 2, size // 2 - 5
    cv2.circle(img, (cx, cy), r, (0, 0, 200), -1)
    cv2.circle(img, (cx, cy), r - 8, (255, 255, 255), -1)
    cv2.rectangle(img, (cx - r + 15, cy - 8), (cx + r - 15, cy + 8),
                  (0, 0, 200), -1)
    cv2.circle(img, (cx, cy), r, (0, 0, 150), 3)
    return img


def _draw_warning(size=128) -> np.ndarray:
    """生成警告类合成图（黄色三角形，内有感叹号）"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 200
    pts = np.array([
        [size // 2, 8],
        [size - 8,  size - 8],
        [8,         size - 8]
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], (0, 200, 220))
    cv2.polylines(img, [pts], True, (0, 140, 180), 3)
    # 感叹号
    cv2.rectangle(img, (size//2-4, size//2-20), (size//2+4, size//2+15),
                  (30, 30, 30), -1)
    cv2.circle(img, (size//2, size//2+28), 4, (30, 30, 30), -1)
    return img


def _draw_mandatory(size=128) -> np.ndarray:
    """生成指示类合成图（蓝色圆形，内有白色箭头）"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 200
    cx, cy, r = size // 2, size // 2, size // 2 - 5
    cv2.circle(img, (cx, cy), r, (200, 80, 0), -1)
    # 箭头
    cv2.arrowedLine(img, (cx, cy + r//2), (cx, cy - r//2),
                    (255, 255, 255), 8, tipLength=0.4)
    return img


def _draw_guide(size=128) -> np.ndarray:
    """生成指路类合成图（绿色矩形，内有白色文字）"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 200
    pad = 10
    cv2.rectangle(img, (pad, pad), (size-pad, size-pad), (0, 140, 0), -1)
    cv2.putText(img, 'GO', (size//2-20, size//2+10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return img


def _draw_stop(size=128) -> np.ndarray:
    """生成停车类合成图（八边形，红底白字STOP）"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 200
    cx, cy, r = size//2, size//2, size//2 - 5
    angles = np.linspace(np.pi/8, 2*np.pi + np.pi/8, 9)[:-1]
    pts = np.array([[int(cx + r*np.cos(a)), int(cy + r*np.sin(a))]
                    for a in angles], dtype=np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 180))
    cv2.putText(img, 'STOP', (cx-28, cy+8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


DRAW_FUNCS = {
    'prohibit':  _draw_prohibit,
    'warning':   _draw_warning,
    'mandatory': _draw_mandatory,
    'guide':     _draw_guide,
    'stop':      _draw_stop,
}


def generate_synthetic_dataset(save_dir: str,
                                ideal_per_class: int = 3,
                                tilted_per_class: int = 2,
                                size: int = 128):
    """
    生成合成测试数据集（用于无真实数据时的功能验证）
    :param save_dir:         data/raw/ 根目录
    :param ideal_per_class:  每类理想图数量
    :param tilted_per_class: 每类倾斜图数量
    :param size:             图像尺寸
    """
    rng = np.random.default_rng(42)

    for cls in SIGN_CLASSES:
        draw_fn = DRAW_FUNCS[cls]

        # ── 理想图 ──
        ideal_cls_dir = os.path.join(save_dir, 'ideal', cls)
        os.makedirs(ideal_cls_dir, exist_ok=True)
        for i in range(ideal_per_class):
            img = draw_fn(size)
            # 轻微随机噪声
            noise = rng.integers(-15, 15, img.shape, dtype=np.int16)
            img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            path  = os.path.join(ideal_cls_dir, f'{cls}_{i+1:02d}.png')
            cv2.imwrite(path, img)

        # ── 倾斜图 ──
        tilted_dir = os.path.join(save_dir, 'tilted')
        os.makedirs(tilted_dir, exist_ok=True)
        for i in range(tilted_per_class):
            img   = draw_fn(size)
            angle = rng.uniform(10, 35) * rng.choice([-1, 1])
            M     = cv2.getRotationMatrix2D((size//2, size//2), float(angle), 1.0)
            img   = cv2.warpAffine(img, M, (size, size),
                                   borderMode=cv2.BORDER_REPLICATE)
            path  = os.path.join(tilted_dir,
                                 f'{cls}_tilted_{i+1:02d}.png')
            cv2.imwrite(path, img)

    print(f"[✓] 合成数据集已生成至: {save_dir}")


def register_all_templates(recognizer, dataset: Dict,
                            ideal_only: bool = True):
    """
    将数据集中的图像注册为模板
    :param recognizer:  TrafficSignRecognizer 实例
    :param dataset:     scan_dataset() 返回的字典
    :param ideal_only:  是否只用理想图注册（推荐True）
    """
    from preprocess import preprocess_pipeline
    count = 0
    for cls, items in dataset.items():
        for path, is_tilted in items:
            if ideal_only and is_tilted:
                continue
            try:
                img = preprocess_pipeline(path)
                recognizer.register_template(cls, img)
                count += 1
            except Exception as e:
                print(f"[!] 跳过 {path}: {e}")
    print(f"[✓] 已注册 {count} 张模板")