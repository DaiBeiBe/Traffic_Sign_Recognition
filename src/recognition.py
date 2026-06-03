"""
recognition.py - 识别模块
方法一：胡矩特征 + 最近邻（欧氏距离）
方法二：模板匹配（归一化互相关 TM_CCOEFF_NORMED）
方法三：两者融合投票
"""

import cv2
import numpy as np
import os
import pickle
from typing import Dict, List, Optional, Tuple


class HuMomentClassifier:
    def __init__(self):
        self.templates: Dict[str, List[np.ndarray]] = {}

    def add_template(self, label: str, feature: np.ndarray):
        self.templates.setdefault(label, []).append(feature)

    def predict(self, feature: np.ndarray) -> Tuple[str, float]:
        if not self.templates:
            raise RuntimeError("模板库为空，请先注册模板")
        best_label = 'unknown'
        best_dist  = float('inf')
        for label, feats in self.templates.items():
            for tmpl_feat in feats:
                dist = np.linalg.norm(feature - tmpl_feat)
                if dist < best_dist:
                    best_dist  = dist
                    best_label = label
        return best_label, best_dist

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.templates, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.templates = pickle.load(f)


class TemplateMatcher:
    def __init__(self, template_size: Tuple = (64, 64)):
        self.template_size = template_size
        self.templates: Dict[str, List[np.ndarray]] = {}

    def add_template(self, label: str, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        gray = cv2.resize(gray, self.template_size)
        self.templates.setdefault(label, []).append(gray)

    def match_single(self, query_gray: np.ndarray,
                     template_gray: np.ndarray) -> float:
        q      = cv2.resize(query_gray, self.template_size).astype(np.float32)
        result = cv2.matchTemplate(q, template_gray.astype(np.float32),
                                   cv2.TM_CCOEFF_NORMED)
        return float(result[0][0])

    def predict(self, roi: np.ndarray) -> Tuple[str, float]:
        if not self.templates:
            raise RuntimeError("模板库为空，请先注册模板")
        query_gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        best_label  = 'unknown'
        best_score  = -float('inf')
        for label, tmpl_list in self.templates.items():
            for tmpl in tmpl_list:
                score = self.match_single(query_gray, tmpl)
                if score > best_score:
                    best_score = score
                    best_label = label
        return best_label, best_score

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'templates': self.templates, 'size': self.template_size}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.templates     = data['templates']
        self.template_size = data['size']


class TrafficSignRecognizer:
    """
    融合识别器：胡矩分类 + 模板匹配
    支持三种模式：'hu' | 'template' | 'fusion'
    """

    def __init__(self, method: str = 'fusion'):
        self.method = method
        self.hu_clf = HuMomentClassifier()
        self.tm_clf = TemplateMatcher()

        # ★ 放宽阈值：实景倾斜图特征分布更分散
        self.hu_dist_threshold  = 6.0    # 原 5.0
        self.tm_score_threshold = 0.25   # 原 0.3

    def register_template(self, label: str,
                          img: np.ndarray,
                          feature: Optional[np.ndarray] = None):
        self.tm_clf.add_template(label, img)
        if feature is None:
            from binarization import binarize_pipeline
            from feature_extraction import extract_features
            binary  = binarize_pipeline(img)
            feature = extract_features(binary)
        self.hu_clf.add_template(label, feature)

    def predict(self, roi: np.ndarray,
                feature: Optional[np.ndarray] = None) -> Dict:
        if feature is None:
            from binarization import binarize_pipeline
            from feature_extraction import extract_features
            binary  = binarize_pipeline(roi)
            feature = extract_features(binary)

        result = {
            'hu_label': None, 'hu_dist':  None,
            'tm_label': None, 'tm_score': None,
            'label': 'unknown', 'method': self.method,
            'confidence': 0.0
        }

        if self.method in ('hu', 'fusion'):
            hu_label, hu_dist = self.hu_clf.predict(feature)
            result['hu_label'] = hu_label
            result['hu_dist']  = hu_dist

        if self.method in ('template', 'fusion'):
            tm_label, tm_score = self.tm_clf.predict(roi)
            result['tm_label'] = tm_label
            result['tm_score'] = tm_score

        if self.method == 'hu':
            if hu_dist <= self.hu_dist_threshold:
                result['label']      = hu_label
                result['confidence'] = max(0.0, 1.0 - hu_dist / self.hu_dist_threshold)

        elif self.method == 'template':
            if tm_score >= self.tm_score_threshold:
                result['label']      = tm_label
                result['confidence'] = (tm_score + 1) / 2

        elif self.method == 'fusion':
            hu_valid = (result['hu_dist']  is not None and
                        result['hu_dist']  <= self.hu_dist_threshold)
            tm_valid = (result['tm_score'] is not None and
                        result['tm_score'] >= self.tm_score_threshold)

            if hu_valid and tm_valid:
                if hu_label == tm_label:
                    result['label']      = hu_label
                    result['confidence'] = 0.9
                else:
                    # 不一致：胡矩对旋转/倾斜更稳定，优先采用
                    result['label']      = hu_label
                    result['confidence'] = 0.5
            elif hu_valid:
                result['label']      = hu_label
                result['confidence'] = max(0.0, 0.7 * (1.0 - hu_dist / self.hu_dist_threshold))
            elif tm_valid:
                result['label']      = tm_label
                result['confidence'] = 0.6 * (tm_score + 1) / 2

        return result

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.hu_clf.save(os.path.join(save_dir, 'hu_classifier.pkl'))
        self.tm_clf.save(os.path.join(save_dir, 'tm_classifier.pkl'))
        print(f"[✓] 分类器已保存至 {save_dir}")

    def load(self, save_dir: str):
        self.hu_clf.load(os.path.join(save_dir, 'hu_classifier.pkl'))
        self.tm_clf.load(os.path.join(save_dir, 'tm_classifier.pkl'))
        print(f"[✓] 分类器已从 {save_dir} 加载")