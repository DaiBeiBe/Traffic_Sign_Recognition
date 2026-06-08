"""
main.py - 交通标志识别系统主程序
用法:
    python main.py                    # 使用合成数据运行完整流程
    python main.py --data data/raw    # 使用真实数据
    python main.py --method hu        # 仅使用胡矩方法
"""

import argparse
import os
import sys
import cv2
import numpy as np

# 将 src 加入模块搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess      import preprocess_pipeline
from localization    import localize, draw_detection, correct_tilt
from binarization    import binarize_pipeline
from feature_extraction import extract_features
from recognition     import TrafficSignRecognizer
from evaluation      import Evaluator
from data_utils      import (scan_dataset, validate_dataset,
                              generate_synthetic_dataset,
                              register_all_templates, SIGN_CLASSES)


# ─────────────────────────────────────────
# 单张图像识别流程
# ─────────────────────────────────────────

# 参数verbose 的核心作用是控制输出的信息量，便于调试、监控或保持简洁。
def process_single(img_path: str,
                   recognizer: TrafficSignRecognizer,
                   true_label: str = None,
                   is_tilted: bool = False,
                   save_dir: str = 'results/detected',
                   verbose: bool = True) -> dict:
    """
    对单张图像执行完整识别流水线
    :return: 识别结果字典（含 pred 和 vis 图）
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. 预处理
    img = preprocess_pipeline(img_path)

    # 2. 颜色定位（已知类别时可传入 sign_type 加速）
    roi, mask, detected_type, bbox = localize(img, sign_type=true_label)

    if roi is None:
        if verbose:
            print(f"  [!] 未能定位到标志区域: {os.path.basename(img_path)}")
        return {'label': 'unknown', 'confidence': 0.0,
                'bbox': None, 'detected_type': detected_type}

    # 3. 倾斜矫正（倾斜图像）
    if is_tilted:
        roi = correct_tilt(roi)

    # 4. 二值化
    binary = binarize_pipeline(roi)

    # 5. 特征提取
    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    feature = extract_features(binary, gray=gray)

    # 6. 识别
    result = recognizer.predict(roi, feature=feature)

    # 7. 可视化并保存
    if bbox is not None:
        label_str = (f"{result['label']} ({result['confidence']:.2f})"
                     if result['label'] != 'unknown'
                     else '未识别')
        color = (0, 255, 0) if result['label'] == true_label else (0, 0, 255)
        vis   = draw_detection(img, bbox, label_str, color=color)
    else:
        vis = img.copy()

    fname     = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(save_dir, f'{fname}_result.png')
    cv2.imwrite(save_path, vis)

    if verbose:
        correct_mark = ''
        if true_label:
            correct_mark = ('✓' if result['label'] == true_label else '✗')
        print(f"  {correct_mark} {os.path.basename(img_path):<30s}"
              f"  真实={true_label or 'N/A':12s}"
              f"  预测={result['label']:12s}"
              f"  置信={result['confidence']:.2f}")

    result['detected_type'] = detected_type
    result['bbox']          = bbox
    return result


# ─────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────

def main(args):
    print("\n" + "=" * 60)
    print("          交通标志识别系统")
    print("=" * 60)

    data_root = args.data

    # ── Step 0: 生成合成数据（如无真实数据）──────────────
    if not os.path.exists(data_root) or args.synthetic:
        print("\n[Step 0] 生成合成测试数据集…")
        generate_synthetic_dataset(data_root,
                                   ideal_per_class=3,
                                   tilted_per_class=2)
    else:
        print(f"\n[Step 0] 使用数据目录: {data_root}")

    # ── Step 1: 扫描数据集 ──────────────────────────────
    print("\n[Step 1] 扫描数据集")
    dataset = scan_dataset(data_root)
    dataset_ok = validate_dataset(dataset)
    if not dataset_ok:
        print("[!] 数据不满足实验要求，请补充图像后再运行。")

    # ── Step 2: 构建识别器并注册模板 ──────────────────────
    print(f"\n[Step 2] 初始化识别器（方法={args.method}）")
    recognizer = TrafficSignRecognizer(method=args.method)
    register_all_templates(recognizer, dataset, ideal_only=True)

    # ── Step 3: 批量识别并评估 ─────────────────────────────
    print("\n[Step 3] 批量识别")
    evaluator = Evaluator(classes=SIGN_CLASSES)

    for cls in SIGN_CLASSES:
        items = dataset[cls]
        if not items:
            continue
        print(f"\n  ── {cls} ──")
        for path, is_tilted in items:
            result = process_single(
                path, recognizer,
                true_label=cls,
                is_tilted=is_tilted,
                save_dir=args.output,
                verbose=True
            )
            evaluator.add_result(path, cls, result, is_tilted=is_tilted)

    # ── Step 4: 生成评估报告 ──────────────────────────────
    print("\n[Step 4] 生成评估报告")
    evaluator.generate_report(save_dir='results/report')

    print("\n[完成] 检测结果图像保存在:", args.output)


# ─────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='交通标志识别系统')
    parser.add_argument('--data',      default='data/raw',
                        help='数据集根目录（默认: data/raw）')
    parser.add_argument('--method',    default='fusion',
                        choices=['hu', 'template', 'fusion'],
                        help='识别方法（默认: fusion）')
    parser.add_argument('--output',    default='results/detected',
                        help='检测结果保存目录')
    parser.add_argument('--synthetic', action='store_true',
                        help='强制重新生成合成数据集')
    args = parser.parse_args()

    # 将工作目录切换到项目根
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main(args)