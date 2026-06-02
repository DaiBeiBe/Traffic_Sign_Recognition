"""
evaluation.py - 评估模块
功能：准确率统计、混淆矩阵、误差原因分析、结果可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from typing import List, Dict, Tuple
from collections import defaultdict

# 支持中文显示
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

LABEL_NAMES = {
    'prohibit':  '禁止类',
    'warning':   '警告类',
    'mandatory': '指示类',
    'guide':     '指路类',
    'stop':      '停车类',
    'unknown':   '未识别',
}

ERROR_REASONS = {
    '颜色偏差':     '光照变化导致颜色阈值失效',
    '遮挡':        '标志被部分遮挡，特征不完整',
    '倾斜角度大':   '倾斜超过矫正能力',
    '相似类别混淆': '两类标志形状/颜色相近',
    '图像模糊':    '分辨率不足导致特征丢失',
    '尺度过小':    '标志在图像中占比过小',
}


class Evaluator:
    def __init__(self, classes: List[str] = None):
        self.classes = classes or list(LABEL_NAMES.keys())[:-1]
        self.reset()

    def reset(self):
        self.records: List[Dict] = []   # 每条记录

    def add_result(self, image_path: str, true_label: str,
                   pred_result: Dict, is_tilted: bool = False):
        """
        添加一条识别结果
        :param image_path:  图像路径（用于报告）
        :param true_label:  真实标签
        :param pred_result: recognition.py 返回的结果字典
        :param is_tilted:   是否为倾斜图像
        """
        pred_label = pred_result.get('label', 'unknown')
        correct    = (pred_label == true_label)
        self.records.append({
            'path':       image_path,
            'true':       true_label,
            'pred':       pred_label,
            'correct':    correct,
            'tilted':     is_tilted,
            'confidence': pred_result.get('confidence', 0.0),
            'hu_dist':    pred_result.get('hu_dist'),
            'tm_score':   pred_result.get('tm_score'),
        })

    # ─────────────────────────────────────────
    # 统计方法
    # ─────────────────────────────────────────

    def overall_accuracy(self) -> float:
        if not self.records:
            return 0.0
        return sum(r['correct'] for r in self.records) / len(self.records)

    def accuracy_by_class(self) -> Dict[str, Dict]:
        stats = {c: {'total': 0, 'correct': 0} for c in self.classes}
        for r in self.records:
            lbl = r['true']
            if lbl in stats:
                stats[lbl]['total']   += 1
                stats[lbl]['correct'] += int(r['correct'])
        result = {}
        for c, s in stats.items():
            acc = s['correct'] / s['total'] if s['total'] > 0 else 0.0
            result[c] = {**s, 'accuracy': acc}
        return result

    def accuracy_tilted_vs_ideal(self) -> Dict[str, float]:
        ideal_total  = sum(1 for r in self.records if not r['tilted'])
        ideal_correct= sum(1 for r in self.records if not r['tilted'] and r['correct'])
        tilt_total   = sum(1 for r in self.records if r['tilted'])
        tilt_correct = sum(1 for r in self.records if r['tilted'] and r['correct'])
        return {
            'ideal_acc':  ideal_correct / ideal_total  if ideal_total  > 0 else 0.0,
            'tilted_acc': tilt_correct  / tilt_total   if tilt_total   > 0 else 0.0,
            'ideal_n':    ideal_total,
            'tilted_n':   tilt_total,
        }

    def confusion_matrix(self) -> Tuple[np.ndarray, List[str]]:
        labels = self.classes
        n = len(labels)
        idx = {l: i for i, l in enumerate(labels)}
        cm  = np.zeros((n, n), dtype=int)
        for r in self.records:
            t = r['true']
            p = r['pred']
            if t in idx:
                pi = idx.get(p, -1)
                if pi >= 0:
                    cm[idx[t]][pi] += 1
        return cm, labels

    def error_records(self) -> List[Dict]:
        return [r for r in self.records if not r['correct']]

    # ─────────────────────────────────────────
    # 可视化
    # ─────────────────────────────────────────

    def plot_confusion_matrix(self, save_path: str = None):
        cm, labels = self.confusion_matrix()
        display_labels = [LABEL_NAMES.get(l, l) for l in labels]
        n = len(labels)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(display_labels, rotation=30, ha='right')
        ax.set_yticklabels(display_labels)
        ax.set_xlabel('预测标签')
        ax.set_ylabel('真实标签')
        ax.set_title('混淆矩阵')

        thresh = cm.max() / 2
        for i in range(n):
            for j in range(n):
                color = 'white' if cm[i, j] > thresh else 'black'
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center', color=color, fontsize=12)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"[✓] 混淆矩阵已保存: {save_path}")
        plt.show()

    def plot_accuracy_bar(self, save_path: str = None):
        stats  = self.accuracy_by_class()
        labels = [LABEL_NAMES.get(c, c) for c in self.classes]
        accs   = [stats[c]['accuracy'] * 100 for c in self.classes]

        fig, ax = plt.subplots(figsize=(9, 5))
        colors  = ['#4CAF50' if a >= 80 else '#FF9800' if a >= 50 else '#F44336'
                   for a in accs]
        bars = ax.bar(labels, accs, color=colors, width=0.5, edgecolor='white')
        ax.axhline(self.overall_accuracy() * 100, color='steelblue',
                   linewidth=1.5, linestyle='--', label=f'总体准确率 {self.overall_accuracy()*100:.1f}%')
        ax.set_ylim(0, 110)
        ax.set_ylabel('准确率 (%)')
        ax.set_title('各类别识别准确率')
        ax.legend()

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2, f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"[✓] 准确率柱状图已保存: {save_path}")
        plt.show()

    # ─────────────────────────────────────────
    # 误差分析报告
    # ─────────────────────────────────────────

    def _infer_error_reason(self, record: Dict) -> str:
        """简单推断误差原因"""
        if record.get('hu_dist') and record['hu_dist'] > 8:
            return '倾斜角度大'
        if record.get('tm_score') and record['tm_score'] < 0.2:
            return '图像模糊'
        if record['pred'] != 'unknown':
            return '相似类别混淆'
        return '颜色偏差'

    def generate_report(self, save_dir: str = 'results/report'):
        """生成文字报告"""
        os.makedirs(save_dir, exist_ok=True)
        lines = []

        # 总体统计
        overall = self.overall_accuracy()
        comp    = self.accuracy_tilted_vs_ideal()
        lines += [
            "=" * 60,
            "          交通标志识别系统评估报告",
            "=" * 60,
            f"\n【总体识别准确率】: {overall*100:.2f}%  "
            f"({sum(r['correct'] for r in self.records)}/{len(self.records)})\n",
            f"  理想图像准确率: {comp['ideal_acc']*100:.2f}%"
            f"  ({comp['ideal_n']} 张)",
            f"  倾斜图像准确率: {comp['tilted_acc']*100:.2f}%"
            f"  ({comp['tilted_n']} 张)\n",
        ]

        # 各类别准确率
        lines.append("【各类别准确率】")
        stats = self.accuracy_by_class()
        for cls in self.classes:
            s = stats[cls]
            name = LABEL_NAMES.get(cls, cls)
            lines.append(f"  {name:8s}: {s['accuracy']*100:6.2f}%"
                         f"  ({s['correct']}/{s['total']})")

        # 误差分析
        errors = self.error_records()
        lines += [f"\n【误差案例分析】（共 {len(errors)} 例）", "-" * 40]
        reason_count = defaultdict(int)
        for r in errors:
            reason = self._infer_error_reason(r)
            reason_count[reason] += 1
            tname = LABEL_NAMES.get(r['true'], r['true'])
            pname = LABEL_NAMES.get(r['pred'], r['pred'])
            lines.append(
                f"  文件: {os.path.basename(r['path'])}\n"
                f"    真实={tname}  预测={pname}  "
                f"置信度={r['confidence']:.2f}\n"
                f"    推断原因: {reason} — {ERROR_REASONS.get(reason,'')}"
            )

        lines += ["\n【误差原因统计】"]
        for reason, cnt in sorted(reason_count.items(),
                                  key=lambda x: -x[1]):
            lines.append(f"  {reason}: {cnt} 例")

        report_text = "\n".join(lines)
        report_path = os.path.join(save_dir, 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(report_text)
        print(f"\n[✓] 报告已保存至: {report_path}")

        # 绘图
        self.plot_confusion_matrix(
            save_path=os.path.join(save_dir, 'confusion_matrix.png'))
        self.plot_accuracy_bar(
            save_path=os.path.join(save_dir, 'accuracy_bar.png'))

        return report_text