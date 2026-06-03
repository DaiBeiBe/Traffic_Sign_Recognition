# 🚦 交通标志识别系统

> 基于 HSV 颜色定位、二值化、Hu 矩特征与模板匹配的交通标志检测与分类系统

---

## 📖 项目简介

本项目实现了一套完整的交通标志检测与识别流程，支持五类交通标志（禁止、警告、指示、指路、停车）的识别，并对倾斜图像具备一定的鲁棒性。

系统流程：

1. **预处理**：自适应光照增强 + 去噪 + 缩放至统一尺寸
2. **定位**：HSV 颜色空间掩码 + 轮廓筛选，提取感兴趣区域（ROI）
3. **倾斜矫正**：椭圆拟合透视矫正（蓝色圆形标志）+ 最小外接矩形旋转矫正（通用）
4. **二值化**：Otsu 与自适应阈值交集 + 形态学后处理
5. **特征提取**：对数变换 Hu 矩（7 维）+ 形状描述符（3 维）
6. **识别**：Hu 矩最近邻 / 模板匹配 / 融合投票，三种模式可选
7. **评估**：准确率统计、混淆矩阵、误差原因分析

---

## 📁 目录结构

```
traffic_sign_recognition/
│
├── data/
│   └── raw/
│       ├── ideal/                  # 理想图像，按类别分子文件夹
│       │   ├── prohibit/
│       │   ├── warning/
│       │   ├── mandatory/
│       │   ├── guide/
│       │   └── stop/
│       └── tilted/                 # 倾斜图像，文件名以类别名为前缀
│                                   # 例：prohibit_tilted_01.png
│
├── src/
│   ├── preprocess.py               # 预处理：读取、去噪、光照增强、缩放
│   ├── localization.py             # 定位：HSV 颜色掩码、轮廓筛选、倾斜矫正
│   ├── binarization.py             # 二值化：Otsu / 自适应 / 两者交集
│   ├── feature_extraction.py       # 特征提取：Hu 矩、形状特征、HOG（可选）
│   ├── recognition.py              # 识别：HuMomentClassifier、TemplateMatcher、TrafficSignRecognizer
│   ├── evaluation.py               # 评估：准确率统计、混淆矩阵、误差分析、可视化
│   └── data_utils.py               # 数据工具：扫描、校验、合成数据生成、模板注册
│
├── results/
│   ├── detected/                   # 带标注框的检测结果图像
│   └── report/                     # 评估报告、混淆矩阵图、准确率柱状图
│
├── main.py                         # 主程序入口
└── requirements.txt
```

---

## 🧩 模块说明

### `preprocess.py`

| 函数                                | 说明                                                                                   |
| ----------------------------------- | -------------------------------------------------------------------------------------- |
| `load_image(path)`                  | 读取图像，文件不存在或无法解码时抛出异常                                               |
| `resize_image(img, size=(128,128))` | 缩放至目标尺寸（INTER_AREA）                                                           |
| `denoise(img, method)`              | 去噪：`gaussian` / `median` / `bilateral`                                              |
| `gamma_correction(img, gamma)`      | 伽马校正，`gamma < 1` 提亮暗部                                                         |
| `enhance_contrast(img)`             | 自适应 CLAHE 增强：mean_V < 60 时先做伽马提亮，clipLimit 按亮度分档（4.0 / 3.0 / 2.0） |
| `preprocess_pipeline(path)`         | 完整流水线：增强对比度 → 去噪 → 缩放                                                   |

---

### `localization.py`

基于 HSV 颜色阈值定位候选区域，各类别阈值如下：

| 类别            | 颜色 | H 范围        | S 下限 | V 下限 |
| --------------- | ---- | ------------- | ------ | ------ |
| prohibit / stop | 红色 | 0–10、160–180 | 90     | 60     |
| warning         | 黄色 | 10–40         | 70     | 60     |
| mandatory       | 蓝色 | 95–135        | 60     | 30     |
| guide           | 绿色 | 40–80         | 80     | 60     |

关键函数：

| 函数                                   | 说明                                                                              |
| -------------------------------------- | --------------------------------------------------------------------------------- |
| `build_color_mask(hsv, sign_type)`     | 生成指定类别的颜色掩码，含形态学开/闭运算                                         |
| `filter_contours(contours, sign_type)` | 按面积、圆形度筛选轮廓；warning 类额外验证三角形形状                              |
| `build_mask_auto(hsv)`                 | 遍历所有类别，按「面积 × 形状奖励系数」评分选最优（圆形红色 ×1.4，圆形蓝色 ×1.2） |
| `correct_tilt(roi)`                    | 两步矫正：① 椭圆拟合透视矫正（长短轴比 < 0.82 时触发）② minAreaRect 旋转矫正      |
| `localize(img, sign_type)`             | 完整定位接口，返回 `(roi, mask, detected_type, bbox)`                             |

---

### `binarization.py`

| 函数                                    | 说明                                           |
| --------------------------------------- | ---------------------------------------------- |
| `otsu_binarize(gray)`                   | Otsu 自动阈值二值化                            |
| `adaptive_binarize(gray)`               | 高斯自适应阈值（block_size=21, C=5）           |
| `mask_binarize(roi, sign_type)`         | 用颜色掩码反转生成二值图                       |
| `morphology_clean(binary)`              | 形态学开运算去噪 + 闭运算填洞                  |
| `binarize_pipeline(roi, method='both')` | 默认 `'both'`：Otsu ∩ 自适应，光照不均时更鲁棒 |

---

### `feature_extraction.py`

| 函数                                                 | 说明                                                                  |
| ---------------------------------------------------- | --------------------------------------------------------------------- |
| `compute_hu_moments(binary)`                         | 计算 7 维 Hu 矩，对平移/旋转/缩放不变                                 |
| `log_transform_hu(hu)`                               | 对数变换缩小数量级差异，保留符号                                      |
| `compute_shape_features(binary)`                     | 3 维形状特征：圆形度、长宽比、凸包填充率                              |
| `compute_hog(gray)`                                  | HOG 特征（可选，默认关闭）                                            |
| `extract_features(binary, gray, use_hog, use_shape)` | 综合入口，默认返回 Hu 矩（7 维）+ 形状特征（3 维）= **10 维**特征向量 |

---

### `recognition.py`

包含三个类：

**`HuMomentClassifier`**：最近邻分类器，以欧氏距离衡量特征相似度。

**`TemplateMatcher`**：归一化互相关模板匹配（`TM_CCOEFF_NORMED`），查询图与模板均统一缩放至 64×64。

**`TrafficSignRecognizer`**：融合识别器，支持三种模式：

| 模式             | 说明                                                                                   |
| ---------------- | -------------------------------------------------------------------------------------- |
| `hu`             | 仅使用 Hu 矩，距离 ≤ 6.0 时输出结果                                                    |
| `template`       | 仅使用模板匹配，分数 ≥ 0.25 时输出结果                                                 |
| `fusion`（默认） | 两者均有效且一致 → 置信度 0.9；不一致 → Hu 矩优先，置信度 0.5；仅一者有效 → 输出该结果 |

---

### `evaluation.py`

| 方法                                                   | 说明                              |
| ------------------------------------------------------ | --------------------------------- |
| `add_result(path, true_label, pred_result, is_tilted)` | 记录一条识别结果                  |
| `overall_accuracy()`                                   | 总体准确率                        |
| `accuracy_by_class()`                                  | 各类别准确率                      |
| `accuracy_tilted_vs_ideal()`                           | 理想图 vs 倾斜图准确率对比        |
| `confusion_matrix()`                                   | 返回混淆矩阵及标签列表            |
| `plot_confusion_matrix(save_path)`                     | 绘制并保存混淆矩阵热力图          |
| `plot_accuracy_bar(save_path)`                         | 绘制各类别准确率柱状图            |
| `generate_report(save_dir)`                            | 生成完整文字报告 + 两张可视化图像 |

误差原因自动推断规则：hu_dist > 8 → 倾斜角度大；tm_score < 0.2 → 图像模糊；预测非 unknown → 相似类别混淆；其余 → 颜色偏差。

---

### `data_utils.py`

| 函数                                          | 说明                                                                          |
| --------------------------------------------- | ----------------------------------------------------------------------------- |
| `scan_dataset(data_dir)`                      | 扫描 `ideal/<cls>/` 与 `tilted/` 目录，返回 `{cls: [(path, is_tilted), ...]}` |
| `validate_dataset(dataset)`                   | 校验是否满足「每类 ≥ 3 张理想图、≥ 2 张倾斜图」的要求                         |
| `generate_synthetic_dataset(save_dir, ...)`   | 生成合成测试数据（每类可指定理想图/倾斜图数量，倾斜角随机 10°–35°）           |
| `register_all_templates(recognizer, dataset)` | 将理想图（默认）注册为模板，调用 `preprocess_pipeline` 后送入识别器           |

---

## 📊 数据集规格

| 类别   | 目录名      | 形状       | 颜色特征              |
| ------ | ----------- | ---------- | --------------------- |
| 禁止类 | `prohibit`  | 圆形       | 红色圆环 + 横杠       |
| 警告类 | `warning`   | 等边三角形 | 黄色背景 + 黑色感叹号 |
| 指示类 | `mandatory` | 圆形       | 蓝色背景 + 白色箭头   |
| 指路类 | `guide`     | 矩形       | 绿色背景 + 白色文字   |
| 停车类 | `stop`      | 正八边形   | 红色背景 + 白色 STOP  |

理想图存放于 `data/raw/ideal/<类别名>/`，每类至少 3 张。  
倾斜图存放于 `data/raw/tilted/`，文件名须以类别名开头（如 `warning_tilted_01.png`），每类至少 2 张。

---

## 🔧 安装

### 依赖

- Python 3.7+
- OpenCV 4.5+
- NumPy
- Matplotlib

```bash
pip install -r requirements.txt
```

---

## 🚀 使用方法

### 完整流程（使用合成数据）

```bash
python main.py
```

无真实数据时自动生成合成数据集（每类 3 张理想图 + 2 张倾斜图）并运行完整流程。

### 完整流程（使用真实数据）

```bash
python main.py --data data/raw
```

### 指定识别方法

```bash
# 仅使用 Hu 矩
python main.py --method hu

# 仅使用模板匹配
python main.py --method template

# 融合模式（默认）
python main.py --method fusion
```

### 强制重新生成合成数据

```bash
python main.py --synthetic
```

### 完整参数说明

| 参数          | 默认值             | 说明                                   |
| ------------- | ------------------ | -------------------------------------- |
| `--data`      | `data/raw`         | 数据集根目录                           |
| `--method`    | `fusion`           | 识别方法：`hu` / `template` / `fusion` |
| `--output`    | `results/detected` | 检测结果图像保存目录                   |
| `--synthetic` | —                  | 强制重新生成合成数据集                 |

---

## 📂 输出说明

运行结束后，结果保存在以下位置：

```
results/
├── detected/               # 每张输入图像对应一张带标注框的结果图
└── report/
    ├── evaluation_report.txt   # 文字报告（总体/各类/倾斜 vs 理想准确率 + 误差分析）
    ├── confusion_matrix.png    # 混淆矩阵热力图
    └── accuracy_bar.png        # 各类别准确率柱状图
```

---

## 📝 已知局限与改进方向

- 颜色阈值在极端光照（夜间、逆光）下仍可能失效
- 倾斜矫正仅处理平面旋转，不支持严重透视形变
- 合成数据与真实场景存在域偏移，建议补充真实样本后再评估
- 可扩展方向：
  - 接入深度学习骨干（YOLO / MobileNet）提升复杂场景准确率
  - 增加 Lab、YCrCb 颜色空间支持
  - 支持视频流实时检测
  - 数据增广模块（亮度抖动、高斯噪声、仿射变换）
