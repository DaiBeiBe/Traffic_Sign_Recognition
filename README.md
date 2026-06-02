# 🚦 交通标识识别项目

> 基于颜色空间定位、二值化、胡矩特征与模板匹配的交通标志识别系统

## 📖 项目简介

本项目实现了一套完整的交通标志检测与识别流程，支持**五类**理想交通标志（禁止、警告、指示、指路、停车）的识别，并能处理一定角度倾斜的交通标志图像。系统采用颜色空间进行区间定位，结合二值化与形态学操作，提取**Hu矩**特征，最后通过模板匹配或特征比对完成分类。

---

## 📁 项目结构

```
traffic_sign_recognition/
│
├── data/                        # 数据集目录
│   ├── raw/                     # 原始图像
│   │   ├── ideal/               # 理想交通标志（5类）
│   │   │   ├── prohibit/        # 禁止类（红圈）
│   │   │   ├── warning/         # 警告类（黄色三角）
│   │   │   ├── mandatory/       # 指示类（蓝圆）
│   │   │   ├── guide/           # 指路类（矩形）
│   │   │   └── stop/            # 停车类
│   │   └── tilted/              # 倾斜交通标志（每类≥2幅）
│   └── templates/               # 模板图像库
│
├── src/                         # 核心源码
│   ├── preprocess.py            # 预处理模块
│   ├── localization.py          # 区间定位模块（颜色空间）
│   ├── binarization.py          # 二值化模块
│   ├── feature_extraction.py    # 特征提取模块（Hu矩）
│   ├── recognition.py           # 识别模块（Hu矩/模板匹配）
│   ├── evaluation.py            # 评估模块（准确率/误差分析）
│   └── data_utils.py        # 数据扫描、合成图像生成、模板注册
│
├── results/                     # 结果输出
│   ├── detected/                # 检测结果图像
│   └── report/                  # 准确率统计与误差分析报告
│
├── main.py                      # 主程序入口
└── requirements.txt             # 依赖库列表
```

---

## 🧩 模块说明

| 模块     | 文件                    | 功能描述                            |
| -------- | ----------------------- | ----------------------------------- |
| 预处理   | `preprocess.py`         | 图像缩放、去噪、光照归一化          |
| 区间定位 | `localization.py`       | 基于颜色空间（RGB/HSV）提取候选区域 |
| 二值化   | `binarization.py`       | 自适应阈值、形态学闭运算            |
| 特征提取 | `feature_extraction.py` | 计算Hu不变矩特征                    |
| 识别     | `recognition.py`        | 模板匹配 + 最小距离分类器           |
| 评估     | `evaluation.py`         | 统计识别准确率，生成误差分析报告    |

---

## 🔧 安装与配置

### 环境要求

- Python 3.7+
- OpenCV 4.5+
- NumPy
- Matplotlib（用于结果可视化）

### 安装步骤

1. 克隆仓库

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 准备数据集  
   将原始图像放置于 `data/raw/ideal/` 对应类别文件夹中，倾斜图像放置于 `data/raw/tilted/`。

---

## 🚀 使用方法

### 运行完整流程

```bash
python main.py --mode full --input data/raw/tilted --output results/
```

### 单独执行某一步骤

```bash
# 仅定位与二值化
python src/localization.py --image path/to/image.jpg

# 仅特征提取
python src/feature_extraction.py --image path/to/binary.jpg

# 识别单张图像
python src/recognition.py --image path/to/sign.jpg --template-dir data/templates/
```

### 评估模型

```bash
python src/evaluation.py --ground-truth data/raw/ideal --test-set data/raw/tilted
```

---

## 📊 结果输出

- **检测图像**：保存在 `results/detected/`，包含原始图像、定位框、识别标签。
- **统计报告**：`results/report/accuracy.txt` 包含总体准确率、每类准确率、混淆矩阵。
- **误差分析**：`results/report/error_analysis.txt` 记录识别失败的案例及可能原因。

---

## ✨ 特性亮点

- ✅ 支持**五类**交通标志，覆盖常见形状（圆形、三角形、矩形）
- ✅ 对**倾斜图像**具有鲁棒性（≥2幅/类测试）
- ✅ 使用**Hu矩**实现旋转/缩放不变特征
- ✅ 可选**模板匹配**或**特征距离**两种识别策略
- ✅ 自动生成**准确率统计**与**误差报告**

---

## 🗂 数据集说明

| 类别   | 文件夹名    | 形状       | 颜色特征              |
| ------ | ----------- | ---------- | --------------------- |
| 禁止类 | `prohibit`  | 圆形       | 红色圆环 + 斜杠       |
| 警告类 | `warning`   | 等边三角形 | 黄色背景 + 黑边       |
| 指示类 | `mandatory` | 圆形       | 蓝色背景 + 白色符号   |
| 指路类 | `guide`     | 矩形       | 绿/蓝底色 + 文字/箭头 |
| 停车类 | `stop`      | 正八边形   | 红色底色 + 白色文字   |

> 倾斜图像存储在 `tilted/` 下，每类至少包含2张不同倾斜角度的样本。

---

## 📝 待改进方向

- [ ] 增加深度学习模型（YOLO/CNN）提升复杂场景识别率
- [ ] 支持视频流实时检测
- [ ] 集成更多颜色空间（Lab、YCrCb）
- [ ] 添加图像增广模块以扩充数据集

---
