# Screen Detector V3 - Trainer

单阶段 CNN + FFT + DWT Branch 三分类训练系统。

## 功能

- 单阶段三分类训练 (natural/screenshot/screen_photo)
- EfficientNet-B0 + FFT Frequency Branch + DWT Wavelet Branch 融合模型
- Mixed Precision Training (AMP)
- Focal Loss 处理类别不平衡
- WeightedRandomSampler 过采样
- 数据增强 (albumentations)
- ONNX/TorchScript 模型导出

## 快速开始

```bash
uv sync --group train

# 训练三分类模型
uv run python -m trainer train

# 导出 ONNX 模型
uv run python -m trainer export
```

## 目录结构

```
trainer/
├── config.py           # 训练配置 (数据映射/超参数)
├── model.py            # 融合模型 (EfficientNet + FFT + DWT Branch)
├── fft_branch.py       # Frequency Branch (ResBlock)
├── dataset.py          # 三输入数据集 (RGB + FFT + DWT)
├── train.py            # 两阶段训练 (AMP)
├── validate.py         # 验证指标 (accuracy/precision/recall/f1/fpr)
├── augment.py          # 数据增强
├── losses.py           # Focal Loss
└── export_onnx.py      # ONNX 导出 (三输入 dynamic_axes)
```

## 数据要求

```
data/input/
├── natural_photo/      # 自然照片 (含子目录递归扫描)
├── screenshot/         # 截图 + 屏幕内容
├── screen_photo/       # 拍屏照片
└── hard_negative/      # 难例负样本 (UI 海报/深色模式/动漫 UI)
```

**数据映射**:

```
natural_photo/  →  "natural"
screenshot/     →  "screenshot"
hard_negative/  →  "screenshot"
screen_photo/   →  "screen_photo"
```

**当前数据集统计** (2026-06-30):
- natural_photo: 939 张
- screenshot: 1081 张
- screen_photo: 319 张
- hard_negative: 484 张
- **总计**: 2823 张

## 模型架构

```
Input Image (224x224)
    │
    ├─→ RGB Branch: EfficientNet-B0 → LayerNorm(1280)
    │
    ├─→ FFT Branch: FFT Spectrum → ResBlock×2 → LayerNorm(256)
    │
    └─→ DWT Branch: Haar Wavelet → ResBlock×2 → LayerNorm(256)
                    ↓
            Concat (1792 dim)
                    ↓
        Dropout→Linear(1792,512)→ReLU→Dropout→Linear(512,3)
                    ↓
        natural / screenshot / screen_photo
```

## 训练策略

### 两阶段训练

**Stage A: 分类头训练** (10 epochs)
- 冻结 backbone
- 仅训练 classifier + freq_branch + dwt_branch
- 学习率: 1e-3

**Stage B: 微调** (20 epochs)
- 解冻 backbone 最后 6 层
- 差异化学习率 (backbone: 1e-4, classifier: 1e-3)
- CosineAnnealingLR 调度器

### 损失函数

**Focal Loss** (处理类别不平衡):
- gamma = 3.0
- alpha = [1.0, 1.0, 1.5] (natural, screenshot, screen_photo)

### 最佳模型选择

```python
best_metric = 0.5 * screen_photo_f1 + 0.3 * accuracy + 0.2 * macro_f1
```

## 配置

`trainer/config.py` 中的关键配置:

```python
# 模型
MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 224
NUM_CLASSES = 3

# 训练
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 20
TRAIN_VAL_SPLIT = 0.8

# 类别权重
CLASS_WEIGHTS_THREE_CLASS = [1.0, 1.0, 1.5]

# Focal Loss
FOCAL_LOSS_GAMMA = 3.0
USE_FOCAL_LOSS = True

# 过采样
USE_WEIGHTED_SAMPLER = True
HARD_NEGATIVE_WEIGHT = 3

# 最佳指标权重
BEST_METRIC_F1_WEIGHT = 0.5
BEST_METRIC_ACCURACY_WEIGHT = 0.3
BEST_METRIC_MACRO_F1_WEIGHT = 0.2
```

## 输出

训练完成后生成:

**Checkpoints** (`trainer/checkpoints/`):
- `three_class_best.pth` - 最佳模型权重
- `three_class_final.pth` - 最终模型权重

**Logs** (`trainer/logs/`):
- `three_class_confusion_matrix.png` - 混淆矩阵
- `three_class_training_history.png` - 训练曲线

**导出模型** (`inference/models/`):
- `three_class.onnx` - ONNX 模型 (推荐)
- `three_class.torchscript` - TorchScript 模型

## 最新训练结果

**训练时间**: ~50 分钟 (RTX GPU)

**验证集指标**:

| 指标 | 值 |
|------|-----|
| Overall Accuracy | 88.68% |
| Macro F1 | 85.69% |

**各类别指标**:

| 类别 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| natural | 92.31% | 90.17% | 91.23% |
| screenshot | 92.26% | 89.49% | 90.85% |
| screen_photo | 69.23% | 81.82% | 75.00% |

## 环境要求

- Python >= 3.11, < 3.13
- CUDA 支持的 GPU（推荐）
- 至少 4GB GPU 显存

## 依赖

- torch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0
- albumentations >= 1.3.0
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- onnx >= 1.14.0
- onnxruntime >= 1.16.0
