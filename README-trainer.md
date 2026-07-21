# Screen Detector V3 - Trainer

CNN + FFT + DWT Branch 三分类训练系统。

## 功能

- 三分类训练 (natural / screenshot / screen_photo)
- EfficientNet-B0 + FFT Frequency Branch + DWT Wavelet Branch 融合模型
- Mixed Precision Training (AMP)
- Focal Loss (γ=2.0) + label smoothing + EMA 处理类别不平衡
- WeightedRandomSampler 过采样
- 数据增强 (albumentations)
- ONNX / TorchScript 模型导出
- 配套 `experiment/cnn_fft_dwt_ablation/` 消融实验 harness（15 配置矩阵 + 自动选优）

## 快速开始

```bash
uv sync --group train

# 训练三分类模型（标准流程）
uv run python -m trainer train

# 导出 ONNX 模型
uv run python -m trainer export

# 消融实验工作流（推荐先跑一遍）
uv run python experiment/cnn_fft_dwt_ablation/harness.py screen   # 跑 15 配置矩阵（~4 小时）
uv run python experiment/cnn_fft_dwt_ablation/show.py              # 查看排行榜
uv run python experiment/cnn_fft_dwt_ablation/finalist.py --seeds 42 2024 7  # 决赛多种子
uv run python experiment/cnn_fft_dwt_ablation/finalize_export.py trainer/checkpoints/three_class_best.pth inference/models/three_class.onnx
uv run python experiment/cnn_fft_dwt_ablation/deploy_eval.py       # 端到端部署验证
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
└── hard_negative/      # 难例负样本 (UI 海报/深色模式/动漫 UI) - 仅 train
```

**数据映射**:

```
natural_photo/  →  "natural"
screenshot/     →  "screenshot"
hard_negative/  →  "screenshot" (仅 train，不入 val/test)
screen_photo/   →  "screen_photo"
```

**当前数据集统计** (2026-07-21):
- 总样本: **2,922 张** (去重后)
- 切分: seed=42 严格 stratified **0.70 / 0.15 / 0.15 = train 2,189 / val 364 / test 369**（**无样本级泄漏**）

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
            Concat (1536 dim)
                    ↓
        Dropout→Linear(1536,3)
                    ↓
        natural / screenshot / screen_photo
```

## 训练策略（消融验证后的推荐配置）

### 两阶段训练

**Stage A: 分类头训练** (10 epochs)
- 冻结 backbone
- 仅训练 classifier + freq_branch + spatial_norm + freq_norm
- 学习率: 1e-3

**Stage B: 微调** (20 epochs)
- 解冻 backbone 最后 **1 个 MBConv stage**（消融后从 6 个降为 1 个）
- 差异化学习率 (backbone: 1e-4, classifier: 1e-3)
- CosineAnnealingLR 调度器
- **EMA (decay=0.999) 全程开启**

### 损失函数

**Focal Loss** (处理类别不平衡，消融验证后的最优):
- gamma = **2.0**（消融前为 3.0，已下调）
- alpha = [1.0, 1.0, 1.5] (natural, screenshot, screen_photo)
- **label_smoothing = 0.05**

### 最佳模型选择

```python
best_metric = 0.5 * screen_photo_f1 + 0.3 * accuracy + 0.2 * macro_f1
```

## 配置

`trainer/config.py` 中的关键配置（已按消融结果调整）:

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
TRAIN_VAL_SPLIT = 0.85  # 0.70 train / 0.15 val / 0.15 test（消融 harness）

# 类别权重
CLASS_WEIGHTS_THREE_CLASS = [1.0, 1.0, 1.5]

# Focal Loss
FOCAL_LOSS_GAMMA = 2.0         # 消融前 3.0 → 2.0
USE_FOCAL_LOSS = True

# 正则化
LABEL_SMOOTHING = 0.05         # 消融验证 0.05 > 0 / 0.10
EMA_DECAY = 0.999              # 消融验证必开
UNFREEZE_STAGES = 1            # 消融前 6 → 1（小数据集过拟合）

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
- `three_class.onnx` - ONNX 模型 (推荐, 22.2 MB)
- `three_class.torchscript` - TorchScript 模型

## 最新训练结果（2026-07-21 部署）

**训练数据**: 2,922 张，seed=42 严格 stratified 切分，train/val/test = 2,189/364/369
**训练时长**: ~25 分钟 / 单 seed（RTX 3060 6GB）

**3 种子决赛结果**（10+20 epoch）:

| Seed | test_acc | test_sp_f1 | test_macro_f1 |
|---|---|---|---|
| **42（已部署）** | **0.9322** | **0.8548** | **0.9174** |
| 2024 | 0.9133 | 0.8276 | 0.8963 |
| 7 | 0.9079 | 0.7788 | 0.8817 |
| **AVG** | **0.9178** | **0.8204** | **0.8985** |

**ONNX 端到端测试集评估**（含 TTA，CPU 推理）:

| 指标 | 值 |
|---|---|
| Accuracy (raw) | 0.9079 |
| Accuracy (unknown→sp) | **0.9187** |
| screen_photo F1 (raw) | **0.8500** |
| screen_photo F1 (unknown→sp) | 0.8397 |
| screen_photo Recall (unknown→sp) | **0.9322** |
| Macro F1 | 0.9039 |
| 延迟 (mean / p50 / p95) | 234ms / 222ms / 313ms |

**vs 旧基线** (acc 0.8946, macro_f1 0.8668, sp_f1 0.7630):
- accuracy **+2.4pp**
- macro_f1 **+3.7pp**
- screen_photo F1 **+7.7pp**

> 新数据切分下，旧基线的 `hard_negative` 标签泄漏已修复，**对比数字比表面看起来更显著**。

## 环境要求

- Python >= 3.11, < 3.13
- CUDA 支持的 GPU（推荐 6GB+）
- Windows / Linux / macOS 均可（Windows 上 DataLoader num_workers 必须为 0）

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
