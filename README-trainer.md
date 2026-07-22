# Screen Detector V3 - Trainer

CNN + FFT + DWT Branch 三分类训练系统。

## 功能

- 三分类训练 (natural / screenshot / screen_photo)
- EfficientNet-B0 + FFT Frequency Branch + DWT Wavelet Branch 融合模型
- Mixed Precision Training (AMP)
- Focal Loss (γ=2.0) + label smoothing + EMA 处理类别不平衡
- WeightedRandomSampler 过采样
- 本地 hard-example 清单与发布回归门槛
- 数据增强 (albumentations)
- ONNX / TorchScript 模型导出
- 配套 `experiment/cnn_fft_dwt_ablation/` 消融实验 harness（15 配置矩阵 + 自动选优）

## 快速开始

```bash
uv sync --group train

# 训练三分类发布模型（消融胜出配置，自动缓存/分层切分/备份 checkpoint）
uv run python -m trainer train

# 仅用于复现历史结果的旧训练器
uv run python -m trainer train_legacy

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
├── config.py           # 历史训练器与共享路径配置
├── release_train.py    # 当前发布训练入口
├── hard_examples.txt   # 已确认误判样本（固定进 train + 加权）
├── model.py            # 融合模型 (EfficientNet + FFT + DWT Branch)
├── fft_branch.py       # Frequency Branch (ResBlock)
├── dataset.py          # 三输入数据集 (RGB + FFT + DWT)
├── train.py            # 历史两阶段训练器（train_legacy）
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
screen_photo/   →  "screen_photo"
hard_negative/  →  按子目录名称恢复真实标签（仅 train，不入 val/test）
```

例如 `screenshot_to_screen_photo/` 的真实标签仍是 `screenshot`，
`screen_photo_to_screenshot/` 的真实标签仍是 `screen_photo`；UI、IDE、海报等通用难例映射为 `screenshot`。
发布管线按规范化路径去重，不会再把同一 hard-negative 同时加载成两个冲突标签。

`trainer/hard_examples.txt` 记录已经人工确认的线上/本地误判。清单中的图片必须存在于主数据集，发布切分会确保它们只进入 train，当前发布默认使用 2× sampler 权重；最佳 checkpoint 先要求这些回归样本全部正确，再按验证集指标选优。

**当前数据集统计** (2026-07-22):
- 总样本: **2,928 张** (去重后)
- 切分: seed=42 严格 stratified **0.70 / 0.15 / 0.15 = train 2,194 / val 366 / test 368**（**无样本级泄漏**）
- 数据集指纹: `6367b7638c3871a81e05e4ca41f2bf87ed12d711ff1ac0ad9c96a3495a6acc2a`

## 模型架构

```
Input Image (224x224)
    │
    ├─→ RGB Branch: EfficientNet-B0 → LayerNorm(1280)
    │
    └─→ Frequency Branch
         ├─ FFT Spectrum → CNN → 64
         └─ Haar DWT → CNN → 64
                       ↓ fusion → LayerNorm(256)
                    ↓
        Concat RGB 1280 + Frequency 256 = 1536
                    ↓
        Dropout→Linear(1536,3)
                    ↓
        natural / screenshot / screen_photo
```

## 训练策略（当前发布默认配置）

### 两阶段训练

**Stage A: 分类头训练** (6 epochs)
- 冻结 backbone
- 仅训练 classifier + freq_branch + spatial_norm + freq_norm
- 学习率: 1e-3

**Stage B: 微调** (12 epochs)
- 解冻 backbone 最后 **3 个 MBConv stage**（当前 2026-07-22 发布候选）
- 差异化学习率 (backbone: 1e-4, classifier: 1e-3)
- CosineAnnealingLR 调度器
- **EMA (decay=0.999) 全程开启**

### 损失函数

**Focal Loss** (处理类别不平衡，沿用消融验证后的稳定设置):
- gamma = **2.0**（消融前为 3.0，已下调）
- alpha = [1.0, 1.0, 1.5] (natural, screenshot, screen_photo)
- **label_smoothing = 0.05**

### 最佳模型选择

```python
best_metric = 0.5 * screen_photo_f1 + 0.3 * accuracy + 0.2 * macro_f1
```

发布训练使用字典序门槛：先最大化 `hard_examples` 的通过数（目标为全部通过），再最大化上述验证指标。这样新增的已确认回归不会被一个总体指标略高、但仍误判目标图片的早期 checkpoint 覆盖。

## 配置

发布训练参数定义在 `trainer/release_train.py`，模型/数据实现位于
`experiment/cnn_fft_dwt_ablation/harness.py`。`trainer/config.py` 与
`trainer/train.py` 保留给 `train_legacy` 和 PAH-ViT 研究流程，不代表当前发布默认值。

```python
# 模型
MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 224
NUM_CLASSES = 3

# 训练
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS_HEAD = 6
EPOCHS_FINETUNE = 12
SPLIT = (0.70, 0.15, 0.15)  # 主数据严格分层；hard-negative 只进 train

# 类别权重
CLASS_WEIGHTS_THREE_CLASS = [1.0, 1.0, 1.5]

# Focal Loss / 正则化
FOCAL_LOSS_GAMMA = 2.0
LABEL_SMOOTHING = 0.05
EMA_DECAY = 0.999
UNFREEZE_STAGES = 3

# 过采样
USE_WEIGHTED_SAMPLER = True
HARD_EXAMPLE_WEIGHT = 2.0

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
- `release_training_result.json` - 发布配置、切分指纹、选择 epoch 与 val/test 指标
- `three_class_confusion_matrix.png` - 旧训练器生成的混淆矩阵（如运行 `train_legacy`）
- `three_class_training_history.png` - 旧训练器生成的训练曲线（如运行 `train_legacy`）

**可复现缓存** (`experiment/cnn_fft_dwt_ablation/`，不纳入 Git):
- `split.json` - 含数据/难例指纹的固定分层切分；数据变化时自动重建
- `cache/*.npz` - 以路径、大小和 mtime 版本化的 FFT/DWT 特征
- `exp/<release-id>/history.json` - 每个 epoch 的指标与 hard-example 通过数

**导出模型** (`inference/models/`):
- `three_class.onnx` - ONNX 模型 (推荐, 22.2 MB)
- `three_class.torchscript` - TorchScript 临时导出（不作为仓库发布工件）

## 最新训练结果（2026-07-22 部署）

**发布 ID**: `candidate_20260722_unf3_focus2_6x12`，选择 `finetune-11`，hard-example 门槛 **2/2**

**训练数据**: 2,928 张，seed=42 严格 stratified 切分，train/val/test = 2,194/366/368

**训练时长**: 848 秒（不含首次缓存和 RAM 预载；RTX 3060 Laptop 6GB）

| 评估路径 | Accuracy | screen_photo Precision | screen_photo Recall | screen_photo F1 | Macro F1 | Metric |
|---|---:|---:|---:|---:|---:|---:|
| Validation checkpoint（argmax） | 0.9235 | 0.8448 | 0.8448 | 0.8448 | 0.9076 | 0.8810 |
| PyTorch checkpoint（argmax） | **0.9429** | 0.9000 | **0.9153** | **0.9076** | **0.9361** | **0.9239** |
| 生产 ONNX（TTA + OOD + 阈值） | 0.9158 | **0.9286** | 0.8814 | 0.9043 | 0.9262 | 0.9121 |

生产 ONNX 的 `unknown` 直接按错误计数，与 API 行为一致；最近一次 CPU TTA 单跑诊断为 mean 351 ms / p50 334 ms / p95 454 ms。

**两张目标 screenshot 的 ONNX+TTA 回归**:

| 图片 | 训练前 | 训练后 | screenshot 概率 |
|---|---|---|---:|
| `4a6e…ae8f9.png` | `screen_photo` | **`screenshot`** | 0.5552 |
| `5cdc3…12a62.png` | `natural` | **`screenshot`** | 0.4679 |

**与上一生产 ONNX 的受控部署路径对照**（同一当前 368 张清单、同一 TTA/OOD/阈值）：

| 候选 | Accuracy | screen_photo F1 | Macro F1 | Metric | hard-example 门槛 | 发布资格 |
|---|---:|---:|---:|---:|---:|---|
| 上一生产模型（`4b419976…`） | **0.9402** | **0.8870** | **0.9376** | **0.9131** | 0/2 | 无 |
| README 上一版发布（`cbba84bd…`） | 0.9158 | 0.8598 | 0.9145 | 0.8875 | 2/2 | 有 |
| 当前发布（`96aedc9f…`） | 0.9158 | **0.9043** | 0.9262 | 0.9121 | **2/2** | **有** |

当前发布相对 README 上一版可发布模型提升 +4.45 pp screen_photo F1、+1.16 pp macro F1、+2.46 pp metric。更早的 `4b419976…` 工件普通 metric 仍略高，但不满足两张指定截图必须正确的发布条件。检查点/工件选择采用字典序门槛：先判断 hard examples 是否全部通过，再在合格候选中比较 metric。

**相对 README 历史值**（acc 0.8946, macro_f1 0.8668, sp_f1 0.7630）：

- accuracy **+2.1pp**
- macro_f1 **+5.9pp**
- screen_photo F1 **+14.1pp**

> README 历史值的训练切分和 `hard_negative` 处理与当前发布不同，只能作为背景，不替代上方同部署路径的工件对照。

历史 3 种子 finalist 结果仍保留在 `experiment/cnn_fft_dwt_ablation/REPORT.md`；其切分和 hard-example 策略与当前发布不同，不再标作部署指标。

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
