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

# 审计内容级重复/标签冲突并生成或验证冻结评测清单
uv run python -m trainer audit

# 训练三分类发布模型（消融胜出配置，自动缓存/冻结切分/备份 checkpoint）
uv run python -m trainer train

# 仅用于复现历史结果的旧训练器
uv run python -m trainer train_legacy

# 导出 ONNX 模型
uv run python -m trainer export

# 消融实验工作流（推荐先跑一遍）
uv run python experiment/cnn_fft_dwt_ablation/harness.py screen   # 跑 15 配置矩阵（~4 小时）
uv run python experiment/cnn_fft_dwt_ablation/show.py              # 查看排行榜
uv run python experiment/cnn_fft_dwt_ablation/run_candidate.py --validation-only --id clean_s42 --split-seed 42 --seed 42 --unfreeze 3 --focus-weight 2 --epochs-head 6 --epochs-finetune 12
uv run python experiment/cnn_fft_dwt_ablation/finalize_export.py trainer/checkpoints/three_class_best.pth inference/models/three_class.onnx
uv run python experiment/cnn_fft_dwt_ablation/deploy_eval.py       # 端到端部署验证
```

若单次执行时长受限，可加 `--max-total-epochs 12` 在完整 epoch 后安全暂停，再以完全相同参数和 ID 加 `--resume` 继续。`last.pth` 原子保存模型、EMA、优化器、调度器、AMP scaler、采样器、增强以及所有 RNG 状态；续训/不中断 1+1 epoch 控制实验的 history 与模型张量完全一致。

## 目录结构

```
trainer/
├── config.py           # 历史训练器与共享路径配置
├── release_train.py    # 当前发布训练入口
├── data_audit.py       # 内容身份、标签决定与冻结切分审计入口
├── data_audit.json     # 最近一次机器可读数据审计报告
├── content_label_overrides.json # 按 SHA-256 记录的人工复核标签决定
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
发布管线按文件内容 SHA-256 去重，而不是只按路径去重。字节相同但标签不同的内容必须在
`trainer/content_label_overrides.json` 中有明确复核决定，否则审计与训练都会失败。只要某个内容身份曾出现在
`hard_negative/`，它及其他目录中的字节级副本都只进入 train。

`trainer/hard_examples.txt` 记录已经人工确认的线上/本地误判。清单中的图片必须存在于主数据集，发布切分会确保它们只进入 train，当前发布默认使用 2× sampler 权重；最佳 checkpoint 先要求这些回归样本全部正确，再按验证集指标选优。

**已审计当前数据视图**（2026-08-04）：

- 原始路径 3,021 个；唯一内容 2,889 个；按内容去除重复路径 132 个。
- 18 组原始标签冲突均已解决；19 条复核决定，无未使用决定。
- 冻结切分 seed=42：train 2,163 / val 361 / test 365；跨集合内容重合为 0，评测集 hard-negative 内容为 0。
- 当前数据集指纹：`6b75f6188ffb2c6f0ecb9fd8fadddb0d61dd7fa5edc112a8ea29283017e25aa5`。
- 评测集指纹：`da74a983a7af3b5c1f73d1c80ccd5a7ed84a290e6f72dfb615a5eb6f73390eff`。

`--split-seed` 只控制首次生成的冻结评测身份；`--seed` 控制模型初始化、采样器和增强。冻结后新增内容只进入 train，val/test 内容缺失或改标会直接失败。训练前执行 `uv run python -m trainer audit`。

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

**可复现清单与缓存**：

- `experiment/cnn_fft_dwt_ablation/split.json` - 纳入 Git 的相对路径冻结切分；新增内容不重排 val/test
- `trainer/content_label_overrides.json` - 纳入 Git 的内容级标签决定
- `trainer/data_audit.json` - 纳入 Git 的数据审计快照
- `cache/*.npz` - 以路径、大小和 mtime 版本化的 FFT/DWT 特征
- `exp/<release-id>/history.json` - 每个 epoch 的指标与 hard-example 通过数

**导出模型** (`inference/models/`):
- `three_class.onnx` - ONNX 模型 (推荐, 22.2 MB)
- `three_class.torchscript` - TorchScript 临时导出（不作为仓库发布工件）

## 最新训练结果（2026-08-04 部署）

**发布 ID**：`release_20260804_unf3_focus2_6x12`，选择 `finetune-9`，hard-example 门槛 **2/2**。

**发布训练快照**：数据指纹 `6b75f618…`，原始路径 3,021 个、唯一内容 2,889 个；内容清洗冻结切分 seed=42，train/val/test = 2,163/361/365，跨集合内容重合为 0。

### Validation 选择

| 训练 seed | Accuracy | screen_photo F1 | Macro F1 | Metric | 门槛 |
|---:|---:|---:|---:|---:|---:|
| **42** | **0.9363** | **0.9369** | **0.9359** | **0.9365** | **2/2** |

候选选择只看 hard-example 门槛和 validation；通过后一次性打开冻结 test。

| 评估路径 | Accuracy | screen_photo Precision | screen_photo Recall | screen_photo F1 | Macro F1 | Metric |
|---|---:|---:|---:|---:|---:|---:|
| Validation 胜出指标（argmax） | 0.9363 | 0.9455 | 0.9286 | 0.9369 | 0.9359 | 0.9365 |
| 冻结 test，PyTorch（argmax） | **0.9233** | **0.9074** | **0.8596** | **0.8829** | **0.9137** | **0.9012** |
| 生产 ONNX（TTA + OOD + 阈值） | 0.8959 | 0.9020 | 0.8070 | 0.8519 | 0.9000 | 0.8747 |

生产 ONNX 的 16 个 `unknown` 直接按错误计数，与 API 行为一致。置信度分布为 63 high、141 medium、145 low、16 OOD；CPU TTA 单跑为 mean 333.4 ms / p50 317.1 ms / p95 410.1 ms。机器可读结果为 `experiment/cnn_fft_dwt_ablation/deploy_eval_release_20260804.json`。

**生产工件**：

- `three_class_best.pth` / `three_class_final.pth`：SHA-256 `642a5c8a…`
- `three_class.onnx`：22.18 MB，SHA-256 `aae260f3…`，已通过 PyTorch/ONNX parity
- 端到端结果：`experiment/cnn_fft_dwt_ablation/deploy_eval_release_20260804.json`

**两张目标 screenshot 的 ONNX+TTA 回归**：

| 图片 | 结果 | screenshot 概率 | screen_photo 概率 |
|---|---|---:|---:|
| `4a6e…ae8f9.png` | **`screenshot`** | 0.5231 | 0.3159 |
| `5cdc3…12a62.png` | **`screenshot`** | 0.4551 | 0.2836 |

2026-07-22 PyTorch argmax 历史值为 acc 0.9429 / sp_f1 0.9076 / macro_f1 0.9361 / metric 0.9239，数值与本次接近，但来自不同切分，不能当作受控优劣结论。当前版本以可审计的内容身份、标签、冻结评测集、validation 选种和一次性 test 作为发布依据。

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
