# Screen Detector V3 - Trainer

CNN + FFT + DWT Branch 三分类训练系统。

## 功能

- 三分类训练 (natural / screenshot / screen_photo)
- EfficientNet-B0 + FFT Frequency Branch + DWT Wavelet Branch 融合模型
- Mixed Precision Training (AMP)
- Focal Loss (γ=2.0) + label smoothing + EMA 处理类别不平衡
- WeightedRandomSampler 过采样
- Canary / Frozen challenge / Rolling error pool / 真 OOD 分层评测
- 数据增强 (albumentations)
- ONNX / TorchScript 模型导出
- 配套 `experiment/cnn_fft_dwt_ablation/` 消融实验 harness（15 配置矩阵 + 自动选优）

## 快速开始

```bash
uv sync --group train

# 只从 daily-package zip 抽出 screen_photo（不落盘 normal_photo）
uv run python -m trainer ingest

# 审计内容级重复/标签冲突并生成或验证冻结评测清单
uv run python -m trainer audit

# 训练三分类发布候选（自动缓存/冻结切分；不会覆盖当前生产 checkpoint）
uv run python -m trainer train --id <candidate-id>

# 仅用于复现历史结果的旧训练器
uv run python -m trainer train_legacy

# 仅导出当前 canonical checkpoint；新候选应从 exp/<candidate-id>/best.pth 显式导出
uv run python -m trainer export

# 消融实验工作流（推荐先跑一遍）
uv run python experiment/cnn_fft_dwt_ablation/harness.py screen   # 跑 15 配置矩阵（~4 小时）
uv run python experiment/cnn_fft_dwt_ablation/show.py              # 查看排行榜
uv run python experiment/cnn_fft_dwt_ablation/run_candidate.py --validation-only --id clean_s42 --split-seed 42 --seed 42 --unfreeze 3 --canary-weight 2 --epochs-head 6 --epochs-finetune 12
uv run python experiment/cnn_fft_dwt_ablation/finalize_export.py trainer/checkpoints/three_class_best.pth inference/models/three_class.onnx
uv run python experiment/cnn_fft_dwt_ablation/deploy_eval.py       # 端到端部署验证
```

若单次执行时长受限，可加 `--max-total-epochs 12` 在完整 epoch 后安全暂停，再以完全相同参数和 ID 加 `--resume` 继续。`last.pth` 原子保存模型、EMA、优化器、调度器、AMP scaler、采样器、增强以及所有 RNG 状态；续训/不中断 1+1 epoch 控制实验的 history 与模型张量完全一致。

## 目录结构

```
trainer/
├── config.py           # 历史训练器与共享路径配置
├── release_train.py    # 当前发布训练入口
├── ingest_screen_photo.py # 只从 zip 抽出 screen_photo
├── data_audit.py       # 内容身份、标签决定与冻结切分审计入口
├── data_audit.json     # 最近一次机器可读数据审计报告
├── content_label_overrides.json # 按 SHA-256 记录的人工复核标签决定
├── evaluation_sets.py  # 评测集合、隔离审计与四类指标的深模块
├── evaluation_sets/    # Canary/challenge/rolling/OOD/group 元数据清单
├── hard_examples.txt   # 旧工具兼容指针；Canonical 清单已迁移
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

`trainer/evaluation_sets/canary.json` 记录已经人工确认的线上/本地误判。Canary 可留在 train 并使用 2× sampler 权重，但 checkpoint 只按 validation metric 选择；选定后再执行必须全过的 Canary 回归门禁。Canary 不承担泛化统计。`hard_examples.txt` 和 `focus_weight` 只保留为旧命令/checkpoint 兼容别名。

Frozen challenge 目标为 100-300 个独立内容/采集组，永不进入训练或阈值调节；Rolling error pool 只在审核后进入下一轮训练；真 OOD 集与三类闭集分开评测。三者清单框架已建立，但 challenge/OOD 当前没有样本，晋升状态为 `NOT_READY`。

**已审计当前数据视图**（2026-09-06）：

- 原始路径 3,075 个；唯一内容 2,943 个；按内容去除重复路径 132 个。
- 18 组原始标签冲突均已解决；19 条复核决定，无未使用决定。
- 冻结切分 seed=42：train 2,217 / val 361 / test 365；跨集合内容重合为 0，评测集 hard-negative 内容为 0。
- 当前数据集指纹：`d8e5e2030fb4cb29e6404abfc75f64ef5ce1a70bea7b0502814ff7325f192704`。
- 评测集指纹：`da74a983a7af3b5c1f73d1c80ccd5a7ed84a290e6f72dfb615a5eb6f73390eff`。
- 精确 SHA-256 跨 split 重合为 0；五个组字段覆盖均为 0/2,943，状态 `NOT_READY`。
- DCT pHash（汉明距离 ≤8）发现 19 个跨 split 候选对、17 个候选簇，状态 `REVIEW_REQUIRED`；候选不直接等于已证实泄漏。
- `uv run python -m trainer ingest` 从四份 daily-package zip 只写入 5 张唯一 `screen_photo`；`normal_photo` 未落盘。

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

发布训练只按上述 validation metric 选择 checkpoint。Canary 在选定后单独执行并阻止已知回归晋升，不参与 epoch 排序，也不作为泛化统计。最终晋升还必须分别满足 Frozen challenge、真 OOD、组级与感知近似隔离门禁。

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
CANARY_WEIGHT = 2.0  # 底层 focus_weight 字段仅用于旧 checkpoint 兼容

# 增量数据（默认关闭；2026-09-06 候选用 CLI 打开）
# remix_alpha=0.2  # Mixup/Remix，从零重训在本次 +5 样本上失败
# init_checkpoint / distill_alpha / boost_weight  # 生产热启动 + LwF + 新样本过采样

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
- `exp/<release-id>/history.json` - 每个 epoch 的 validation 指标与 Canary 回归结果

**导出模型** (`inference/models/`):
- `three_class.onnx` - ONNX 模型 (推荐, 22.2 MB)
- `three_class.torchscript` - TorchScript 临时导出（不作为仓库发布工件）

## 最新训练结果（2026-08-07 部署）

**发布 ID**：`release_20260807_unf3_focus2_6x12`，历史上选择 `finetune-12`、hard-example 门槛 **2/2**；这两张图现按 Canary 解释，不能证明泛化改善。

**发布训练快照**：数据指纹 `972fc082…`，原始路径 3,041 个、唯一内容 2,909 个；内容清洗冻结切分 seed=42，train/val/test = 2,183/361/365，跨集合内容重合为 0。

### Validation 选择

| 训练 seed | Accuracy | screen_photo F1 | Macro F1 | Metric | 门槛 |
|---:|---:|---:|---:|---:|---:|
| **42** | **0.9363** | **0.9550** | **0.9399** | **0.9463** | **2/2** |

该历史候选按旧规则选择；新规则只按 validation 选 checkpoint，随后执行 Canary，并将 365 张 test 视为历史闭集基准。

| 评估路径 | Accuracy | screen_photo Precision | screen_photo Recall | screen_photo F1 | Macro F1 | Metric |
|---|---:|---:|---:|---:|---:|---:|
| Validation 胜出指标（argmax） | 0.9363 | 0.9636 | 0.9464 | 0.9550 | 0.9399 | 0.9463 |
| 冻结 test，PyTorch（argmax） | **0.9425** | **0.8966** | **0.9123** | **0.9043** | **0.9340** | **0.9217** |
| 生产 ONNX（TTA + OOD + 阈值） | 0.9233 | 0.9123 | 0.9123 | 0.9123 | 0.9296 | 0.9190 |

生产 ONNX 的 10 个 `unknown` 直接按错误计数，与 API 行为一致；因 365 张均有已知类标签，它们是已知类误拒而不是真 OOD 命中。置信度分布为 93 high、148 medium、114 low、10 个低置信度拒绝；CPU TTA 单跑为 mean 353.8 ms / p50 327.4 ms / p95 471.8 ms。机器可读结果为 `experiment/cnn_fft_dwt_ablation/deploy_eval_release_20260807.json`。

**生产工件**：

- `three_class_best.pth` / `three_class_final.pth`：SHA-256 `cfd5c75c…`
- `three_class.onnx`：22.18 MB，SHA-256 `c53b00d5…`，已通过 PyTorch/ONNX parity
- 端到端结果：`experiment/cnn_fft_dwt_ablation/deploy_eval_release_20260807.json`

**两张目标 screenshot 的 ONNX+TTA 回归**：

| 图片 | 结果 | screenshot 概率 | screen_photo 概率 |
|---|---|---:|---:|
| `4a6e…ae8f9.png` | **`screenshot`** | 0.5818 | 0.2641 |
| `5cdc3…12a62.png` | **`screenshot`** | 0.5319 | 0.1690 |

2026-07-22 PyTorch argmax 历史值为 acc 0.9429 / sp_f1 0.9076 / macro_f1 0.9361 / metric 0.9239，数值与本次接近，但来自不同切分，不能当作受控优劣结论。当前版本以可审计的内容身份、标签、冻结评测集、validation 选种和一次性 test 作为发布依据。

### 2026-08-11 重训记录（未部署）

当前数据视图新增 29 个 train-only 唯一内容。seed=42 的 `release_20260811_unf3_focus2_6x12` 选择 finetune-7，validation 为 acc 0.9086 / sp_f1 0.8762 / macro_f1 0.9012 / metric 0.8909；训练 argmax 门槛 2/2，但导出 ONNX 经真实 Predictor 后两张目标截图均为 `unknown`，生产门槛 0/2。该候选在冻结 test 的生产路径为 acc 0.8849 / sp_f1 0.8364 / macro_f1 0.8964 / metric 0.8629，24 个 OOD。

seed=2024 的 `release_20260811_s2024_unf3_focus2_6x12` 选择 finetune-11，validation 为 acc 0.9224 / sp_f1 0.9000 / macro_f1 0.9181 / metric 0.9104；训练门槛 2/2，但生产门槛只有 1/2，因此 test 保持封存。两个候选都通过 PyTorch/ONNX parity，但都没有替换 2026-08-07 生产工件。

### 2026-09-06 候选（未部署）

+5 张 train-only `screen_photo` 后，从零 Remix Mixup（`candidate_20260906_remix_sp5`）test metric 0.9111、Canary 0/2。改为从生产 checkpoint 热启动 + LwF 蒸馏 + 新样本 8× 过采样（`candidate_20260906_lwf_sp5`，finetune-2）：

| 路径 | Accuracy | SP F1 | Macro F1 | Metric | Canary |
|---|---:|---:|---:|---:|---:|
| Validation | 0.9446 | 0.9636 | 0.9486 | **0.9549** | **2/2** |
| 冻结 test PyTorch argmax | 0.9479 | 0.9060 | 0.9389 | **0.9252** | — |
| 候选 ONNX Predictor | 0.9288 | 0.9217 | 0.9369 | **0.9269** | **2/2** |

高于 README 生产记录 0.9217 / 0.9190。生产 `three_class.onnx` 未覆盖。证据：`experiment/cnn_fft_dwt_ablation/deploy_eval_candidate_20260906_lwf_sp5.json`。

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
