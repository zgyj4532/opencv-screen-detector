# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目概述

Screen Detector V3 - 基于 Python + OpenCV + CNN 的三类图像来源识别系统，分类：`natural`（自然图像）、`screenshot`（屏幕截图）、`screen_photo`（相机拍摄屏幕）。

## 常用命令

```bash
# 安装依赖
uv sync

# 安装训练依赖
uv sync --group train

# 启动 API 服务 (端口 8325)
uv run python main.py

# 只从 daily-package zip 抽出 screen_photo
uv run python -m trainer ingest

# 训练候选（不覆盖当前生产 checkpoint）
uv run python -m trainer train --id <candidate-id>

# 复现历史训练器
uv run python -m trainer train_legacy

# 导出当前 canonical checkpoint；候选应使用 finalize_export.py 从 exp/<candidate-id>/best.pth 显式导出
uv run python -m trainer export

# 运行测试
uv run pytest tests/ -v

# 代码检查
uv run ruff check .

# 代码格式化
uv run ruff format .
```

## 架构

### 核心模型：CNN + FFT + DWT Branch 融合

```
Image → ┌──────────────────┬───────────────────────────┐
        │ RGB Branch       │ Frequency Branch          │
        │ EfficientNet-B0  │ FFT 64 + DWT 64 → 256 dim │
        │ 1280 dim         │                           │
        └──────────────────┴───────────────────────────┘
                       ↓ 两分支各自 LayerNorm
                       ↓ Concat (1536 dim) → Classifier
```

- **Spatial Branch**: EfficientNet-B0 提取 RGB 空间特征 (1280维)
- **Frequency Branch**: FFT + DWT 频域特征 (256维)
- **Fusion**: 分支 LayerNorm → Concat → Dropout → MLP 分类器

### 训练流程

两阶段训练策略：
1. **Stage A**: 冻结 backbone，只训练分类头和频域分支 (6 epochs)
2. **Stage B**: 解冻 backbone 最后 3 个 MBConv stage 微调 (12 epochs)

关键组件：
- **损失函数**: Focal Loss (gamma=2.0, alpha=[1.0, 1.0, 1.5], label smoothing=0.05)
- **权重平滑**: EMA decay=0.999
- **最佳检查点**: 仅最大化 validation `best_metric = 0.5 * screen_photo_f1 + 0.3 * accuracy + 0.2 * macro_f1`；选定后再通过 `trainer/evaluation_sets/canary.json` 回归门槛
- **数据增强**: 透视变换、运动模糊、噪声、随机擦除等

### 推理流程

1. TTA (Test-Time Augmentation): 原图 + 水平翻转，概率平均
2. 低置信度拒绝: max_prob < 0.45 → 返回 unknown；真 OOD 能力必须用独立 OOD 标签集评测
3. 阈值后处理: screen_photo 概率 >= 0.60 → 强制判定为 screen_photo
4. 置信度分级: accept (>=0.92) / review (0.75-0.92) / ignore (<0.75)

## 目录结构

- `trainer/` - 发布训练、历史训练器、模型、导出与验证
- `trainer/evaluation_sets/` - Canary、Frozen challenge、Rolling error pool、真 OOD 与采集组元数据清单
- `inference/` - 推理系统 (predictor, model_loader, fft_service, api/)
- `shared/` - 共享模块 (fft_transform.py 训练/推理共用)
- `data/input/` - 数据集 (natural_photo/, screenshot/, screen_photo/, hard_negative/)
- `experiment/cnn_fft_dwt_ablation/` - 消融、发布训练 harness 与部署评估
- `tests/` - 单元、集成与本地回归测试

## 关键配置

发布训练配置在 `trainer/release_train.py` 与 `experiment/cnn_fft_dwt_ablation/harness.py`：
- `IMAGE_SIZE = 224`, `BATCH_SIZE = 16`
- `LEARNING_RATE = 1e-3`, `WEIGHT_DECAY = 1e-4`
- `FOCAL_LOSS_GAMMA = 2.0`, `LABEL_SMOOTHING = 0.05`, `UNFREEZE_STAGES = 3`, Canary sampler weight = 2.0（底层 `focus_weight` 仅兼容历史 checkpoint）

`trainer/config.py` 仅服务 `train_legacy` 与 PAH-ViT 研究路径，不代表发布默认值。

推理配置在 `inference/config.py`：
- 使用 Pydantic Settings，可通过 `configure()` 运行时覆盖
- `ood_threshold = 0.45`, `screen_photo_threshold = 0.60`, `confidence_high = 0.92`, `confidence_medium = 0.75`

## 代码规范

- 使用 `uv` 管理依赖和运行命令
- Ruff 格式化: line-length=120, 双引号, LF 换行
- 类型注解: 使用 Python 3.11+ 语法 (X | Y 而非 Optional[X])
- 导入顺序: isort 风格, 相对导入允许 (TID252 忽略)

## 数据流

训练: `data/input/{类别目录}/` → 去重/固定分层切分 → `CachedDataset` → DataLoader → 模型
推理: 图片 → `normalize_rgb()` + `FFTService.get_fft_input()` → ONNX → 后处理

模型路径: `inference/models/three_class.onnx`
