# Screen Detector V3

基于 Python + OpenCV + CNN 的三类图像来源识别系统。

## 系统架构

采用**单阶段 CNN + FFT + DWT Branch**架构，一次推理完成三分类：

```
Image
   ↓
┌──────────────┬──────────────┬──────────────┐
│   RGB Branch │  FFT Branch  │  DWT Branch  │
│ EfficientNet │  全局频域特征 │  局部频域特征 │
│   1280 dim   │   256 dim    │   256 dim    │
└──────────────┴──────────────┴──────────────┘
   ↓ Concat (1792 dim)
   ↓
natural / screenshot / screen_photo
   ↓
OOD 检测 (max_prob < 0.45 → unknown)
   ↓
screen_score 后处理 (宁可误报拍屏，不能漏报拍屏)
   ↓
置信度分级 (accept/review/ignore)
```

### 标签体系

| 标签 | 含义 | 包含内容 | 样本数 |
|------|------|----------|--------|
| `natural` | 真实自然图像 | 风景、人像、室内、动物、食物、街景、天空、树木 | 947 |
| `screenshot` | 屏幕内容 | 截图、PPT、IDE、UI、terminal、聊天记录、软件界面 | 1034 |
| `screen_photo` | 相机拍摄屏幕 | 手机拍摄的屏幕照片 | 282 |

> 注: 数据集包含 528 个 hard_negative 样本用于增强模型鲁棒性。

### 置信度分级

| 置信度 | 处理方式 |
|--------|----------|
| >= 0.92 | 直接输出 (accept) |
| 0.75 - 0.92 | 人工审核 (review) |
| < 0.75 | 忽略 (ignore) |
| < 0.45 | OOD 检测，返回 unknown |

### 模型性能

**最新训练结果** (2026-06-23, CNN+FFT+DWT):

**验证集指标**:

| 指标 | 值 |
|------|-----|
| Overall Accuracy | 81.07% |
| Macro Precision | 75.63% |
| Macro Recall | 85.81% |
| Macro F1 | 76.62% |

**各类别验证集指标**:

| 类别 | Precision | Recall | F1 | FPR |
|------|-----------|--------|-----|-----|
| natural | 92.43% | 91.94% | 92.18% | 3.74% |
| screenshot | 95.06% | 72.64% | 82.35% | 4.96% |
| screen_photo | 39.39% | **92.86%** | 55.32% | 15.87% |

**20% 随机样本测试**:

| 指标 | 值 |
|------|-----|
| Overall Accuracy | **93.57%** |
| screen_photo Recall | **89.09%** |

**训练配置**:
- 数据集: 2799 张图片 (train: 2239, val: 560)
- 架构: EfficientNet-B0 + FFT Branch + DWT Branch
- 损失函数: Focal Loss (gamma=3.0, alpha=[1,1,4])
- 最佳模型选择: `best_metric = 0.7 * screen_photo_recall + 0.3 * accuracy`
- 两阶段训练: Stage A (head only, 10 epochs) + Stage B (fine-tune, 40 epochs)
- 数据增强: 强化增强（透视变换、运动模糊、噪声、随机擦除等）
- 推理后处理: screen_score 阈值（宁可误报拍屏，不能漏报拍屏）
- 训练时间: ~94 分钟 (RTX GPU)

## 快速开始

### 安装依赖

```bash
uv sync
```

### 启动 API 服务

```bash
uv run python main.py
```

API 服务运行在 `http://localhost:8325`

### 测试接口

```bash
# 健康检查
curl http://localhost:8325/api/health

# 文件上传检测
curl -X POST http://localhost:8325/api/detect/upload \
  -F "file=@test.jpg"

# URL 检测
curl -X POST http://localhost:8325/api/detect \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/test.jpg"}'
```

## 项目结构

```
opencv-screen-detector/
├── main.py                         # API 入口
├── pyproject.toml                  # 推理端依赖
├── shared/                         # 共享模块
│   └── fft_transform.py            # FFT 频谱变换 (训练/推理共享)
├── inference/                      # 推理系统
│   ├── models/
│   │   ├── cnn_fft_3class.onnx     # 旧版 3-class 模型
│   │   ├── three_class.onnx        # 最新 3-class ONNX 模型
│   │   └── three_class.torchscript # TorchScript 模型
│   ├── config.py                   # 推理配置 (Settings dataclass)
│   ├── predictor.py                # 单阶段推理器 (TTA/OOD)
│   ├── model_loader.py             # ONNX 模型加载
│   ├── fft_service.py              # FFT 缓存服务 (LRU)
│   ├── preprocess.py               # RGB 预处理 (normalize_rgb)
│   ├── api/
│   │   ├── app.py                  # FastAPI 应用
│   │   ├── router.py               # API 路由
│   │   ├── predictor.py            # 预测器生命周期管理
│   │   ├── schema.py               # Pydantic 模型
│   │   └── utils.py                # 工具函数
│   ├── batch_detect.py             # 批量检测
│   ├── image_index.py              # 图片索引 (异步 I/O)
│   └── scheduler.py                # 后台清理
├── trainer/                        # 训练系统
│   ├── config.py                   # 训练配置
│   ├── model.py                    # 融合模型 (EfficientNet + FFT Branch)
│   ├── fft_branch.py               # Frequency Branch (ResBlock)
│   ├── dataset.py                  # 双输入数据集
│   ├── train.py                    # 单阶段训练 (AMP)
│   ├── validate.py                 # 验证指标
│   ├── augment.py                  # 数据增强
│   └── export_onnx.py              # ONNX 导出
├── tests/                          # 测试
│   ├── conftest.py
│   ├── test_fft_transform.py
│   ├── test_dataset.py
│   ├── test_package.py
│   └── test_classify_extracted.py
├── data/
    ├── input/
    │   ├── natural_photo/          # 自然照片
    │   ├── screenshot/             # 截图 + 屏幕内容
    │   ├── screen_photo/           # 拍屏照片
    │   └── hard_negative/          # 难例负样本
    └── upload/                     # API 上传缓存
```

## API 文档

### POST /api/detect/upload

文件上传检测。

**请求**: `multipart/form-data`，字段 `file`

**响应**:
```json
{
  "image_id": "hash",
  "is_screen": true
}
```

### POST /api/detect

URL 检测。

**请求**: `application/json`
```json
{"url": "https://example.com/test.jpg"}
```

### GET /api/health

健康检查。返回模型加载状态和错误信息。

### POST /api/package

打包指定时间戳之后的图片为 ZIP 文件。

**请求**: `application/json`
```json
{
  "after_timestamp": "2026-06-09T00:00:00Z"
}
```

**响应**: `application/zip` 流式下载

**ZIP 文件结构**:
```
images_YYYYMMDD_HHMMSS.zip
├── screen_photo/      # 屏幕拍摄图片
│   ├── hash1.jpg
│   └── hash2.png
└── normal_photo/      # 非屏幕图片
    ├── hash3.jpg
    └── hash4.webp
```

**性能优化**:
- ✅ 使用临时文件替代 BytesIO，内存占用稳定在 50-200MB
- ✅ 使用 `compresslevel=1` 降低 CPU 占用 70-90%
- ✅ 1MB 分块流式下载，支持 50GB+ 数据导出
- ✅ BackgroundTask 自动清理临时文件

**限制**:
| 参数 | 限制值 | 说明 |
|------|--------|------|
| `MAX_FILES` | 10,000 | 最大文件数量 |
| `MAX_EXPORT_SIZE` | 20GB | 最大导出大小 |
| `CHUNK_SIZE` | 1MB | 流式下载块大小 |

**错误响应**:
- `404`: 指定时间戳之后没有找到图片
- `413`: 导出超过文件数量或大小限制

### POST /api/classify

更新图片分类。

## 训练指南

```bash
# 安装训练依赖
uv sync --group train

# 训练三分类模型
uv run python -m trainer train

# 导出 ONNX 模型
uv run python -m trainer export
```

### 数据爬取

```bash
export UNSPLASH_ACCESS_KEY="your_key"
uv run scripts/fetch_natural_photos.py
```

## 测试

```bash
uv run pytest tests/ -v
```

## 依赖

### 推理端
- opencv-python-headless
- numpy
- pillow
- fastapi + uvicorn
- httpx
- onnxruntime

### 训练端
- torch + torchvision
- timm (EfficientNet)
- albumentations
- scikit-learn
- matplotlib

## License

MIT
