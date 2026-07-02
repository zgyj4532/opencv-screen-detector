# Screen Detector V3 - Inference

单阶段 CNN + FFT + DWT Branch 三分类推理系统。

## 功能

- 单阶段 CNN 推理 (EfficientNet-B0 + FFT + DWT Branch, 3-class)
- OOD 检测 (unknown 类别)
- TTA (Test Time Augmentation)
- FFT/DWT 预处理缓存
- 阈值后处理 (sp_prob >= 0.60)
- 置信度分级 (accept/review/ignore)
- FastAPI REST API 服务
- 批量检测支持

## 快速开始

```bash
# 安装依赖 (在项目根目录)
uv sync

# 启动 API 服务
uv run python main.py
```

## 目录结构

```
inference/
├── models/
│   ├── three_class.onnx        # 最新 3-class ONNX 模型
│   └── three_class.torchscript # TorchScript 模型
├── api/                        # FastAPI 服务
│   ├── app.py                  # FastAPI 应用
│   ├── router.py               # API 路由
│   ├── predictor.py            # 预测器生命周期管理
│   ├── schema.py               # Pydantic 模型
│   └── utils.py                # 工具函数
├── config.py                   # 配置 (模型路径/阈值/标签)
├── predictor.py                # 单阶段推理器
├── model_loader.py             # ONNX 模型加载
├── fft_service.py              # FFT/DWT 缓存服务
├── preprocess.py               # RGB 预处理
├── batch_detect.py             # 批量检测
├── image_index.py              # 图片索引
└── scheduler.py                # 后台清理
```

## API 接口

### POST /api/detect/upload

文件上传检测。

```bash
curl -X POST http://localhost:8325/api/detect/upload \
  -F "file=@test.jpg"
```

**响应:**

```json
{
  "image_id": "hash",
  "is_screen": true
}
```

### POST /api/detect

URL 检测。

```bash
curl -X POST http://localhost:8325/api/detect \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/test.jpg"}'
```

### GET /api/health

健康检查。

```bash
curl http://localhost:8325/api/health
```

**响应:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "load_error": null
}
```

### POST /api/package

打包指定时间戳之后的图片为 ZIP 文件。

```bash
curl -X POST http://localhost:8325/api/package \
  -H "Content-Type: application/json" \
  -d '{"after_timestamp": "2026-06-09T00:00:00Z"}'
```

## Python 调用

```python
from pathlib import Path
from inference.predictor import ScreenDetectorPredictor

# 初始化预测器
predictor = ScreenDetectorPredictor()

# 预测单张图片
result = predictor.predict(Path("path/to/image.jpg"))

print(result["class"])           # natural/screenshot/screen_photo/unknown
print(result["confidence"])      # 0.0 - 1.0
print(result["confidence_tier"]) # high/medium/low/ood
print(result["action"])          # accept/review/ignore
print(result["probabilities"])   # {"natural": 0.1, "screenshot": 0.2, "screen_photo": 0.7}
```

## 推理流程

```
Image (224x224)
   ↓
┌──────────────┬──────────────┬──────────────┐
│   RGB Branch │  FFT Branch  │  DWT Branch  │
│ EfficientNet │  全局频域特征 │  局部频域特征 │
│   1280 dim   │   256 dim    │   256 dim    │
└──────────────┴──────────────┴──────────────┘
   ↓ Concat (1792 dim)
   ↓
Classifier (3-class logits)
   ↓
Softmax → probabilities
   ↓
OOD 检测 (max_prob < 0.45 → unknown)
   ↓
阈值后处理:
  - if sp_prob >= 0.60 → screen_photo
  - else → argmax(natural, screenshot, screen_photo)
   ↓
置信度分级:
  - >= 0.92 → accept
  - 0.75-0.92 → review
  - < 0.75 → ignore
```

## 配置

`inference/config.py` 中的关键配置:

```python
# 模型路径
model_path = "inference/models/three_class.onnx"

# 图像处理
image_size = 224
input_channels = 3

# 类别名称
class_names = ["natural", "screenshot", "screen_photo"]

# 置信度阈值
confidence_high = 0.92    # >= accept
confidence_medium = 0.75  # >= review
ood_threshold = 0.45      # < unknown

# API
api_host = "0.0.0.0"
api_port = 8325

# 归一化 (ImageNet stats)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

## 模型性能

**20% 随机样本验证** (474 张):

| 指标 | 值 |
|------|-----|
| Overall Accuracy | **94.51%** |
| Macro F1 | **93.93%** |

**各类别指标**:

| 类别 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| natural | 98.43% | 94.47% | 96.41% |
| screenshot | 95.77% | 95.77% | 95.77% |
| screen_photo | 88.89% | 90.32% | 89.60% |

## 性能优化

- **TTA**: 水平翻转增强，平均概率提升稳定性
- **FFT/DWT 缓存**: LRU 缓存避免重复计算
- **阈值后处理**: sp_prob >= 0.60 优化 precision-recall 平衡
- **OOD 检测**: max_prob < 0.45 过滤低置信度预测

## 环境要求

- Python >= 3.11, < 3.13
- ONNX Runtime (CPU 或 GPU)

## 依赖

- opencv-python-headless
- numpy
- pillow
- fastapi + uvicorn
- httpx
- onnxruntime
