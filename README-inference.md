# Screen Detector V3 - Inference

单阶段 CNN + FFT + DWT Branch 三分类推理系统。

## 功能

- 单阶段 CNN 推理 (EfficientNet-B0 + FFT + DWT Branch, 3-class)
- OOD 检测 (unknown 类别)
- TTA (Test Time Augmentation)
- FFT/DWT 预处理缓存
- 可配置阈值后处理（默认 sp_prob >= 0.60）
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

服务会在创建 `StreamingResponse` 之前预检文件数与未压缩总大小。超过 10,000 个文件或 20 GiB 时会直接返回普通 413 JSON 响应，不会在 ZIP 流已经开始后再抛异常；数据库查询上下文也会在流式传输前关闭。`tests/test_package.py` 的 10 个打包 API 测试均已通过。

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
│   1280 dim   │    64 dim    │    64 dim    │
└──────────────┴──────────────┴──────────────┘
   ↓ FFT + DWT 融合为 Frequency 256 dim
   ↓ Concat (1280 + 256 = 1536 dim)
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
screen_photo_threshold = 0.60  # >= 强制 screen_photo

# API
api_host = "0.0.0.0"
api_port = 8325

# 归一化 (ImageNet stats)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

## 模型性能

**`candidate_20260722_unf3_focus2_6x12` 生产 ONNX 端到端评估**（2026-07-22，368 张独立测试图，含 TTA/OOD/阈值）：

| Accuracy | Macro F1 | screen_photo Precision | screen_photo Recall | screen_photo F1 |
|---:|---:|---:|---:|---:|
| **0.9158** | **0.9262** | **0.9286** | 0.8814 | **0.9043** |

`unknown` 在上述指标中按错误分类，与真实 API 行为一致。置信度分布为 high 49 / medium 139 / low 168 / OOD 12；最近一次 CPU TTA 单跑诊断为 mean 351 ms / p50 334 ms / p95 454 ms。

两张已确认截图回归均已通过生产路径：`4a6e…ae8f9.png` 与 `5cdc3…12a62.png` 的预测均为 `screenshot`，概率分别为 0.5552 与 0.4679。

**与上一生产 ONNX 的同路径对照**（同一当前 368 张测试清单、同一 TTA/OOD/阈值）：

| 候选 | Accuracy | Macro F1 | screen_photo F1 | Metric | screenshot 门槛 | 发布资格 |
|---|---:|---:|---:|---:|---:|---|
| 上一生产模型（`4b419976…`） | **0.9402** | **0.9376** | **0.8870** | **0.9131** | 0/2 | 无 |
| README 上一版发布（`cbba84bd…`） | 0.9158 | 0.9145 | 0.8598 | 0.8875 | 2/2 | 有 |
| 当前模型（`96aedc9f…`） | 0.9158 | 0.9262 | **0.9043** | 0.9121 | **2/2** | **有** |

当前模型优于 README 上一版可发布模型，同时保持 hard-example 门槛 2/2。更早的 `4b419976…` 工件普通 metric 仍略高，但会把两张必须修复的截图分别判成 `screen_photo` 和 `natural`，不具备发布资格。串行 CPU 延迟受预热与系统负载影响，没有用于此次选择。

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
