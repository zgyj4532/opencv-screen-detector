# Screen Detector V3 - Inference

单阶段 CNN + FFT Branch 三分类推理系统。

## 功能

- 单阶段 CNN 推理 (EfficientNet-B0 + FFT Branch, 3-class)
- OOD 检测 (unknown 类别)
- TTA (Test Time Augmentation)
- FFT 预处理缓存
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
├── README.md
├── models/
│   └── cnn_fft_3class.onnx    # 单阶段 3-class 模型
├── api/                # FastAPI 服务
├── config.py           # 配置 (模型路径/阈值/标签)
├── predictor.py        # 单阶段推理器
├── preprocess.py       # RGB 预处理
├── fft_transform.py    # FFT 频谱变换
├── batch_detect.py     # 批量检测
├── image_index.py      # 图片索引
└── scheduler.py        # 后台清理
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
  "image_id": "uuid",
  "filename": "test.jpg",
  "class_name": "screen_photo",
  "confidence": 0.9759,
  "probabilities": {"natural": 0.01, "screen_like": 0.01, "screen_photo": 0.98},
  "stage": 2,
  "confidence_tier": "high",
  "action": "accept"
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

## Python 调用

```python
from inference.predictor import ScreenDetectorPredictor

predictor = ScreenDetectorPredictor()
result = predictor.predict("path/to/image.jpg")

print(result["class"])        # natural/screenshot/screen_photo/unknown
print(result["confidence"])   # 0.0 - 1.0
print(result["confidence_tier"])  # high/medium/low/ood
print(result["action"])       # accept/review/ignore
```

## 推理流程

```
Image → CNN+FFT 3-class → natural/screenshot/screen_photo
                              ↓
                         OOD 检测 (max_prob < 0.45 → unknown)
                              ↓
                         screen_photo 阈值 (prob >= 0.35)
                              ↓
                         置信度分级 (accept/review/ignore)
```

## 配置

`inference/config.py` 中的关键配置:

- `model_path` - 3-class 模型路径 (`cnn_fft_3class.onnx`)
- `ood_threshold = 0.45` - OOD 检测阈值
- `confidence_high = 0.92` - 高置信度阈值
- `confidence_medium = 0.75` - 中置信度阈值
- `api_port = 8325` - API 端口
