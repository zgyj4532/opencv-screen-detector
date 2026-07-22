# OpenCV Screen Detector

[![Python](https://img.shields.io/badge/python-3.11--3.12-blue)](pyproject.toml) [![PyTorch](https://img.shields.io/badge/training-PyTorch-ee4c2c)](https://pytorch.org/) [![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/) [![ONNX Runtime](https://img.shields.io/badge/inference-ONNX%20Runtime-005CED)](https://onnxruntime.ai/)

**English** | [简体中文](README_CN.md)

## Project Introduction

OpenCV Screen Detector classifies an image source as **natural**, **screenshot**, or **screen photo** (a camera photograph of a display). It pairs spatial appearance with FFT and Haar-DWT frequency features, and ships a FastAPI service backed by ONNX Runtime.

The repository contains the deployable EfficientNet-B0 + FFT + DWT model, its PyTorch training pipeline, and reproducible CNN/DeiT research experiments. It is a classifier, not an object detector: it predicts one label for each input image.

## Project Highlights

- Three-source classification with an ONNX model that accepts RGB, FFT, and DWT inputs.
- Shared OpenCV preprocessing keeps FFT/DWT transforms consistent between training and serving.
- TTA, low-confidence/OOD handling, and screen-photo thresholding are built into inference.
- FastAPI supports upload and URL detection, classification updates, health checks, and ZIP export.
- Research harness compares CNN, FFT, DeiT, and DWT+FFT+DeiT variants.
- **`experiment/cnn_fft_dwt_ablation/` ablation harness**: 15+ configs train sequentially, results land in a leaderboard, ONNX export with PyTorch parity check, end-to-end deployment verification.

## Table of Contents

<details>
<summary>Open the table of contents</summary>

- [Features](#features)
- [Project Architecture](#project-architecture)
- [Model Architecture](#model-architecture)
- [Image Classification Pipeline](#image-classification-pipeline)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [REST API](#rest-api)
- [Project Structure](#project-structure)
- [Experimental Results](#experimental-results)
- [Performance Comparison](#performance-comparison)
- [Model Evolution](#model-evolution)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgements](#acknowledgements)
</details>

## Features

| Area | Included capability |
| --- | --- |
| Classes | `natural`, `screenshot`, `screen_photo`; `unknown` for low-confidence inference |
| Training | Two-stage fine-tuning, AMP when CUDA is available, Focal Loss (γ=2) + label smoothing, EMA (decay 0.999), weighted sampling, hard negatives |
| Features | ImageNet-normalized RGB, log-magnitude FFT, Haar DWT sub-bands |
| Inference | ONNX Runtime, horizontal-flip TTA, LRU feature cache, confidence tiers |
| Service | FastAPI, image upload/URL ingestion, SQLite image index, streaming ZIP packages |
| Research | `experiment/cnn_fft_dwt_ablation/` ablation harness (15-config matrix, automatic leaderboard); CNN/FFT/DeiT ablations |
| Deployment verification | `experiment/cnn_fft_dwt_ablation/deploy_eval.py` end-to-end ONNX benchmark script |

## Project Architecture

```mermaid
flowchart TD
    Image[Image] --> API[FastAPI API or Python caller]
    API --> Ingest[Validate and persist image]
    Ingest --> Preprocess[OpenCV preprocessing]
    Preprocess --> RGB[RGB tensor]
    Preprocess --> Frequency[Shared FFT and Haar-DWT transforms]
    RGB --> Runtime[ONNX Runtime]
    Frequency --> Runtime
    Runtime --> Post[Flip-TTA average, OOD and threshold rules]
    Post --> Result[Class, confidence, probabilities, action]
    Dataset[Dataset folders] --> Trainer[PyTorch trainer]
    Trainer --> Checkpoint[PyTorch checkpoint]
    Checkpoint --> Export[ONNX export]
    Export --> Runtime
```

## Model Architecture

### Deployed CNN + FFT + DWT model

The default export is `ScreenDetectorModelWithDWT`. EfficientNet-B0 produces 1,280 spatial features. A two-stream frequency branch processes a one-channel FFT spectrum and four Haar-DWT sub-bands, then produces 256 features. Layer-normalized features are concatenated to 1,536 dimensions and classified into three logits.

```mermaid
graph TD
    RGB[RGB: B x 3 x 224 x 224] --> EN[EfficientNet-B0]
    EN --> SN[LayerNorm: 1280]
    FFT[FFT: B x 1 x 224 x 224] --> FCNN[FFT CNN: Conv, ResBlocks]
    DWT[DWT: B x 4 x 224 x 224] --> DCNN[DWT CNN: Conv, ResBlocks]
    FCNN --> FF[Frequency fusion]
    DCNN --> FF
    FF --> FN[LayerNorm: 256]
    SN --> CAT[Concat: 1536]
    FN --> CAT
    CAT --> Head[Dropout - Linear 512 - ReLU - Dropout]
    Head --> Out[3 logits]
```

### CNN + FFT research variant

This two-input variant remains available through `create_model(use_dwt=False)` and is used by the research code.

```mermaid
graph TD
    A[RGB image] --> B[EfficientNet-B0]
    B --> C[LayerNorm: 1280]
    D[FFT spectrum] --> E[Conv - ResBlock - Conv - ResBlock]
    E --> F[Pool and FC: 256]
    C --> G[Concat: 1536]
    F --> G
    G --> H[MLP classifier]
    H --> I[natural / screenshot / screen_photo]
```

### DWT + FFT + DeiT research variant

The experimental triple-stream model uses RGB DeiT-Small features, a second DeiT-Small stream for a replicated three-channel FFT spectrum, and a DWT CNN. It is not the model loaded by the default API.

```mermaid
graph TD
    R[RGB] --> RD[DeiT-Small]
    RD --> RF[RGB features: 384]
    F[FFT spectrum] --> Rep[Repeat 1 channel to 3]
    Rep --> FD[DeiT-Small]
    FD --> FF[FFT features: 384]
    R --> Haar[Haar DWT]
    Haar --> DC[DWT CNN]
    DC --> DF[DWT features: 256]
    RF --> Join[Concat: 1024]
    FF --> Join
    DF --> Join
    Join --> Classifier[MLP classifier]
    Classifier --> Logits[3 logits]
```

## Image Classification Pipeline

```mermaid
flowchart LR
    Input[Read image] --> Resize[Resize to 224 x 224]
    Resize --> Branch{Build inputs}
    Branch --> RGB[BGR to RGB and ImageNet normalization]
    Branch --> FFT[Grayscale FFT log magnitude]
    Branch --> DWT[Haar DWT: LL, LH, HL, HH]
    RGB --> Original[ONNX pass: original]
    FFT --> Original
    DWT --> Original
    RGB --> Flip[ONNX pass: horizontal flip]
    FFT --> Flip
    DWT --> Flip
    Original --> Average[Average class probabilities]
    Flip --> Average
    Average --> OOD{Maximum probability below 0.45?}
    OOD -- Yes --> Unknown[unknown / ignore]
    OOD -- No --> Threshold{screen_photo >= 0.60?}
    Threshold -- Yes --> Screen[screen_photo]
    Threshold -- No --> Argmax[Class argmax]
```

## Dataset

Place images under `data/input`. The dataset is local and is not distributed with this repository.

```mermaid
flowchart TD
    N[data/input/natural_photo] --> Natural[natural]
    S[data/input/screenshot] --> Screenshot[screenshot]
    H[data/input/hard_negative] --> HLabel[True class derived from subdirectory]
    P[data/input/screen_photo] --> Photo[screen_photo]
    Natural --> Split[seed=42 stratified 0.70 / 0.15 / 0.15 split]
    Screenshot --> Split
    Photo --> Split
    HLabel -.-> |train only| Split
    Split --> Augment[Train augmentation and feature generation]
    Split --> Validate[Validation and test transforms]
```

| Directory | Training label | Intended content |
| --- | --- | --- |
| `natural_photo/` | `natural` | Camera photos such as people, scenes, objects, and interiors |
| `screenshot/` | `screenshot` | Screenshots, UI, IDEs, slides, terminal windows, and chats |
| `hard_negative/` | Derived from subdirectory | Confirmed boundary cases keep their true class; train only, never in val/test |
| `screen_photo/` | `screen_photo` | Camera photographs of displays |

**Current deployment training data** (updated 2026-07-22):
- Total samples: **2,928 images** (deduplicated; includes hard_negative)
- Split: seed=42 strict stratified, **train 2,194 / val 366 / test 368** (**no sample-level leakage**)
- Dataset fingerprint: `6367b7638c3871a81e05e4ca41f2bf87ed12d711ff1ac0ad9c96a3495a6acc2a`
- The previous baseline leaked `hard_negative` samples across train/val with 3× weights as `screenshot`; the new pipeline removes that leakage.
- Confirmed local regressions in `trainer/hard_examples.txt` stay in train, receive explicit sampler weight, and act as a release-checkpoint gate.

Dataset composition changes over time; report the exact split with every new model.

## Installation

Requirements: Python `>=3.11,<3.13` and [uv](https://docs.astral.sh/uv/). Training is practical on a CUDA-capable GPU; serving can use ONNX Runtime on CPU.

```bash
# Run from this repository's root directory.
uv sync
```

The commands above work in PowerShell on Windows and in a POSIX shell on Linux/macOS. Install training dependencies only when needed:

```bash
uv sync --group train
```

## Quick Start

### Start the API

```bash
uv run python main.py
```

Open `http://127.0.0.1:8325/docs` for the generated OpenAPI interface.

### Run Python inference

```python
from pathlib import Path
from inference.predictor import ScreenDetectorPredictor

result = ScreenDetectorPredictor().predict(Path("path/to/image.jpg"))
print(result["class"], result["confidence"])
```

### Train and export ONNX

```bash
uv sync --group train
uv run python -m trainer train
uv run python -m trainer export
```

The exporter writes `inference/models/three_class.onnx`, which is the path used by the service.

## Training

### Two-stage training strategy

The release trainer initializes EfficientNet-B0 with pretrained weights, trains the head and frequency branch for 6 epochs, then unfreezes the last 3 MBConv stages for 12 fine-tuning epochs. It first requires every available sample in `trainer/hard_examples.txt` to classify correctly, then selects checkpoints with:

`0.5 * screen_photo_f1 + 0.3 * accuracy + 0.2 * macro_f1`

```mermaid
flowchart LR
    Data[Dataset folders] --> Load[Dataset: RGB, FFT, DWT]
    Load --> A[Stage A: frozen backbone, 6 epochs]
    A --> B[Stage B: unfreeze last 3 MBConv stages, 12 epochs]
    B --> Eval[Accuracy, precision, recall, F1]
    Eval --> Best[Best checkpoint]
    Best --> Export[ONNX and TorchScript export]
```

```bash
# Reproducible release training (clean split, versioned cache, hard-example gate)
uv run python -m trainer train
uv run python -m trainer export

# Historical trainer retained only for baseline reproduction
uv run python -m trainer train_legacy

uv run python -m trainer ablation --modules baseline,arcface,attention --epochs-head 5 --epochs-finetune 10
```

Training outputs belong in `trainer/checkpoints/` and `trainer/logs/`.

### Current release configuration

The 2026-07-22 release uses the same loss, EMA, backbone, augmentation, and FFT+DWT choices from the ablation sweep, but uses a newer controlled candidate setting that beat the README's previous deployable release on the current split:

| Setting | Value | Note |
| --- | --- | --- |
| Focal Loss γ | 2.0 | γ=3.0 costs ~3pp sp_f1 |
| Class α | [1.0, 1.0, 1.5] | α=2.0 slightly worse; α=2.5 collapses |
| Label smoothing | 0.05 | both 0 and 0.10 are worse |
| EMA | decay=0.999 | disabling costs 2.4pp sp_f1 |
| Unfrozen stages | 3 | current release candidate; earlier 15-config screen favored 1 stage |
| Hard-example sampler weight | 2.0 | selected from 1×/2×/4× candidates on 2026-07-22 |
| Epochs | 6 + 12 | faster current release run; selected checkpoint is finetune epoch 11 |
| Backbone | efficientnet_b0 | B1 gives no measurable gain on a 6GB GPU |
| Augmentation | moderate | strong aug costs ~5pp acc |
| FFT + DWT | both enabled | removing DWT costs 5pp sp_f1 |

### `experiment/cnn_fft_dwt_ablation/` ablation workflow

```bash
# 1) Launch the 15-config matrix (sequential, ~4h, writes to leaderboard.jsonl)
PYTHONUNBUFFERED=1 nohup uv run python -u experiment/cnn_fft_dwt_ablation/harness.py screen > experiment/cnn_fft_dwt_ablation/logs/screen.log 2>&1 &

# 2) View leaderboard (sorted by test_metric)
uv run python experiment/cnn_fft_dwt_ablation/show.py

# 3) Train the winner with multiple seeds for stability
uv run python experiment/cnn_fft_dwt_ablation/finalist.py --seeds 42 2024 7

# 4) Export ONNX and verify PyTorch <-> ONNX numerical parity
uv run python experiment/cnn_fft_dwt_ablation/finalize_export.py trainer/checkpoints/three_class_best.pth inference/models/three_class.onnx

# 5) End-to-end deployment verification on the current 368-image test set
uv run python experiment/cnn_fft_dwt_ablation/deploy_eval.py
```

`experiment/cnn_fft_dwt_ablation/` contains:
- `harness.py` — training loop, model, FFT/DWT cache, RAM preload, `ExpConfig` dataclass
- `finalist.py` — multi-seed training with auto-selection
- `finalize_export.py` — ONNX export with numerical parity check
- `deploy_eval.py` — end-to-end ONNX performance benchmark
- `show.py` — leaderboard viewer
- `leaderboard.jsonl` — all experimental records (JSONL append format)
- `REPORT.md` — full Chinese ablation report

`trainer/release_train.py` is the canonical wrapper around the current release configuration. It validates the dataset fingerprint, writes per-epoch history, backs up the previous checkpoint during training, and publishes `three_class_best.pth` only after a successful run. `trainer/train.py` remains available through `train_legacy` and does not define release defaults.

The optional PAH-ViT workflow is experimental:

```bash
uv run python -m trainer train_pahvit
uv run python -m trainer benchmark
uv run python -m trainer validate_pahvit
```

## Inference

The current ONNX graph has three named inputs: `rgb_input` (`B×3×224×224`), `fft_input` (`B×1×224×224`), and `dwt_input` (`B×4×224×224`). `ScreenDetectorPredictor` constructs all of them from a file path.

```bash
uv run python -m inference.batch_detect path/to/images inference/output/results.json
```

The predictor averages original and horizontally flipped predictions. It returns `unknown` when the maximum probability is below `0.45`; otherwise a `screen_photo` probability of at least `0.60` takes precedence, followed by argmax. Confidence actions are `accept` (≥0.92), `review` (≥0.75), and `ignore` (<0.75).

## REST API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Service and model-load status |
| `POST /api/detect/upload` | Upload an image and return whether it is a screen photo |
| `POST /api/detect` | Download an image URL and classify it |
| `POST /api/classify` | Update a stored image classification |
| `POST /api/package` | Stream a ZIP of indexed images after a timestamp |

```bash
curl http://127.0.0.1:8325/api/health

curl -X POST http://127.0.0.1:8325/api/detect/upload \
  -F "file=@path/to/image.jpg"

curl -X POST http://127.0.0.1:8325/api/detect \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/image.jpg"}'
```

Example detection response:

```json
{"image_id":"<sha256>","is_screen":true}
```

`/api/package` accepts an ISO-8601 timestamp, for example `{"after_timestamp":"2026-07-01T00:00:00Z"}`. The stream is limited to 10,000 files and 20 GiB before compression. Both limits are validated before the streaming response starts, so an oversized request returns a normal JSON 413 response.

## Project Structure

```text
opencv-screen-detector/
├── main.py                    # Uvicorn entry point
├── pyproject.toml             # Python dependencies and tool settings
├── Dockerfile.inference       # Inference image definition
├── shared/                    # Shared FFT/DWT transforms
├── inference/                 # ONNX Runtime predictor and FastAPI service
│   ├── api/                   # Application, routes, schemas, utilities
│   ├── models/                # three_class.onnx
│   ├── predictor.py           # TTA and post-processing
│   └── batch_detect.py        # Recursive batch inference
├── trainer/                   # CNN+FFT+DWT training and export
│   ├── release_train.py       # Canonical release-training wrapper
│   ├── model.py               # EfficientNet/frequency fusion models
│   ├── hard_examples.txt      # Confirmed local regression manifest
│   ├── dataset.py             # Legacy RGB/FFT/DWT datasets
│   ├── train.py               # Historical trainer (train_legacy)
│   └── ablation.py            # Optimization ablations
├── experiment/                # Multi-model experiment runner
│   └── cnn_fft_dwt_ablation/  # Ablation and release harness
│       ├── harness.py         # Training loop + ExpConfig + versioned cache
│       ├── finalist.py        # Historical multi-seed finalist training
│       ├── finalize_export.py # Historical ONNX parity helper
│       ├── deploy_eval.py     # Production ONNX end-to-end evaluation
│       ├── show.py            # Leaderboard viewer
│       ├── leaderboard.jsonl  # Experiment records
│       └── REPORT.md          # Full Chinese report
├── tests/                     # Unit and integration tests
└── data/                      # Local datasets and generated outputs (ignored)
```

## Experimental Results

### Current deployment (2026-07-22, `candidate_20260722_unf3_focus2_6x12`)

The current 6+12 epoch release run selected `finetune-11`, passed both confirmed hard examples, and exported a 22.2 MB ONNX model (`96aedc9f…`). Its PyTorch argmax result is accuracy 0.9429 / screen_photo F1 0.9076 / macro F1 0.9361 / metric 0.9239.

Relevant deployable artifacts were evaluated through the same ONNX Runtime path, on the same current 368-image test list, with identical TTA, OOD, and threshold settings:

| Production ONNX candidate | Accuracy | screen_photo F1 | Macro F1 | Metric | Required screenshot gate | Release eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Previous production (`4b419976…`) | **0.9402** | **0.8870** | **0.9376** | **0.9131** | 0/2 | No |
| Previous README release (`cbba84bd…`) | 0.9158 | 0.8598 | 0.9145 | 0.8875 | 2/2 | Yes |
| Current release (`96aedc9f…`) | 0.9158 | **0.9043** | 0.9262 | 0.9121 | **2/2** | **Yes** |

The current release improves over the README's previous deployable release by +4.45 pp screen-photo F1, +1.16 pp macro F1, and +2.46 pp metric, with accuracy tied. The older `4b419976…` artifact is still slightly higher on raw aggregate metric, but it fails both mandatory screenshot regressions and is not release-eligible. The production view treats `unknown` as an incorrect classification, matching the real API.

The older README-only baseline (accuracy 0.8946 / screen_photo F1 0.7630 / macro F1 0.8668) is retained as historical context, not as the controlled artifact comparison above. Its training split and `hard_negative` handling differ from the current release, and the previous artifact also has a different training provenance from the current fixed split.

**Confirmed screenshot regressions** (production ONNX with TTA):

| Image | Before | After | screenshot probability | screen_photo probability |
| --- | --- | --- | ---: | ---: |
| `4a6e…ae8f9.png` | `screen_photo` | **`screenshot`** | 0.5552 | 0.3260 |
| `5cdc3…12a62.png` | `natural` | **`screenshot`** | 0.4679 | 0.3175 |

**Confidence distribution**: 49 accept (high ≥0.92), 139 review (medium 0.75–0.92), 168 low-confidence ignore, and 12 OOD ignore. The latest single-run CPU TTA diagnostic was mean 351 ms / p50 334 ms / p95 454 ms. Serial latency runs are sensitive to warm-up and machine load, so latency was not used to select between same-architecture artifacts.

### Archived 3-seed finalist training

To verify stability, the winning config was trained with 3 seeds (42 / 2024 / 7):

| Seed | test_acc | test_sp_f1 | test_macro_f1 | metric |
| --- | ---: | ---: | ---: | ---: |
| **42 (previous finalist)** | **0.9322** | **0.8548** | **0.9174** | **0.8952** |
| 2024 | 0.9133 | 0.8276 | 0.8963 | 0.8670 |
| 7 | 0.9079 | 0.7788 | 0.8817 | 0.8381 |
| **AVG** | **0.9178** | **0.8204** | **0.8985** | **0.8668** |

These archived runs used the earlier 369-image split and did not include the two-example release gate. Their mean (acc 0.9178, sp_f1 0.8204, macro_f1 0.8985) remains historical context; current deployment metrics are the 2026-07-22 results above.

### Ablation matrix (15 configs, screening stage H=6+F=12)

Full records live in `experiment/cnn_fft_dwt_ablation/leaderboard.jsonl`. Summary sorted by test_metric:

| Rank | id | test_acc | test_spF1 | macro_F1 | metric | Change vs. baseline |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 🥇 | **a_unf1** | **0.9431** | 0.8943 | **0.9342** | **0.9169** | unfreeze=1 (winner) |
| 🥈 | a_unf3 | 0.9295 | **0.9000** | 0.9239 | 0.9136 | unfreeze=3 |
| 🥉 | a_b1 | 0.9295 | 0.8926 | 0.9219 | 0.9095 | B1 backbone + unfreeze=1 |
| 4 | ref | 0.9322 | 0.8814 | 0.9227 | 0.9049 | clean default (unfreeze=2) |
| 5 | a_alpha20 | 0.9295 | 0.8780 | 0.9201 | 0.9019 | α=2.0 |
| 6 | a_noema | 0.9187 | 0.8571 | 0.9070 | 0.8856 | no EMA |
| 7 | a_gamma3 | 0.9214 | 0.8522 | 0.9079 | 0.8841 | γ=3.0 |
| 8 | a_ls0 | 0.9295 | 0.8448 | 0.9128 | 0.8838 | no label smoothing |
| 9 | a_ls10 | 0.9214 | 0.8308 | 0.9050 | 0.8728 | label smoothing=0.10 |
| 10 | a_fftonly | 0.9187 | 0.8308 | 0.9024 | 0.8715 | no DWT |
| 11 | old_focal | 0.9051 | 0.8293 | 0.8906 | 0.8643 | old strategy (γ3, heavy aug, all-unfreeze, no EMA) |
| 12 | a_cb99 | 0.9133 | 0.8214 | 0.8945 | 0.8636 | class-balanced sampling β=0.99 |
| 13 | a_heavyaug | 0.8943 | 0.8254 | 0.8810 | 0.8572 | strong augmentation |
| 14 | a_attn | 0.9106 | 0.7767 | 0.8827 | 0.8381 | +CBAM |
| 15 | a_alpha25 | 0.8943 | 0.7521 | 0.8664 | 0.8176 | α=2.5 |

**Key ablation findings**:
- ✅ **Wins**: Focal (γ=2) + LS=0.05, EMA, unfreeze 1 stage, B0 + FFT + DWT, moderate augmentation
- ❌ **Hurts**: CBAM attention (sp_f1 −10pp), class-balanced sampling, strong augmentation (−5pp), γ=3, α=2.5 (−13pp), full unfreeze
- 🏆 **Old vs new strategy**: old (γ3 + heavy + all-unfreeze + no-EMA) → new (γ2 + LS + EMA + unf1) gains **+5.2pp sp_f1**

### Research comparison

These five archived runs are single trials from the experiment harness; their 1500-image training set and validation split differ from the deployed run above. The table uses macro F1, and precision/recall/F1 in the final three columns refer specifically to `screen_photo`.

| Model | Accuracy | Macro F1 | SP precision | SP recall | SP F1 | Inference time | Model size |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| EfficientNet-B0 (CNN) | 94.39% | 92.57% | 86.49% | 88.89% | 87.67% | Not recorded | Not recorded |
| CNN + FFT | 93.58% | 89.46% | 81.82% | 75.00% | 78.26% | Not recorded | Not recorded |
| DeiT-Small | 93.85% | 91.61% | 82.05% | 88.89% | 85.33% | Not recorded | Not recorded |
| FFT + DeiT | 93.32% | 91.71% | 88.57% | 86.11% | 87.32% | Not recorded | Not recorded |
| DWT + FFT + DeiT | 93.85% | 91.95% | 82.50% | 91.67% | 86.84% | Not recorded | Not recorded |

The repository does contain a 22.2 MB `three_class.onnx` deployable artifact. It must not be compared directly with the research table because its training procedure and validation split differ.

## Performance Comparison

The current production path reaches **acc 0.9158 / sp_f1 0.9043 / macro_f1 0.9262** on its leakage-free held-out split. Relative to the historical README-only baseline, that is +2.1 pp accuracy, +14.1 pp screen-photo F1, and +5.9 pp macro F1, but the split difference makes that comparison contextual rather than controlled.

On the same current 368-image list and deployment pipeline, the current release beats the README's previous deployable release (metric 0.9121 vs. 0.8875) while preserving the 2/2 screenshot gate. The older production artifact remains slightly higher on aggregate metric (0.9131) but fails the mandatory screenshot gate 0/2, so it is not a valid release candidate.

For research models, the CNN baseline had the highest overall accuracy and screen-photo F1 (87.67%), while DWT+FFT+DeiT had the highest recorded screen-photo recall (91.67%). One trial per model and aggressive early stopping mean those observations are directional, not production benchmarks.

For a deployment comparison, benchmark exported models on the target hardware with the same warm-up, image set, batch size, and ONNX Runtime provider. Record p50/p95 latency, throughput, peak memory, artifact size, class-wise metrics, and mandatory regression gates. `experiment/cnn_fft_dwt_ablation/deploy_eval.py` accepts `--model`, `--output`, and `--label`; the latest current-model CPU diagnostic was mean 351 ms / p50 334 ms / p95 454 ms, including TTA.

## Model Evolution

```mermaid
timeline
    title Model evolution in this repository
    Rule-based binary classifier : Historical approach
    Two-stage CNN : Historical approach
    Single-stage CNN : Three-class formulation
    CNN plus FFT : Spatial and frequency fusion experiments
    ViT experiments : Transformer baseline
    DeiT experiments : Single and multi-stream research variants
    DWT plus FFT plus DeiT : Highest archived screen-photo recall
    Image forensics optimization : Focal Loss, hard negatives, TTA, threshold analysis
```

## Deployment

Run the API directly with `uv run python main.py`, or build the supplied container:

```bash
docker build -f Dockerfile.inference -t opencv-screen-detector:local .
docker run --rm -p 8325:8325 \
  -v "${PWD}/inference/models:/app/inference/models:ro" \
  -v "${PWD}/data:/app/data" \
  opencv-screen-detector:local
```

The Dockerfile copies the tracked LFS model under `inference/models/`, but not the ignored training data. The image exposes port 8325 and persists uploads/index data under `/app/data`; ensure Git LFS materializes the ONNX object before building locally.

## Configuration

Runtime settings are defined in `inference/config.py`; release-training defaults are in `trainer/release_train.py` and `experiment/cnn_fft_dwt_ablation/harness.py`. `trainer/config.py` serves legacy and PAH-ViT research paths.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `image_size` | 224 | Spatial input width and height |
| `ood_threshold` | 0.45 | Maximum probability below this returns `unknown` |
| `confidence_high` | 0.92 | `accept` threshold |
| `confidence_medium` | 0.75 | `review` threshold |
| `screen_photo_threshold` | 0.60 | screen_photo probability at or above this forces that class |
| `BATCH_SIZE` | 16 | Standard training batch size |
| `EPOCHS_HEAD` / `EPOCHS_FINETUNE` | 10 / 20 | Two training stages |
| `FOCAL_LOSS_GAMMA` | 2.0 | Focal Loss focusing parameter (adjusted from 3.0 after ablation) |
| `LABEL_SMOOTHING` | 0.05 | Label smoothing (ablation showed 0.05 > 0 / 0.10) |
| `EMA_DECAY` | 0.999 | Weight EMA decay |
| `UNFREEZE_STAGES` | 1 | Number of MBConv stages unfrozen in stage B |
| `HARD_EXAMPLE_WEIGHT` | 4.0 | Sampler multiplier for paths in `trainer/hard_examples.txt` |

Keep preprocessing, model inputs, and thresholds aligned when substituting a model. The current service expects the three-input ONNX graph described in [Inference](#inference).

## Future Work

- ~~Run repeated seeds and cross-validation for the research models.~~ (Done in `experiment/cnn_fft_dwt_ablation/finalist.py` with 3 seeds.)
- ~~Create reproducible release artifacts with a fixed split and hard-example gate.~~ (Done by `trainer/release_train.py`.)
- Calibrate decision thresholds on a dedicated validation/calibration set; keep the test-set threshold scan diagnostic-only.
- Further grow and audit screen-photo boundary cases, document dataset provenance.
- Add CI checks for training/export compatibility and API smoke tests.
- Explore finer-grained FFT recalibration (per-bin mean/std), Mixup/CutMix, and multi-task joint training.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

Built with [PyTorch](https://pytorch.org/), [timm](https://github.com/huggingface/pytorch-image-models), [OpenCV](https://opencv.org/), [FastAPI](https://fastapi.tiangolo.com/), [ONNX Runtime](https://onnxruntime.ai/), and [uv](https://docs.astral.sh/uv/).
