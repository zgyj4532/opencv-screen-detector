# Screen Detector V3 - 2026-07-28 论文导向消融报告

## 1. 本次目标

- 增加 API/uvicorn 维护日志。
- 基于更新后的 `data/input` 重新构建固定 split。
- 参考近年视觉分类论文，对当前 `RGB backbone + FFT + DWT` 架构做小成本消融。
- 每个候选训练 5 个 epoch（2 head + 3 finetune），筛出 top3。
- 对 top3 做 5 次随机 200 图复评。
- 导出最佳候选 ONNX，并与现有生产 ONNX 做真实部署路径评估。

## 2. 论文依据与架构判断

参考方向：

- EfficientNetV2: https://arxiv.org/abs/2104.00298
- ConvNeXt V2: https://arxiv.org/abs/2301.00808
- MobileOne: https://arxiv.org/abs/2206.04040
- Fast Fourier Convolution: https://arxiv.org/abs/2012.11214
- Wavelet Pooling for CNNs: https://arxiv.org/abs/1805.08620

结论：

- 当前项目使用的 `FFT + DWT` 频域分支仍有合理性：屏幕照片、截图和自然图的差异常出现在周期纹理、重采样痕迹、边缘高频和局部小波细节上，不建议删除频域分支。
- 数据规模仍偏小（主数据 2485 + hard_negative 484），大模型或大 Transformer 不是优先方向。
- 更值得验证的是轻量/现代 CNN backbone 替换：EfficientNetV2-B0、ConvNeXt V2 Atto、MobileOne-S0。
- 现有生产模型的 18 epoch 训练深度明显强于本次 5 epoch 快筛候选；短训结果只能说明方向，不能直接替代部署评估。

## 3. 数据与切分

固定 split seed=42，更新后数据指纹：

```json
{
  "dataset_fingerprint": "bd41821ff5e1275769f07f293162e20b0aebd5f2491b417805b9ccd802f6503d",
  "main_count": 2485,
  "hard_negative_count": 484,
  "focus_count": 2,
  "train": 2222,
  "val": 371,
  "test": 376
}
```

## 4. 5 Epoch 消融结果

排序指标：`0.5 * screen_photo_f1 + 0.3 * accuracy + 0.2 * macro_f1`

| rank | id | backbone | test_acc | sp_f1 | sp_precision | sp_recall | macro_f1 | metric |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `pg260727_cnv2_atto` | ConvNeXt V2 Atto | 0.8856 | 0.8302 | 0.9565 | 0.7333 | 0.8740 | 0.8556 |
| 2 | `pg260727_mobileone` | MobileOne-S0 | 0.8670 | 0.7536 | 0.6667 | 0.8667 | 0.8464 | 0.8062 |
| 3 | `pg260727_effv2b0` | EfficientNetV2-B0 | 0.8324 | 0.7706 | 0.8571 | 0.7000 | 0.8173 | 0.7985 |
| 4 | `pg260727_b0_unf3` | EfficientNet-B0 | 0.8138 | 0.6667 | 0.8462 | 0.5500 | 0.7804 | 0.7336 |
| 5 | `pg260727_b0_unf1` | EfficientNet-B0 | 0.7580 | 0.5745 | 0.7941 | 0.4500 | 0.7112 | 0.6569 |

本次快筛下，ConvNeXt V2 Atto 最好，说明现代小型 ConvNet backbone 有继续长训价值。

## 5. Top3 随机 200 图复评

每个候选在全 `data/input` 样本池中随机抽取 200 张，重复 5 次。

| rank | id | acc mean | sp_f1 mean | macro_f1 mean | metric mean | metric std |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `pg260727_cnv2_atto` | 0.9100 | 0.8094 | 0.8866 | 0.8550 | 0.0183 |
| 2 | `pg260727_effv2b0` | 0.8690 | 0.7887 | 0.8391 | 0.8229 | 0.0520 |
| 3 | `pg260727_mobileone` | 0.8650 | 0.6938 | 0.8332 | 0.7730 | 0.0363 |

复评仍选择 `pg260727_cnv2_atto` 作为最佳新候选。

记录文件：

- `experiment/cnn_fft_dwt_ablation/leaderboard.jsonl`
- `experiment/cnn_fft_dwt_ablation/random200_top3_eval.json`

## 6. ONNX 导出与部署评估

最佳新候选已导出并通过 PyTorch/ONNX parity：

- checkpoint: `experiment/cnn_fft_dwt_ablation/exp/pg260727_cnv2_atto/best.pth`
- candidate ONNX: `experiment/cnn_fft_dwt_ablation/pg260727_cnv2_atto.onnx`
- candidate ONNX sha256: `0af4e3c41cf0db83d98f7bb384667df9a41661a4908f15674084627a39648025`

真实生产 predictor 路径（TTA + OOD + screen_photo threshold）对比：

| model | onnx path | acc | sp_f1 | sp_precision | sp_recall | macro_f1 | metric |
|---|---|---:|---:|---:|---:|---:|---:|
| existing production | `inference/models/three_class.onnx` | 0.9601 | 0.9412 | 0.9492 | 0.9333 | 0.9618 | 0.9510 |
| best new candidate | `experiment/cnn_fft_dwt_ablation/pg260727_cnv2_atto.onnx` | 0.7926 | 0.7423 | 0.9730 | 0.6000 | 0.8158 | 0.7721 |

部署决策：

- 不替换 `inference/models/three_class.onnx`，因为现有生产 ONNX 在更新数据的真实部署评估中仍明显优于本次 5 epoch 最佳新候选。
- 当前生产模型已经是本次比较中的最佳部署模型，保持部署状态。
- 新候选 ONNX 保留在实验目录，可用于后续长训或阈值校准研究。

记录文件：

- `experiment/cnn_fft_dwt_ablation/deploy_eval_existing_production.json`
- `experiment/cnn_fft_dwt_ablation/deploy_eval_candidate_pg260727_cnv2_atto.json`

## 7. 代码变更

- `main.py`: 启动 uvicorn 时输出 host、port、日志等级和 access log 状态。
- `inference/log.py`: uvicorn logger 接入 DEBUG，支持 `SCREEN_DETECTOR_LOG_LEVEL` 控制输出过滤。
- `inference/api/app.py`: 增加 startup/shutdown 和 HTTP 请求开始/结束/失败耗时日志。
- `inference/model_loader.py`: 增加 ONNX provider、模型大小、输入输出、加载失败、空闲卸载日志。
- `inference/predictor.py`: 增加单次预测类别、置信度、分级、动作和耗时日志。
- `trainer/model.py`: `load_model()` 可从 checkpoint 自动恢复 `use_dwt`、attention 配置，支持新候选导出。
- `experiment/cnn_fft_dwt_ablation/harness.py`: 通用化 timm backbone 解冻逻辑，支持 `blocks` 和 `stages`。
- `experiment/cnn_fft_dwt_ablation/paper_guided_ablation.py`: 本次 5 epoch 论文导向消融队列。
- `experiment/cnn_fft_dwt_ablation/eval_top3_random200.py`: top3 随机 200 图复评脚本。

## 8. 验证

已执行：

```bash
uv run ruff check main.py inference\api\app.py inference\log.py inference\model_loader.py inference\predictor.py trainer\model.py experiment\cnn_fft_dwt_ablation\harness.py experiment\cnn_fft_dwt_ablation\paper_guided_ablation.py experiment\cnn_fft_dwt_ablation\eval_top3_random200.py --output-format concise
uv run python -m py_compile main.py inference\api\app.py inference\log.py inference\model_loader.py inference\predictor.py trainer\model.py experiment\cnn_fft_dwt_ablation\harness.py experiment\cnn_fft_dwt_ablation\paper_guided_ablation.py experiment\cnn_fft_dwt_ablation\eval_top3_random200.py
uv run pytest tests/test_inference_thresholds.py tests/test_target_screenshots.py -v
```

结果：ruff 通过，py_compile 通过，3 个相关 pytest 通过。

## 9. 后续建议

- 若要真正挑战现有生产模型，应对 `pg260727_cnv2_atto` 做 18 epoch 或更长训练，并单独做阈值校准；5 epoch 快筛不足以作为生产替换依据。
- 可以把生产部署评估的 per-image INFO 日志降到 DEBUG 或给评估脚本增加 quiet 模式，避免批量评估输出过大。
