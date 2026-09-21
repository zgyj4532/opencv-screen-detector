# 2026-09-22：9 月 15–21 日拍屏增量训练

7 份指定的 daily-package ZIP 中共有 25 张 `screen_photo`，全部为新增内容。
按 SHA-256 去重导入 `data/input/screen_photo/`，其他类别不落盘，原始 ZIP 保留。
重复执行导入核验：新增写入 0、跳过同内容图片 25、跳过非目标文件 233。
本轮比较三组训练参数、两个随机种子；结果范围限于这组预先固定的实验。

## 数据和协议

| ZIP 日期 | 新增拍屏图片 |
| --- | ---: |
| 2026-09-15 | 9 |
| 2026-09-16 | 5 |
| 2026-09-17 | 1 |
| 2026-09-18 | 4 |
| 2026-09-19 | 2 |
| 2026-09-20 | 2 |
| 2026-09-21 | 2 |

原始路径 3,105 个，唯一内容 2,973 个，重复路径 132 个。
train / val / test 为 **2,247 / 361 / 365**；新增 25 张全部为 train-only。
原有 9 月 14 日实验的 5 张训练图片仍在本轮数据中。
val/test 内容清单与运行前逐项比较一致，三个 split 之间的精确内容重合为零。

- 数据指纹：`ec60352964610fb891013be4f39caa9cc846fd03439e51f4c43f37233d8e6c32`
- 评测指纹：`da74a983a7af3b5c1f73d1c80ccd5a7ed84a290e6f72dfb615a5eb6f73390eff`
- 来源清单及目标哈希：[data/ingest.json](data/ingest.json)
- 独立审计快照：[data/audit.json](data/audit.json)
- 冻结切分：[data/split.json](data/split.json)
- 训练前冻结协议：[data/protocol.json](data/protocol.json)

保留目录标签作为本次训练来源标签；通过图片解码和内容冲突检查，不代表新增标签经过独立人工复核。
已解决 18 组旧标签冲突，未改动既有人工标签决定。19 个跨 split pHash 候选对仍待审核；
采集组元数据、Frozen challenge 和真 OOD 数据仍不完整。

## 对照设计

模型结构全部保持 EfficientNet-B0 + FFT + Haar DWT。
每次从当前生产 checkpoint 热启动，微调 8 epoch，seed 为 42 和 2024。

| 组别 | 主学习率 | 蒸馏权重 | 对照问题 |
| --- | ---: | ---: | --- |
| `baseline` | 3e-4 | 0 | 同数据、同预算的常规微调 |
| `low_lr` | 1e-4 | 0 | 降低学习率能否改善验证质量 |
| `lwf` | 3e-4 | 1 | 旧模型 KL 蒸馏能否减少遗忘 |

其余参数相同：batch=8、AMP、最后 3 个 MBConv stage 解冻、Focal gamma=2、
alpha=[1,1,1.5]、label smoothing=0.05、EMA=0.999、weight decay=1e-4、
Canary 2×、新增图片 8× 采样、Remix 关闭、蒸馏温度 2。
骨干和频域分支实际学习率为主学习率的 0.1 倍，分类头使用主学习率。
训练和验证 batch 均限制为 8，显存分配上限为 70%，主机只缓存 64 张图片，六次训练串行执行。

每次 checkpoint 只按 validation 的 `0.5 × screen_photo F1 + 0.3 × accuracy + 0.2 × macro F1` 选择。
随后导出 ONNX，核验数值一致性，再使用真实 `ScreenDetectorPredictor`、TTA 和固定阈值评估 validation。
按两个 seed 的 Predictor validation 综合指标均值比较组别，要求两个 seed 均通过 Canary。
主种子预先固定为 42，2024 用于复验；不从两个 seed 中挑较高的 test 分数。

所有 validation 齐备后写入不可覆盖的 `reports/selection.json`。
仅允许胜出组、同数据 baseline 的两个 seed，以及当前生产模型各评估一次历史 test。
读取 test 后不再调参数、改阈值或重新选型。
`unknown` 计为闭集错误；它不是真 OOD 命中。新图训练后表现不作为泛化指标。

## 结果

六次训练全部完成，共 48 个微调 epoch。按预先确定的双种子 Predictor validation 均值，
选中 `low_lr`：主学习率 **1e-4**、蒸馏关闭，其余设置见上表。
主交付候选固定为 seed 42、finetune-1；seed 2024 选中 finetune-3，用于复验。

| 参数组 | 训练 val 指标 42 / 2024 | Predictor val 指标 42 / 2024 | Predictor 均值 | Canary |
| --- | --- | --- | ---: | --- |
| 常规 `baseline` | 0.9591 / 0.9628 | 0.9347 / 0.9424 | 0.9385 | 两次均 2/2 |
| 低学习率 `low_lr` | 0.9591 / 0.9640 | 0.9344 / 0.9435 | **0.9390** | 两次均 2/2 |
| 蒸馏 `lwf` | 0.9528 / 0.9638 | 0.9226 / 0.9322 | 0.9274 | 两次均 2/2 |

当前生产模型在同一 validation 上的 Predictor 指标为 0.9241，Canary 2/2。
低学习率组比常规组的均值仅高 **0.00045**，且 seed 42 略低、seed 2024 略高。
这是本轮有限参数搜索的数值胜者，不足以证明低学习率带来稳定收益。
六份 ONNX 均通过随机输入及三类真实输入的数值一致性检查。
单次训练约 15.6–18.2 分钟，峰值 CUDA allocated 为 1,962–2,182 MiB，没有发生 OOM。

选型于北京时间 **2026-09-22 03:07:22** 冻结，见 [reports/selection.json](reports/selection.json)。
随后完成固定 365 张历史 test 的真实 Predictor 比较：

| 模型 | Accuracy | 拍屏 F1 | Macro F1 | 综合指标 | unknown | Canary |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 当前生产 | 0.9233 | 0.9123 | 0.9296 | 0.9190 | 10 | 2/2 |
| 常规，42 | 0.9288 | 0.9217 | 0.9380 | 0.9271 | 12 | 2/2 |
| 常规，2024 | 0.9260 | 0.9043 | 0.9322 | 0.9164 | 11 | 2/2 |
| 低学习率，42（主候选） | 0.9288 | 0.9217 | 0.9378 | 0.9271 | 12 | 2/2 |
| 低学习率，2024（复验） | 0.9260 | 0.8908 | 0.9289 | 0.9090 | 10 | 2/2 |

**结论：完成本轮选型，但没有确认稳定的识别提升；保留生产模型。**
主候选相对生产改对 3 张、改错 1 张，综合指标增加 0.0080；
图片级配对 bootstrap 的 95% 区间为 [-0.0123, 0.0309]。
复验种子的综合指标比生产低 0.0101，拍屏 F1 下降 2.15 个百分点，
指标差区间为 [-0.0417, 0.0197]。两个区间均包含零。
相对同种子常规组，低学习率的指标差分别为 -0.000024 / -0.007463，
也不能把本轮结果写成稳定的参数优化收益。

配对统计使用固定 seed=42、10,000 次同图重采样，仅描述这些历史图片，
未覆盖采集组相关性或训练随机性的完整不确定性。
LwF 未入选，其 test 保持封存；没有依据 test 改选常规组、重调阈值或继续调参。
完整结果见 [reports/results.json](reports/results.json)，逐图记录见 `reports/*.test.json`。
候选模型单独保存用于复核；独立评测门槛仍不完整，生产晋升为 `NOT_READY`。

本轮主候选：[daily20260922_low_lr_s42_b8.onnx](models/daily20260922_low_lr_s42_b8.onnx)，
22,184,058 字节，经 Git LFS 提交。SHA-256：
`95d7bb7b55d309341ac26c1dd85cfea46bec4b61e4646a83d3d6ba4e0395d1db`。
生产 `inference/models/three_class.onnx` 的 SHA-256 保持
`c53b00d5511f0deb358df387f5e21202fc348f2b3f6fbbb8f39e676a03a7b390`。
候选复核可显式指定路径：

```python
from pathlib import Path
from inference.predictor import ScreenDetectorPredictor

predictor = ScreenDetectorPredictor(
    model_path=Path("experiment/daily_20260922/models/daily20260922_low_lr_s42_b8.onnx")
)
result = predictor.predict(Path("path/to/image.jpg"))
```

## 目录

```text
daily_20260922/
├── README.md
├── src/          # 导入、训练、导出、选型与协议测试
├── data/         # 来源、审计、切分与预先固定的协议；不含原图
├── reports/      # 训练历史、逐图预测、选型及比较
├── models/       # 本地候选 ONNX
├── logs/         # 本地运行日志
└── archive/      # 运行前已有变更的备份
```

共享 checkpoint 仍位于 `experiment/cnn_fft_dwt_ablation/exp/daily20260922_<组别>_s<seed>_b8/`。
本实验不依赖未提交的 `daily_20260914` 源码，也不改写其结果。

本轮也修正共享 harness 的蒸馏续训：恢复学生状态之前，始终从原 checkpoint 建立固定教师。
CPU 小模型的连续两 epoch 与暂停一 epoch 后恢复，最终学生张量和训练历史逐项一致；
修复前同一测试失败。该修改不影响未中断训练的计算路径。

## 验证

- 7 份 ZIP 全部通过 CRC 检查；25 张目标图片解码及源/目标 SHA-256 一致。
- 重复导入新增 0 张；冻结 val/test 清单未改变，原 ZIP、生产 checkpoint 和生产 ONNX 哈希未变。
- 六次训练、六份候选 validation、五份预先允许的 test 记录均完整；6 份 ONNX parity 通过。
- `uv run pytest tests/ experiment/daily_20260922/src/tests/ -q`：**57 passed**。
- `uv run ruff check .`、`uv run ruff format --check .`（89 个 Python 文件）和 `git diff --check` 通过。
- [reports/verification.json](reports/verification.json)记录实际输入、模型、源码哈希和依赖版本。

仓库当前仅有 tag / 手动触发的 Docker 发布工作流，本次 PR 不触发该发布流程。
上述为本地测试和候选评估证据，不表示已部署或远程 CI 通过。

## 执行顺序

在仓库根目录运行，需原始数据、生产 checkpoint、冻结切分和 CUDA 环境。
下面是本轮实际执行顺序；已提交的证据禁止覆盖，同一 ID 的已完成结果应直接读取。
重新开展实验请使用新的目录和候选前缀，重新冻结协议。

```powershell
uv sync --group train
uv run python -m experiment.daily_20260922.src.prepare
uv run python -m experiment.daily_20260922.src.queue
# 中断后使用相同环境和输入恢复，已完成报告会保留：
uv run python -m experiment.daily_20260922.src.queue --resume
# 六份真实 Predictor validation 齐备后：
uv run python -m experiment.daily_20260922.src.select
# 本轮已冻结 low_lr；按 test_allowed_ids 各评估一次：
uv run python -m experiment.daily_20260922.src.evaluate production --role test
uv run python -m experiment.daily_20260922.src.evaluate baseline --seed 42 --role test
uv run python -m experiment.daily_20260922.src.evaluate baseline --seed 2024 --role test
uv run python -m experiment.daily_20260922.src.evaluate low_lr --seed 42 --role test
uv run python -m experiment.daily_20260922.src.evaluate low_lr --seed 2024 --role test
uv run python -m experiment.daily_20260922.src.compare
uv run python -m experiment.daily_20260922.src.verify
uv run pytest tests/ experiment/daily_20260922/src/tests/ -q
```

运行前本地 `trainer/data_audit.json` 和共享 `split.json` 已有未提交变化。
原文件快照保存在本地 `archive/`；本轮的完整审计证据以本目录 `data/` 为准。
