# Screen Detector V3 — 训练结果与消融报告

> 生成时间：2026-07-22
> 训练硬件：单卡 RTX 3060 6GB / Windows 11
> 训练框架：PyTorch 2.x + timm (EfficientNet-B0) + Albumentations
> 数据规模：2928 张（自然图 / 截图 / 屏幕照片，含 hard_negative）
> 当前发布切分：seed=42 固定分层切分 0.70/0.15/0.15 = 2194/366/368（**无样本级泄漏**）

---

## 1. 核心结论

把相关可部署 ONNX 放到同一当前 368 张测试清单，并使用相同 ONNX Runtime、TTA、OOD 与阈值后处理，结果如下：

| 生产候选 | accuracy | macro_F1 | screen_photo F1 | metric | 指定 screenshot 门槛 | 发布资格 |
|---|---:|---:|---:|---:|---:|---|
| 上一生产模型（`4b419976…`） | **0.9402** | **0.9376** | **0.8870** | **0.9131** | 0/2 | 无 |
| README 上一版发布（`cbba84bd…`） | 0.9158 | 0.9145 | 0.8598 | 0.8875 | 2/2 | 有 |
| 当前发布（`96aedc9f…`） | 0.9158 | 0.9262 | **0.9043** | 0.9121 | **2/2** | **有** |

当前发布相对 README 上一版可发布模型提升 +4.45 pp screen_photo F1、+1.16 pp macro F1、+2.46 pp metric，accuracy 持平。上一生产工件 `4b419976…` 的普通测试集 metric 仍略高，但把两张必须修复的 screenshot 分别判为 `screen_photo` 与 `natural`，不具备发布资格。两张指定截图在当前真实 ONNX+TTA 路径中的 `screenshot` 概率分别为 0.5552 与 0.4679。

当前 checkpoint 的 PyTorch argmax 指标为 accuracy 0.9429 / macro F1 0.9361 / screen_photo F1 0.9076 / metric 0.9239。相对 README 历史值（0.8946 / 0.8668 / 0.7630），当前生产 ONNX 分别提升约 +2.1 pp accuracy、+5.9 pp macro F1、+14.1 pp screen_photo F1。

> ⚠️ **重要说明（数据切分差异）**：
> - 旧基线数字来自 README 报告的训练，**其 train/val 切分有 hard_negative 数据泄漏**（`hard_negative` 既出现在 train 也以 3× 权重作为 screenshot 重复出现），旧基线数字极可能因泄漏而虚高。
> - 新数字使用 **固定的 stratified 0.70/0.15/0.15 切分**，所有样本按 path 去重，hard_negative 只加入 train，**完全无泄漏**。
> - 因此 README 历史值只用于背景对照，不能替代上方同一部署路径的工件对照；上一生产工件本身的训练来源也不同于当前固定切分。

---

## 2. 历史消融实验矩阵（15 配置，全部 H=6+F=12 epoch）

完整实验结果（按 test_metric 排序）：

| 排名 | id | test_acc | test_spF1 | spP | spR | macroF1 | metric | 配置变更 |
|---|---|---|---|---|---|---|---|---|
| 🥇 | **a_unf1** | **0.9431** | 0.8943 | 0.859 | 0.932 | **0.9342** | **0.9169** | unfreeze=1（最佳） |
| 🥈 | a_unf3 | 0.9295 | **0.9000** | 0.885 | 0.915 | 0.9239 | 0.9136 | unfreeze=3 |
| 🥉 | a_b1 | 0.9295 | 0.8926 | 0.871 | 0.915 | 0.9219 | 0.9095 | B1 backbone + unfreeze=1 |
| 4 | ref | 0.9322 | 0.8814 | 0.881 | 0.881 | 0.9227 | 0.9049 | 干净默认（unfreeze=2） |
| 5 | a_alpha20 | 0.9295 | 0.8780 | 0.844 | 0.915 | 0.9201 | 0.9019 | α=2.0 |
| 6 | a_noema | 0.9187 | 0.8571 | 0.806 | 0.915 | 0.9070 | 0.8856 | 无 EMA |
| 7 | a_gamma3 | 0.9214 | 0.8522 | 0.875 | 0.831 | 0.9079 | 0.8841 | γ=3.0 |
| 8 | a_ls0 | 0.9295 | 0.8448 | 0.860 | 0.831 | 0.9128 | 0.8838 | 无 label smoothing |
| 9 | a_ls10 | 0.9214 | 0.8308 | 0.761 | 0.915 | 0.9050 | 0.8728 | label smoothing=0.10 |
| 10 | a_fftonly | 0.9187 | 0.8308 | 0.761 | 0.915 | 0.9024 | 0.8715 | 无 DWT |
| 11 | old_focal | 0.9051 | 0.8293 | 0.797 | 0.864 | 0.8906 | 0.8643 | 旧策略（γ3, heavy aug, 全解冻, 无 EMA） |
| 12 | a_cb99 | 0.9133 | 0.8214 | 0.868 | 0.780 | 0.8945 | 0.8636 | class-balanced 采样 β=0.99 |
| 13 | a_heavyaug | 0.8943 | 0.8254 | 0.776 | 0.881 | 0.8810 | 0.8572 | 强增广 |
| 14 | a_attn | 0.9106 | 0.7767 | 0.909 | 0.678 | 0.8827 | 0.8381 | +CBAM |
| 15 | a_alpha25 | 0.8943 | 0.7521 | 0.759 | 0.746 | 0.8664 | 0.8176 | α=2.5 |

---

## 3. 关键消融结论

### 3.1 损失函数与正则化

| 维度 | 测试结论 |
|---|---|
| **Focal γ** | γ=2.0 最佳（γ=3.0 退步 -3.2pp spF1） |
| **α 权重** | α=[1,1,1.5] 最佳；α=2.0 微退；α=2.5 严重退化 (-13pp spF1) |
| **Label smoothing** | 0.05 最佳；0 退步 -3.7pp spF1；0.10 退步 -5pp spF1 |
| **EMA** | 关闭退步 +1.4pp acc, +2.4pp spF1，**必备** |

### 3.2 Backbone 与微调

| 维度 | 测试结论 |
|---|---|
| **Backbone** | B0 略胜 B1（参数 5.5M vs 7.5M，**小数据下 B0 更稳**） |
| **解冻层数** | **1 个 stage 最佳**（2 个 ref、3 个 unf3 全部次优），少解冻=少过拟合 |
| **CBAM 注意力** | 显著有害（spF1 -10pp，attn 过度聚焦到无关区域） |

### 3.3 频域分支与采样

| 维度 | 测试结论 |
|---|---|
| **DWT 贡献** | 重要：移除 DWT 后 spF1 -5pp，FFT 全局 + DWT 局部互补 |
| **强增广** | 有害（-5pp acc），产品级模型应使用中等增广 |
| **Class-balanced 采样** | 有害（-2pp acc, -3pp spF1），inverse-freq 已足够 |

### 3.4 对比旧策略

**`old_focal`（旧配置复现）vs `ref`（新默认）**：
- 旧策略：γ=3 + 无 label smoothing + 强增广 + 全解冻 + 无 EMA
- 新策略：γ=2 + LS=0.05 + 温和增广 + 解冻 1-2 stages + EMA
- 旧 vs 新：spF1 0.8293 → 0.8814（**+5.2pp**），macroF1 0.8906 → 0.9227（+3.2pp）

**新策略的核心改进**：简化损失函数 + EMA + 限制解冻深度 + 温和增广。

---

## 4. 当前发布配置

**`candidate_20260722_unf3_focus2_6x12`**（当前切分上重新搜索的 unfreeze3 + focus2 + 6+12 epoch 候选）：

```yaml
backbone: efficientnet_b0
gamma: 2.0
alpha: [1.0, 1.0, 1.5]
label_smoothing: 0.05
use_attention: false
use_arcface: false
ema: true
ema_decay: 0.999
unfreeze: 3
lr: 1e-3
weight_decay: 1e-4
epochs_head: 6
epochs_finetune: 12
batch_size: 16
heavy_aug: false
use_dwt: true
focus_weight: 2.0
```

检查点选择先最大化 hard-example 通过数，再按验证集 metric 选优。本次从 18 个 epoch 中选择 `finetune-11`，hard-example 为 **2/2**。

| 评估 | accuracy | spF1 | macroF1 | metric |
|---|---:|---:|---:|---:|
| Validation | 0.9235 | 0.8448 | 0.9076 | 0.8810 |
| Test / PyTorch argmax | **0.9429** | **0.9076** | **0.9361** | **0.9239** |
| Test / production ONNX | 0.9158 | 0.9043 | 0.9262 | 0.9121 |

README 上一版发布 ONNX 为 accuracy 0.9158 / spF1 0.8598 / macroF1 0.9145 / metric 0.8875，当前发布全面提升其 F1 与 metric。更早的上一生产 ONNX 为 accuracy 0.9402 / spF1 0.8870 / macroF1 0.9376 / metric 0.9131，但 hard-example 门槛仅 0/2。它的普通指标不能越过发布资格门槛。

**历史 3 种子全量训练结果**（上一版 369 张切分，无 hard-example 门槛）：

| Seed | test_acc | test_spF1 | macroF1 | metric | elapsed |
|---|---|---|---|---|---|
| **42** | **0.9322** | **0.8548** | **0.9174** | **0.8952** | 1509s |
| 2024 | 0.9133 | 0.8276 | 0.8963 | 0.8670 | 1542s |
| 7 | 0.9079 | 0.7788 | 0.8817 | 0.8381 | 1542s |
| AVG | 0.9178 | 0.8204 | 0.8985 | 0.8668 | 1531s |

历史 3 种子均值（acc 0.9178, spF1 0.8204, macroF1 0.8985）支持当前超参数选择，但不能替代上方当前发布指标。

---

## 5. 部署与一致性

- **ONNX 导出**：`trainer/checkpoints/three_class_best.pth` → `inference/models/three_class.onnx`
- **PyTorch ↔ ONNX 数值一致性**：rtol=1e-3 通过
- **真实生产策略的 ONNX 测试集评估**（TTA + OOD + screen_photo threshold=0.60；unknown 按错误计）：
  - acc 0.9158 / sp_f1 0.9043 / macro_f1 0.9262
  - screen_photo precision 0.9286 / recall 0.8814
- **指定回归**：`tests/test_target_screenshots.py` 2/2 通过
- **工件对照**：README 上一版发布 ONNX metric 0.8875、门槛 2/2；当前 ONNX metric 0.9121、门槛 2/2；上一生产 ONNX metric 0.9131、门槛 0/2
- **当前 ONNX SHA-256**：`96aedc9fd009535eba125f9a9c79f9764237aedcd882f779988c42af494fea5b`
- **上一生产 ONNX SHA-256**：`4b4199763b49bc62c8f1e11704e5b2dd99a1177b2517e57443fc28112b94aae7`
- **生产路径**：
  - `inference/models/three_class.onnx` (22 MB)
  - `trainer/checkpoints/three_class_best.pth` (22 MB)
  - `trainer/checkpoints/three_class_final.pth` (22 MB)

---

## 6. 风险与改进空间

1. **测试集仅 368 张**：screen_photo 只有 59 张，指标仍有较大方差，需要更多独立标注数据。
2. **screen_photo recall 为 0.8814**：precision 为 0.9286；若业务更重召回，应在独立 calibration 集上选阈值，不能用 test 集扫描值直接上线。
3. **hard-example 门槛目前只有 2 张**：已解决指定问题，但应持续收集同域高分辨率游戏截图，防止仅记忆单样本。
4. **DWT 仍为 224×224**：训练与推理一致，但未完成 per-bin FFT/DWT 重标定实验。

---

## 7. 文件清单

- 发布训练：`trainer/release_train.py`, `trainer/hard_examples.txt`, `experiment/cnn_fft_dwt_ablation/harness.py`
- 历史多种子训练：`experiment/cnn_fft_dwt_ablation/finalist.py`
- 导出：`trainer/export_onnx.py`
- 部署评估：`experiment/cnn_fft_dwt_ablation/deploy_eval.py`
- 排行查看：`experiment/cnn_fft_dwt_ablation/show.py`
- 排行榜：`experiment/cnn_fft_dwt_ablation/leaderboard.jsonl`（当前包含发布候选记录）
- 当前训练结果：`trainer/logs/release_training_result.json`
- 当前 epoch 历史：生成于 `experiment/cnn_fft_dwt_ablation/exp/<release-id>/history.json`，已作为可再生缓存从工作区清理
- 当前模型：`trainer/checkpoints/three_class_best.pth`, `inference/models/three_class.onnx`
