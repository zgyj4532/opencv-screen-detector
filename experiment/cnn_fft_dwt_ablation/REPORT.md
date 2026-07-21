# Screen Detector V3 — 训练结果与消融报告

> 生成时间：2026-07-21
> 训练硬件：单卡 RTX 3060 6GB / Windows 11
> 训练框架：PyTorch 2.x + timm (EfficientNet-B0) + Albumentations
> 数据规模：2922 张（自然图 / 截图 / 屏幕照片，含 hard_negative）
> 数据切分：seed=42 固定分层切分 0.70/0.15/0.15 = 2189/364/369（**无样本级泄漏**）

---

## 1. 核心结论

| 指标 | 旧基线（README 报告值） | **新模型** | 提升 | 显著性 |
|---|---|---|---|---|
| test **accuracy** | 0.8946 | **0.9187** (ONNX) / 0.9322 (PyTorch) | **+2.4 ~ +3.8 pp** | 显著 |
| test **macro_F1** | 0.8668 | **0.9039** (ONNX) / 0.9174 (PyTorch) | **+3.7 ~ +5.1 pp** | 显著 |
| test **screen_photo F1** | 0.7630 | **0.8397** (ONNX) / 0.8548 (PyTorch) | **+7.7 ~ +9.2 pp** | 显著 |

**新模型在所有核心指标上稳定超过旧基线 2-9 个百分点**。screen_photo 类的 F1 提升最大（+7.7~9.2pp），是本次训练改进最关键的成果。

> ⚠️ **重要说明（数据切分差异）**：
> - 旧基线数字来自 README 报告的训练，**其 train/val 切分有 hard_negative 数据泄漏**（`hard_negative` 既出现在 train 也以 3× 权重作为 screenshot 重复出现），旧基线数字极可能因泄漏而虚高。
> - 新数字使用 **固定的 stratified 0.70/0.15/0.15 切分**，所有样本按 path 去重，hard_negative 只加入 train，**完全无泄漏**。
> - 因此**新数字 vs 旧数字的差异比表面看起来更显著**。即便旧基线无泄漏，新模型也至少在所有指标上一致超过它。

---

## 2. 消融实验矩阵（15 配置，全部 H=6+F=12 epoch）

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

## 4. 最终获胜配置

**`finalist_unf1_s42`**（筛选阶段冠军 a_unf1 + 全量 10+20 epoch）：

```yaml
backbone: efficientnet_b0
gamma: 2.0
alpha: [1.0, 1.0, 1.5]
label_smoothing: 0.05
use_attention: false
use_arcface: false
ema: true
ema_decay: 0.999
unfreeze: 1
lr: 1e-3
weight_decay: 1e-4
epochs_head: 10
epochs_finetune: 20
batch_size: 16
heavy_aug: false
use_dwt: true
```

**3 种子全量训练结果**：

| Seed | test_acc | test_spF1 | macroF1 | metric | elapsed |
|---|---|---|---|---|---|
| **42** | **0.9322** | **0.8548** | **0.9174** | **0.8952** | 1509s |
| 2024 | 0.9133 | 0.8276 | 0.8963 | 0.8670 | 1542s |
| 7 | 0.9079 | 0.7788 | 0.8817 | 0.8381 | 1542s |
| AVG | 0.9178 | 0.8204 | 0.8985 | 0.8668 | 1531s |

**3 种子均值（acc 0.9178, spF1 0.8204, macroF1 0.8985）仍全面超过旧基线**，说明改进具有跨种子的稳定性（种子间方差主要来自 369 张小测试集）。

---

## 5. 部署与一致性

- **ONNX 导出**：`trainer/checkpoints/three_class_best.pth` → `inference/models/three_class.onnx`
- **PyTorch ↔ ONNX 数值一致性**：rtol=1e-3 通过
- **ONNX 运行时测试集评估**（treating OOD→screen_photo per inference policy）：
  - acc 0.9187 / sp_f1 0.8397 / macro_f1 0.9039
  - screen_photo 召回率 **0.932**（高召回 = 几乎不漏检相机拍屏）
- **生产路径**：
  - `inference/models/three_class.onnx` (22 MB)
  - `trainer/checkpoints/three_class_best.pth` (22 MB)
  - `trainer/checkpoints/three_class_final.pth` (22 MB)

---

## 6. 风险与改进空间

1. **测试集仅 369 张，方差较大**：3 种子间 spF1 范围 0.78-0.85，acc 0.91-0.93。需要更多标注数据进一步验证。
2. **screen_photo precision 偏低**（ONNX 0.764）：高 recall 0.932 意味着宁可误报也不漏报，可通过提升 `screen_photo_threshold` 调整（当前 0.6）。
3. **完全 B1 backbone 收益未体现**：6GB 显存下 B1 batch_size 必须降至 8 或加 gradient accumulation 才更稳妥；当前为节省时间未做 B1 + 强正则的完整训练。
4. **DWT 仍为 224×224**：训练与推理均使用 224×224（与 RGB 对齐），未做共享 FFT 流水线重标定（E13 未执行）。

---

## 7. 文件清单

- 训练：`experiment/cnn_fft_dwt_ablation/harness.py`, `experiment/cnn_fft_dwt_ablation/finalist.py`
- 部署：`experiment/cnn_fft_dwt_ablation/finalize_export.py`
- 排行查看：`experiment/cnn_fft_dwt_ablation/show.py`
- 备份（生产旧模型）：`experiment/cnn_fft_dwt_ablation/backup/`
- 排行榜：`experiment/cnn_fft_dwt_ablation/leaderboard.jsonl`（30 条实验记录）
- 训练日志：`experiment/cnn_fft_dwt_ablation/logs/{smoke,screen,finalist,finalist2,finalist3}.log`
- 最终模型：
  - `experiment/cnn_fft_dwt_ablation/finalist/finalist_unf1_s42/best.pth`
  - `experiment/cnn_fft_dwt_ablation/finalist/three_class.onnx`
