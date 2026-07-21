=== RECOMMENDED DEFAULT CONFIG ===
### 数据
- 删除 `THREE_CLASS_DATA_MAP['screenshot']` 中的 `'hard_negative'`;`__init__` 按 path 去重 self.samples
- `create_data_loaders` 返回 `(train_loader, val_loader, test_loader)`,三段切分 0.7/0.15/0.15,**StratifiedShuffleSplit**
- `train_data/val_data/test_data` 互斥校验:`set(train_idx) ∩ set(val_idx) == ∅`
- `HardNegative` 全部并入 train;不加 test/val,以免破坏真实分布
- `NUM_WORKERS=4`,`pin_memory=True`

### 优化器 & 阶段
- Stage A(10 epoch):AdamW 覆盖 `classifier + freq_branch + spatial_norm + freq_norm`,`lr=3e-3`,`weight_decay=1e-4`,CosineAnnealingLR
- Stage B(20 epoch):解冻 `backbone.blocks[-2:] + conv_head + bn2 + spatial_norm + freq_norm`,`lr=1e-4`,CosineAnnealingLR
- EMA(decay=0.999)全程开启,val/test 与导出用 `ema.module`

### 损失 & 分类头
- **FocalLoss(γ=2.0, alpha=[1.0, 1.0, 1.5], label_smoothing=0.05)**
- **关闭 OHEM / CenterLoss / ArcFace**(默认)
- 分类头:`LayerNorm(1536) → Dropout(0.3) → Linear(1536, 3)`

### 频域对齐
- `compute_fft_spectrum` / `compute_dwt_features` 在 `TransformSubset` 中,**对增强后的 numpy 图像**计算
- 删除多余的 `A.Resize(224,224)` 让 RandomResizedCrop 一次到位
- 删除 `TransformSubset` 的 uint8 round-trip

### 训练数据增广
- RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.333))
- HorizontalFlip(p=0.5)、Rotate(±15)、Perspective(scale=0.05)
- ColorJitter(0.2, 0.2, 0.2, 0.05)、MoireSimulation(p=0.3)、ScreenReflection(p=0.2)、GaussNoise(p=0.2)、CoarseDropout(p=0.2)

### 评估 & 部署
- `best_checkpoint` 在 **val** 上选(基于 `0.5*sp_f1 + 0.3*acc + 0.2*macro_f1`)
- `final_metrics` 与 `optimize_thresholds` 在 **test** 上跑,grid 范围 [0.30, 0.80] step 0.025
- `three_class_final.pth` 保存 `best_model.state_dict()`,而非最后 epoch
- 导出 ONNX 时 dummy_dwt 形状与实际训练一致(224 或统一改为 112)
- 共享 `RGB_MEAN/STD/FFT_NORM_MEAN/STD` 由 `shared/constants.py` 单一来源维护

### Inference 校准(导出后回写)
- 用 test 集校准:`confidence_high=0.92`, `confidence_medium=0.75`, `ood_threshold=0.45` 在新的最优阈值上重新扫描
- TTA:水平翻转 + 原图,概率平均
- 阈值后处理:屏幕照片概率 ≥ 0.60 → 强制 screen_photo

=== EXPERIMENT MATRIX (by priority) ===
P1 [E01] 数据净化 + train/val/test 三段切分
    hyp: 消除 hard_negative 重复加载与样本级泄漏后,val 指标更接近真实泛化;三段切分(0.7/0.15/0.15 stratified)给阈值优化与 final_metrics 提供无偏集合。
    change: 从 THREE_CLASS_DATA_MAP['screenshot'] 移除 'hard_negative';create_data_loaders 返回 (train_loader, val_loader, test_loader);self.samples 按 path 去重;用 StratifiedShuffleSplit 切分;best_checkpoint 在 val 上选,final_metrics/threshold 在 test 上评估。
P2 [E02] Loss 组合降复杂度:Focal(γ=2)+无 OHEM+无 Center+无 ArcFace
    hyp: 小数据 + 3 类场景下,四种增强机制叠加会过拟合单一样本;只保留 Focal γ=2 + alpha=[1,1,1.5] + Linear head 比当前 Focal γ=3+OHEM+Center+ArcFace 更稳定,屏幕照片 F1 应更高。
    change: config: FOCAL_LOSS_GAMMA=2.0;USE_OHEM=False;USE_CENTER_LOSS=False;USE_ARCFACE=False;保留 CLASS_WEIGHTS=[1,1,1.5]。把 OHEMLoss/CombinedLoss 的 else 分支修复为保留 weight(防回归)。
P3 [E03] EMA + 真实 val_loss
    hyp: 用 EMA(decay 0.999)覆盖主分类头权重,选出对噪声更鲁棒的 checkpoint;同时让 validate_model 返回带 loss 的指标,plot 不再是 0 平线。
    change: 在 train.py 增加 ModelEMA(model, decay=0.999),val 用 ema.module;validate_model 返回 (acc, f1, sp_f1, loss);train.py:327 把 0.0 替换为真实 val_loss;评估时同时报告 raw 与 ema 指标。
P4 [E04] Label Smoothing 0.05
    hyp: label_smoothing=0.05 在 3 类小数据上能减少过自信预测,与 OOD 阈值 0.45 配合更协调,可小幅涨 acc/macro_F1。
    change: FocalLoss.__init__ 增加 smoothing 参数,forward 时把 targets 做 one-hot * (1-smoothing) + smoothing/3;默认 LABEL_SMOOTHING=0.05;A/B 对照 (0, 0.05, 0.10)。
P5 [E05] 屏幕照片类权扫描 alpha=[1,1,1.5/2.0/2.5]
    hyp: 当前 alpha=1.5 对屏幕照片提升不足;在 387 张小样本上把 alpha 调到 2.0-2.5 应能涨 screen_photo recall 与 F1,但超过 3.0 会导致 precision 雪崩。
    change: config 增加 CLASS_WEIGHTS_CANDIDATES = [[1,1,1.0],[1,1,1.5],[1,1,2.0],[1,1,2.5]] 三次小训练取最佳,alpha 直接作用于 FocalLoss.register_buffer。
P6 [E06] Backbone b0 → b1(高风险)
    hyp: B1 参数量约 6.5M vs B0 5.3M,容量提升有助于捕获更细的频域/RGB 区别;但 2900 张小数据过拟合风险显著,需配合强增广 + 早停。
    change: timm.create_model('efficientnet_b1', pretrained=True, num_classes=0)替换 b0;Stage B 仅解冻最后 1 个 MBConv stage;若 val 出现 train-acc 95%+ val-acc 下降 >3%,回退到 b0。
P7 [E07] Stage B 解冻深度扫描(last 1/2/3 stages)
    hyp: 当前 num_layers=6 在 MBConv 顶层粒度下等价于解冻 99.98% 的参数;修复 unfreeze_backbone 后,真正解冻 last 1/2/3 stages 是控制过拟合的关键变量。
    change: unfreeze_backbone 重写为基于 self.backbone.blocks.children();三次实验分别解冻 1/2/3 个 MBConv stage + conv_head+bn2;以 val macro_F1 + screen_photo F1 综合选优。
P8 [E08] FFT/DWT 跨分支对齐(基于增强后图像)
    hyp: 训练时频域分支看到与 RGB 完全一致的图像(包括 MoireSimulation/ScreenReflection/Perspective),能学到与 RGB 互补但对应一致的特征,部署时推理侧拿到的频域与训练分布完全相同。
    change: compute_fft_spectrum/compute_dwt_features 移到 TransformSubset 的 augment 之后;TwoInputDataset 只返回 (uint8_np, label);删去多余的 uint8 round-trip。
P9 [E09] 频域分支消融:仅 FFT / 仅 DWT / 全开
    hyp: 频域分支在 256 维空间里是否有边际增益未量化;FFT 偏全局频谱,DWT 偏局部子带,可能两者之一就够;消融可指导后续架构。
    change: 三组实验:USE_DWT=False(仅 FFT 256)、USE_FFT=False(仅 DWT 256)、USE_DWT=True(FFT+DWT 256);控制 freq_branch 输出维度对齐;比较 acc / sp_f1 / 参数量 / 单 batch 耗时。
P10 [E10] CBAM 与 CoordAttention 消融
    hyp: 在空间分支 1280 维后挂 CBAM/Coord 注意力,可能帮助定位屏幕反光/摩尔纹等局部伪影;但小数据上加注意力易过拟合,需配合 Stage A 不解冻 backbone。
    change: USE_CBAM ∈ {True, False} × USE_COORD ∈ {True, False} 四组实验;评估时记录屏幕照片 TP/FP 变化;若 val 涨 / test 不涨 → 过拟合,关闭。
P11 [E11] Stage A 修复:把 LayerNorm 纳入优化器
    hyp: 把 model.spatial_norm / model.freq_norm 加入 Stage A 优化器,让 head 学到与可适配归一化输出对齐的分类面,Stage B 微调负担下降,整体 macro_F1 提升。
    change: Stage A AdamW 加入 list(model.spatial_norm.parameters()) + list(model.freq_norm.parameters());Stage B 同改。
P12 [E12] Mixup / CutMix(混合增强)
    hyp: Mixup α=0.2 / CutMix α=1.0 在小数据集上常带来 +0.5-1.5% 的 acc 与 macro_F1 提升;但与屏幕照片加权 alpha=1.5 叠加可能造成混淆样本被强制学习。
    change: augment.py 增加 mixup/cutmix 路径,α 扫描 {0.2, 0.4};概率扫描 {0.3, 0.5};关闭 focal 中的 alpha 或同步降低 alpha 避免过度补偿。
P13 [E13] FFT_NORM_MEAN/STD 重标定
    hyp: 当前用 ImageNet 像素 mean/std 归一化 log-magnitude 频谱,把低频峰值压到 +2.4 而大部分 bin 在 -2 附近;基于训练集实际统计(per-bin mean/std)重标定,频域分支的 SNR 提升,可能涨 0.3-1% 的 acc。
    change: 扫描 ~500 张训练图,在共享 FFT 流水线后取 mean/std;更新 shared/fft_transform.py 的 FFT_NORM_MEAN/STD;trainer 与 inference 共享同一份常量。
P14 [E14] 自适应阈值在独立 test 集上重做
    hyp: 把阈值搜索搬移到三段切分中的 test 集(或 k-fold cross-val 平均),消除 best-checkpoint 选优偏差,得到更可信的 screen_photo F1,反哺 inference/config.py 的 confidence_high/medium/ood_threshold。
    change: 训练结束时用 test_loader 取 probs/labels,在 screen_photo threshold ∈ [0.30,0.80] step=0.025 上做 grid,选 max screen_photo F1 对应的阈值;同时把自然/截图类阈值置为 0.5(去掉 best_thresholds 中的死数据)。
P15 [E15] DWT 形状统一(112×112 vs 224×224)
    hyp: shared.compute_dwt_features 实际把每个子带 resize 回 224,但所有 docstring 与 ONNX 期望互相矛盾;统一到 112(去掉无信息的上采样)能砍掉 4x DWT 通道计算,顺便修正 export_onnx.py 的 dummy 形状。
    change: 删除 shared/fft_transform.py:136-141 的 cv2.resize,直接返回 (1,4,112,112);同步 export_onnx.py:48 与 :115 的 dummy_dwt 形状;同步更新所有 docstring;加 shape 断言。
P16 [E16] Class-balanced Samper v.s. 当前 WeightedRandomSampler
    hyp: 当前 sampler 在 hard_negative 重复加载下权重错误;修复后改为按 effective number of samples 的 class-balanced 重权 α=0.999,可比 inverse-frequency 采样更稳。
    change: 实现 class_balanced_weights = (1-β^n_c)/(1-β);sampler 用 torch.utils.data.WeightedRandomSampler;β ∈ {0.9, 0.99, 0.999} 三次扫描。

=== RISKS ===
- E06(B1 backbone)在 2900 张图上极易过拟合:必须配合 Stage B 仅解冻 1 个 MBConv stage + EMA + 强增广;若 test macro_F1 比 E01 默认低超过 1.5%,立即回退 B0。
- E12(Mixup/CutMix)+ alpha=[1,1,1.5] 的组合会放大 class-imbalance 的副作用:建议先在 E01 默认上验证 Mixup 的边际增益(关闭 alpha),再叠加 alpha;若 sp_f1 反而下降,放弃 Mixup。
- E04(label smoothing 0.10)与 Focal γ=2 的组合在三类小数据上可能让屏幕照片 logits 偏平:推荐先 0.05,观察 sp_f1;不要超过 0.10。
- E02 关闭 OHEM/Center/ArcFace 是收益最大的单一改动,但如果最佳 baseline 复现结果意外变差(可能原配置在某些子集上偶然过拟合),需要重新评估 ArcFace 改为温和参数(m=0.10, s=15)而非直接弃用。
- E15(DWT 形状变更)需同步更新 export_onnx.py 与 inference/fft_service 的输入,任何遗漏都会让 ONNX runtime 形状断言失败;只在一组验证实验成功后再正式切换。
- E03 EMA 与 best-checkpoint 选优:需明确 val 评估是用 raw 还是 ema 权重;否则两次 baseline 不可比。建议 val/raw + val/ema 都记录,以 ema 为部署指标,raw 为回归监控。
- 硬件(单卡 RTX 3060 6GB)上 B1 backbone + B0 batch=16 内存已经吃紧:若 E06 触发 OOM,先把 batch=16→8,再考虑 gradient accumulation 2 steps;不要盲目降图像分辨率。
- E14 阈值搜索只在 test 上做会减少"in-sample 高估",但 test 集只有 ~440 张,屏幕照片约 ~58 张,grid step 0.025 上的方差仍大;建议对 best_threshold 做 bootstrap 95% CI 再决定上线阈值。