"""Experiment runner for screen detector ablation study.

Experiments:
- Exp0: CNN Only (EfficientNet-B0) - control group
- Exp1: CNN+FFT (EfficientNet-B0 + FFT Branch)
- Exp2: DeiT (deit_small_patch16_224)
- Exp3: FFT+DeiT (dual-stream)
- Exp4: DWT+FFT+DeiT (triple-stream)

Each experiment runs N trials and outputs:
- metrics.json with accuracy/precision/recall/f1/confusion_matrix
- recall_ranking.md with ScreenPhoto Recall ranking
"""
