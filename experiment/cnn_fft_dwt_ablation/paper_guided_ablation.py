"""Run the 2026-07-27 paper-guided ablation queue.

Each candidate trains for five total epochs (2 frozen-head epochs + 3
fine-tuning epochs) on the current data/input split.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiment.cnn_fft_dwt_ablation.harness import ExpConfig, run_configs


def configs() -> list[ExpConfig]:
    head_epochs = 2
    finetune_epochs = 3
    common = {
        "epochs_head": head_epochs,
        "epochs_finetune": finetune_epochs,
        "batch_size": 16,
        "focus_weight": 2.0,
        "gamma": 2.0,
        "alpha": [1.0, 1.0, 1.5],
        "smoothing": 0.05,
        "ema": True,
        "use_dwt": True,
        "heavy_aug": False,
    }
    return [
        ExpConfig(
            id="pg260727_b0_unf3",
            desc="paper-guided baseline: EfficientNet-B0 FFT+DWT unfreeze3",
            backbone="efficientnet_b0",
            unfreeze=3,
            **common,
        ),
        ExpConfig(
            id="pg260727_b0_unf1",
            desc="small-data regularized baseline: EfficientNet-B0 FFT+DWT unfreeze1",
            backbone="efficientnet_b0",
            unfreeze=1,
            **common,
        ),
        ExpConfig(
            id="pg260727_effv2b0",
            desc="EfficientNetV2-B0 backbone with FFT+DWT branch",
            backbone="tf_efficientnetv2_b0",
            unfreeze=1,
            **common,
        ),
        ExpConfig(
            id="pg260727_cnv2_atto",
            desc="ConvNeXt V2 Atto backbone with FFT+DWT branch",
            backbone="convnextv2_atto",
            unfreeze=1,
            **common,
        ),
        ExpConfig(
            id="pg260727_mobileone",
            desc="MobileOne-S0 deployment-oriented backbone with FFT+DWT branch",
            backbone="mobileone_s0",
            unfreeze=1,
            **common,
        ),
    ]


if __name__ == "__main__":
    run_configs(configs())
