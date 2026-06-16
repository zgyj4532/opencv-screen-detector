"""Main entry point for ViT training (improved version)."""

import sys
from pathlib import Path

from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.train import train_model


def main() -> None:
    """Run ViT training with improved techniques."""
    # 设置代理环境变量
    import os

    os.environ["HTTP_PROXY"] = "http://localhost:7897"
    os.environ["HTTPS_PROXY"] = "http://localhost:7897"

    data_dir = Path(__file__).parent.parent / "data" / "input"
    output_dir = Path(__file__).parent / "outputs"

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Improved training config
    config = {
        "model_name": "vit_small_patch16_224",
        "batch_size": 32,
        "stage1_epochs": 10,  # 增加
        "stage2_epochs": 90,  # 大幅增加
        "num_workers": 0,  # Windows compatibility
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "drop_path_rate": 0.1,  # Stochastic Depth
        "label_smoothing": 0.1,  # Label Smoothing
        "mixup_prob": 0.5,  # Mixup/CutMix 概率
        "use_mixup": True,  # 启用 Mixup/CutMix
    }

    logger.info("=== ViT Improved Training ===")
    logger.info(f"Model: {config['model_name']}")
    logger.info(f"Stage 1 epochs: {config['stage1_epochs']}")
    logger.info(f"Stage 2 epochs: {config['stage2_epochs']}")
    logger.info(f"Drop path rate: {config['drop_path_rate']}")
    logger.info(f"Label smoothing: {config['label_smoothing']}")
    logger.info(f"Mixup/CutMix: {config['use_mixup']}")

    # Run training
    metrics = train_model(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        config=config,
    )

    logger.info("\n=== Final Metrics ===")
    logger.info(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    logger.info(f"Precision: {metrics['precision'] * 100:.2f}%")
    logger.info(f"Recall: {metrics['recall'] * 100:.2f}%")
    logger.info(f"F1-score: {metrics['f1_score'] * 100:.2f}%")


if __name__ == "__main__":
    main()
