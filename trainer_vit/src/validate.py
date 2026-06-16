"""Validation and metrics computation for screen detector models.

Compute accuracy, precision, recall, F1-score, confusion matrix.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from .dataset import LABEL_NAMES


def compute_metrics(
    preds: list[int] | np.ndarray,
    labels: list[int] | np.ndarray,
    label_names: list[str] | None = None,
) -> dict:
    """Compute comprehensive metrics from predictions and labels.

    Args:
        preds: Predicted class indices
        labels: Ground truth class indices
        label_names: List of class names (defaults to LABEL_NAMES)

    Returns:
        Dictionary with all metrics
    """
    if label_names is None:
        label_names = LABEL_NAMES

    preds = np.array(preds)
    labels = np.array(labels)

    # Overall metrics
    accuracy = accuracy_score(labels, preds)
    precision_macro = precision_score(labels, preds, average="macro")
    recall_macro = recall_score(labels, preds, average="macro")
    f1_macro = f1_score(labels, preds, average="macro")

    # Per-class metrics
    precision_per_class = precision_score(labels, preds, average=None)
    recall_per_class = recall_score(labels, preds, average=None)
    f1_per_class = f1_score(labels, preds, average=None)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    # Build per-class metrics
    classes = {}
    for i, name in enumerate(label_names):
        classes[name] = {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
        }

    # Build confusion matrix dict
    confusion = {}
    for i, name in enumerate(label_names):
        confusion[name] = {}
        for j, name2 in enumerate(label_names):
            confusion[name][name2] = int(cm[i][j])

    return {
        "accuracy": float(accuracy),
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1": float(f1_macro),
        "classes": classes,
        "confusion_matrix": confusion,
        "total_samples": len(labels),
    }


def validate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    model_type: str = "deit",
) -> dict:
    """Validate model on validation set.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Device to validate on
        model_type: Model type for input handling

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 2:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
            elif len(batch) == 3:
                images, fft_input, labels = batch
                images = images.to(device)
                fft_input = fft_input.to(device)
                labels = labels.to(device)

                if model_type in ("fft_deit", "dwt_fft_deit"):
                    outputs = model(images, fft_input)
                else:
                    outputs = model(images)
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return compute_metrics(all_preds, all_labels)


def validate_from_checkpoint(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str = "outputs",
    device: str = "cpu",
    model_type: str = "deit",
) -> dict:
    """Validate model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Path to data directory
        output_dir: Path to output directory
        device: Device to validate on
        model_type: Model type

    Returns:
        Dictionary with validation metrics
    """
    from .dataset import create_dataloaders
    from .model import load_deit_model

    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading model...")
    model = load_deit_model(checkpoint_path, str(device))

    # Determine input mode
    input_mode = "rgb"
    if model_type in ("fft_deit", "dwt_fft_deit"):
        input_mode = "fft"

    # Create dataloaders
    logger.info("Creating dataloaders...")
    _, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4,
        val_split=0.2,
        input_mode=input_mode,
    )

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = validate_model(model, val_loader, device, model_type)

    # Save metrics
    metrics_path = output_path / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info("Validation complete!")
    logger.info(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    logger.info(f"Precision: {metrics['precision'] * 100:.2f}%")
    logger.info(f"Recall: {metrics['recall'] * 100:.2f}%")
    logger.info(f"F1-score: {metrics['f1'] * 100:.2f}%")
    logger.info(f"Metrics saved to: {metrics_path}")

    return metrics


def main() -> None:
    """Main validation entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate screen detector model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="deit",
        choices=["deit", "fft_deit", "dwt_fft_deit"],
        help="Model type",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/input",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Path to output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu/cuda)",
    )

    args = parser.parse_args()

    validate_from_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
