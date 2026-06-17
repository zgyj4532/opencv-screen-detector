"""CNN validation script to generate metrics.json for comparison with ViT.

Generates metrics in the same format as ViT for comparison.
Supports single-stage three-class classification: natural, screenshot, screen_photo.
"""

import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .dataset import create_data_loaders
from .model import load_model

# Label names for three-class classification
LABEL_NAMES = ["natural", "screenshot", "screen_photo"]


def compute_cnn_metrics_three_class(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Compute comprehensive metrics for CNN model (single-stage three-class).

    Uses single model with 3-class output:
    - Class 0: natural
    - Class 1: screenshot
    - Class 2: screen_photo

    Args:
        model: Single 3-class CNN model (EfficientNet-B0 + FFT Branch)
        val_loader: Validation data loader
        device: Device to evaluate on

    Returns:
        Dictionary with all metrics (same format as ViT)
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            # Handle different batch formats
            if len(batch) == 3:
                rgb, fft, labels = batch
                rgb = rgb.to(device)
                fft = fft.to(device)
                labels = labels.to(device)

                # Single forward pass: 3-class output
                outputs = model(rgb, fft)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average="macro")
    recall_macro = recall_score(all_labels, all_preds, average="macro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    # Per-class metrics
    precision_per_class = cast(np.ndarray, precision_score(all_labels, all_preds, average=None))
    recall_per_class = cast(np.ndarray, recall_score(all_labels, all_preds, average=None))
    f1_per_class = cast(np.ndarray, f1_score(all_labels, all_preds, average=None))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Build per-class metrics
    classes = {}
    for i, name in enumerate(LABEL_NAMES):
        classes[name] = {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
        }

    # Build confusion matrix dict
    confusion = {}
    for i, name in enumerate(LABEL_NAMES):
        confusion[name] = {}
        for j, name2 in enumerate(LABEL_NAMES):
            confusion[name][name2] = int(cm[i][j])

    return {
        "model": "CNN+FFT (EfficientNet-B0) - Single-stage 3-class",
        "accuracy": float(accuracy),
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1_score": float(f1_macro),
        "classes": classes,
        "confusion_matrix": confusion,
        "total_samples": len(all_labels),
    }


def main():
    """Generate CNN metrics.json for comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate CNN metrics")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="trainer/checkpoints/three_class_best.pth",
        help="Path to 3-class CNN model checkpoint",
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

    args = parser.parse_args()

    # Setup
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load 3-class model
    print(f"Loading 3-class CNN model from {args.checkpoint}...")
    model = load_model(args.checkpoint, str(device))
    model = model.to(device)

    # Create dataloaders with three-class data map
    print("Creating dataloaders...")

    data_map = {
        "natural": ["natural_photo"],
        "screenshot": ["screenshot"],
        "screen_photo": ["screen_photo"],
    }

    _, val_loader, _ = create_data_loaders(
        data_map=data_map,
        data_dir=Path(args.data_dir),
        batch_size=32,
        num_workers=0,
    )

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_cnn_metrics_three_class(model, val_loader, device)

    # Save metrics
    metrics_path = output_path / "cnn_metrics.json"
    with Path(metrics_path).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nCNN Validation complete!")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall: {metrics['recall'] * 100:.2f}%")
    print(f"F1-score: {metrics['f1_score'] * 100:.2f}%")
    print("\nPer-class metrics:")
    for class_name, class_metrics in metrics["classes"].items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision'] * 100:.2f}%")
        print(f"    Recall: {class_metrics['recall'] * 100:.2f}%")
        print(f"    F1: {class_metrics['f1'] * 100:.2f}%")
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
