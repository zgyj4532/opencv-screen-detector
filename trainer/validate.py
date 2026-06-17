"""Validation module for screen detector V3 training.

Includes precision, recall, F1, FPR metrics (修正 #11).
"""

# pyright: reportPrivateImportUsage=none
from collections.abc import Iterable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def validate_model(
    model: nn.Module,
    val_loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: str = "cpu",
    class_names: list[str] | None = None,  # noqa: ARG001
):
    """Validate model on validation set.

    Returns:
        dict with metrics: accuracy, precision, recall, f1, fpr, confusion_matrix
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for rgb, fft, labels in val_loader:
            rgb = rgb.to(device)
            fft = fft.to(device)
            labels = labels.to(device)

            outputs = model(rgb, fft)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # Overall metrics
    precision_macro = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # False Positive Rate (修正 #11)
    fpr_per_class = []
    for i in range(len(cm)):
        fp = cm[:, i].sum() - cm[i, i]  # False positives for class i
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]  # True negatives
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fpr_per_class.append(fpr)

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "fpr_per_class": np.array(fpr_per_class),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def print_metrics(
    metrics: dict[str, Any],
    class_names: list[str] | None = None,
) -> None:
    """Print validation metrics."""
    if class_names is None:
        class_names = ["class_0", "class_1"]

    print("\n" + "=" * 60)
    print("Validation Metrics")
    print("=" * 60)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision:  {metrics['precision_macro']:.4f}")
    print(f"Macro Recall:     {metrics['recall_macro']:.4f}")
    print(f"Macro F1:         {metrics['f1_macro']:.4f}")

    print("\nPer-class Metrics:")
    print("-" * 60)
    header = f"{'Class':<15} {'Precision':<12} {'Recall':<12}"
    print(f"{header} {'F1':<12} {'FPR':<12} {'Acc':<12}")
    print("-" * 60)

    for i, class_name in enumerate(class_names):
        print(
            f"{class_name:<15} "
            f"{metrics['precision_per_class'][i]:<12.4f} "
            f"{metrics['recall_per_class'][i]:<12.4f} "
            f"{metrics['f1_per_class'][i]:<12.4f} "
            f"{metrics['fpr_per_class'][i]:<12.4f} "
            f"{metrics['per_class_accuracy'][i]:<12.4f}"
        )

    print("\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    header = " ".join(f"{name:>12}" for name in class_names)
    print(f"{'Predicted':>12} {header}")
    print(f"{'Actual':>12}")
    for i, name in enumerate(class_names):
        row = " ".join(f"{cm[i, j]:>12d}" for j in range(len(class_names)))
        print(f"{name:>12} {row}")
    print("=" * 60)


def plot_confusion_matrix(
    metrics: dict[str, Any],
    class_names: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot confusion matrix."""
    if class_names is None:
        class_names = ["class_0", "class_1"]

    cm = metrics["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)  # pyright: ignore[reportAttributeAccessIssue]
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()


def plot_training_history(history, save_path=None) -> None:
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()
