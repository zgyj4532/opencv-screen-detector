"""Training module for PAH-ViT (Patch-Aware Hybrid Vision Transformer).

Single-stage training with 3-class classification:
- natural, screenshot, screen_photo

Key features:
- EfficientViT-B0 backbone (CNN+ViT hybrid)
- Learnable Fourier Token Mixer (replaces handcrafted FFT/DWT)
- Patch Token Branch for fine-grained anomaly detection
- Patch Contrastive Loss for screen_photo vs screenshot distinction
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from . import config
from .augment import get_train_transforms, get_val_transforms
from .dataset import create_single_input_data_loaders
from .losses_pahvit import create_pahvit_criterion
from .model_pahvit import (
    PAHVitModel,
    create_pahvit_model,
    load_pahvit_model,
    save_pahvit_model,
)
from .validate import plot_confusion_matrix, plot_training_history, print_metrics


def validate_pahvit_model(
    model: PAHVitModel,
    val_loader,
    device: str = "cpu",
    class_names: list[str] | None = None,
) -> dict:
    """Validate PAH-ViT model on validation set.

    Args:
        model: PAH-ViT model
        val_loader: Validation data loader (rgb, labels)
        device: Device to use
        class_names: Class names

    Returns:
        dict with metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for rgb, labels in val_loader:
            rgb = rgb.to(device)
            labels = labels.to(device)

            logits, _ = model(rgb)
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    import numpy as np

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    precision_macro = precision_score(
        all_labels, all_preds, average="macro", zero_division=0
    )
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fpr_per_class = []
    for i in range(len(cm)):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
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


def train_one_epoch_pahvit(
    model: PAHVitModel,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str = "cpu",
    use_amp: bool = True,
) -> tuple[float, float, float, float]:
    """Train PAH-ViT model for one epoch with Mixed Precision.

    Args:
        model: PAH-ViT model
        train_loader: Training data loader (rgb, labels)
        criterion: Loss function (PAHVitLoss)
        optimizer: Optimizer
        device: Device to use
        use_amp: Whether to use Automatic Mixed Precision

    Returns:
        Tuple of (epoch_loss, epoch_acc, ce_loss, contrastive_loss)
    """
    model.train()

    running_loss = 0.0
    running_ce = 0.0
    running_contrastive = 0.0
    correct = 0
    total = 0

    amp_device = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

    for rgb, labels in train_loader:
        rgb = rgb.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda" if device == "cuda" else "cpu", enabled=use_amp):
            logits, anomaly_scores = model(rgb)
            total_loss, ce_loss, contrastive_loss = criterion(
                logits, anomaly_scores, labels
            )

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item() * rgb.size(0)
        running_ce += ce_loss.item() * rgb.size(0)
        running_contrastive += contrastive_loss.item() * rgb.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_ce = running_ce / total
    epoch_contrastive = running_contrastive / total

    return epoch_loss, epoch_acc, epoch_ce, epoch_contrastive


def train_pahvit(
    data_map: dict[str, list[str]] | None = None,
    class_names: list[str] | None = None,
    class_weights: list[float] | None = None,
    data_dir: Path | None = None,
    epochs_head: int = config.PAH_VIT_EPOCHS_HEAD,
    epochs_finetune: int = config.PAH_VIT_EPOCHS_FINETUNE,
    batch_size: int = config.PAH_VIT_BATCH_SIZE,
    learning_rate: float = config.PAH_VIT_LEARNING_RATE,
    device: str | None = None,
    lambda_contrastive: float = config.PAH_VIT_LAMBDA_CONTRASTIVE,
    use_weighted_sampler: bool = config.USE_WEIGHTED_SAMPLER,
) -> tuple[PAHVitModel, dict, dict]:
    """Train PAH-ViT model.

    Args:
        data_map: Data mapping {class_name: [source_dirs]}
        class_names: Class names for three-class classification
        class_weights: Class weights for imbalanced dataset
        data_dir: Data directory
        epochs_head: Epochs for head training (Stage A)
        epochs_finetune: Epochs for fine-tuning (Stage B)
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
        lambda_contrastive: Weight for Patch Contrastive Loss
        use_weighted_sampler: Whether to use WeightedRandomSampler

    Returns:
        Tuple of (model, history, final_metrics)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if data_dir is None:
        data_dir = config.DATA_DIR

    if data_map is None:
        data_map = config.THREE_CLASS_DATA_MAP

    if class_names is None:
        class_names = config.CLASS_NAMES_THREE_CLASS

    if class_weights is None:
        class_weights = config.CLASS_WEIGHTS_THREE_CLASS

    print(f"\n{'=' * 60}")
    print(f"Training PAH-ViT: {', '.join(class_names)}")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Lambda Contrastive: {lambda_contrastive}")
    print(f"Class Weights: {class_weights}")

    # Create data loaders (single input - RGB only)
    train_loader, val_loader, full_dataset = create_single_input_data_loaders(
        data_map=data_map,
        data_dir=data_dir,
        transform_train=get_train_transforms(),
        transform_val=get_val_transforms(),
        batch_size=batch_size,
        use_weighted_sampler=use_weighted_sampler,
    )

    print(f"Dataset size: {len(full_dataset)} images")
    train_size = int(len(full_dataset) * config.TRAIN_VAL_SPLIT)
    val_size = len(full_dataset) - train_size
    print(f"Train/Val split: {train_size}/{val_size}")

    # Create PAH-ViT model
    model = create_pahvit_model(
        model_name=config.PAH_VIT_MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True,
    )
    model = model.to(device)

    # Create loss criterion
    criterion = create_pahvit_criterion(
        lambda_contrastive=lambda_contrastive,
        class_weights=class_weights,
        focal_gamma=config.PAH_VIT_FOCAL_GAMMA,
    )
    print(f"Loss function: {type(criterion).__name__}")

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "best_metric": [],
    }

    best_val_acc = 0.0
    best_metric = 0.0

    screen_photo_class_idx = (
        class_names.index("screen_photo") if "screen_photo" in class_names else 2
    )

    # ==========================================
    # Stage A: Train classification head
    # ==========================================
    print(f"\n[Stage A] Training classification head ({epochs_head} epochs)")

    optimizer = optim.AdamW(
        list(model.token_mixer.parameters())
        + list(model.patch_branch.parameters())
        + list(model.fusion.parameters())
        + list(model.classifier.parameters()),
        lr=learning_rate,
        weight_decay=config.PAH_VIT_WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_head)

    for epoch in range(epochs_head):
        start_time = time.time()

        train_loss, train_acc, ce_loss, contrastive_loss = train_one_epoch_pahvit(
            model, train_loader, criterion, optimizer, device
        )

        val_metrics = validate_pahvit_model(model, val_loader, device, class_names)
        val_acc = val_metrics["accuracy"]

        screen_photo_f1 = val_metrics["f1_per_class"][screen_photo_class_idx]
        screen_photo_precision = val_metrics["precision_per_class"][screen_photo_class_idx]
        screen_photo_recall = val_metrics["recall_per_class"][screen_photo_class_idx]
        macro_f1 = val_metrics["f1_macro"]
        current_metric = (
            config.BEST_METRIC_F1_WEIGHT * screen_photo_f1
            + config.BEST_METRIC_ACCURACY_WEIGHT * val_acc
            + config.BEST_METRIC_MACRO_F1_WEIGHT * macro_f1
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["best_metric"].append(current_metric)

        if current_metric > best_metric:
            best_metric = current_metric
            best_val_acc = val_acc
            save_pahvit_model(
                model,
                str(config.CHECKPOINT_DIR / "pahvit_best.pth"),
                epoch=epoch,
                optimizer_state_dict=optimizer.state_dict(),
                best_val_acc=best_val_acc,
                best_metric=best_metric,
            )

        elapsed = time.time() - start_time
        print(
            f"  Epoch {epoch + 1}/{epochs_head} - "
            f"Loss: {train_loss:.4f} (CE: {ce_loss:.4f}, CL: {contrastive_loss:.4f}) - "
            f"Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f} - "
            f"SP F1: {screen_photo_f1:.4f} - SP P/R: {screen_photo_precision:.4f}/{screen_photo_recall:.4f} - "
            f"Metric: {current_metric:.4f} - Time: {elapsed:.1f}s"
        )

    # ==========================================
    # Stage B: Fine-tune with unfrozen layers
    # ==========================================
    print(f"\n[Stage B] Fine-tuning ({epochs_finetune} epochs)")

    model.unfreeze_backbone(num_layers=4)

    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": learning_rate * 0.1},
            {"params": model.token_mixer.parameters(), "lr": learning_rate * 0.1},
            {"params": model.patch_branch.parameters(), "lr": learning_rate},
            {"params": model.fusion.parameters(), "lr": learning_rate},
            {"params": model.classifier.parameters(), "lr": learning_rate},
        ],
        weight_decay=config.PAH_VIT_WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_finetune)

    for epoch in range(epochs_finetune):
        start_time = time.time()

        train_loss, train_acc, ce_loss, contrastive_loss = train_one_epoch_pahvit(
            model, train_loader, criterion, optimizer, device
        )

        val_metrics = validate_pahvit_model(model, val_loader, device, class_names)
        val_acc = val_metrics["accuracy"]

        screen_photo_f1 = val_metrics["f1_per_class"][screen_photo_class_idx]
        screen_photo_precision = val_metrics["precision_per_class"][screen_photo_class_idx]
        screen_photo_recall = val_metrics["recall_per_class"][screen_photo_class_idx]
        macro_f1 = val_metrics["f1_macro"]
        current_metric = (
            config.BEST_METRIC_F1_WEIGHT * screen_photo_f1
            + config.BEST_METRIC_ACCURACY_WEIGHT * val_acc
            + config.BEST_METRIC_MACRO_F1_WEIGHT * macro_f1
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["best_metric"].append(current_metric)

        if current_metric > best_metric:
            best_metric = current_metric
            best_val_acc = val_acc
            save_pahvit_model(
                model,
                str(config.CHECKPOINT_DIR / "pahvit_best.pth"),
                epoch=epochs_head + epoch,
                optimizer_state_dict=optimizer.state_dict(),
                best_val_acc=best_val_acc,
                best_metric=best_metric,
            )

        elapsed = time.time() - start_time
        print(
            f"  Epoch {epoch + 1}/{epochs_finetune} - "
            f"Loss: {train_loss:.4f} (CE: {ce_loss:.4f}, CL: {contrastive_loss:.4f}) - "
            f"Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f} - "
            f"SP F1: {screen_photo_f1:.4f} - SP P/R: {screen_photo_precision:.4f}/{screen_photo_recall:.4f} - "
            f"Metric: {current_metric:.4f} - Time: {elapsed:.1f}s"
        )

    # ==========================================
    # Final evaluation
    # ==========================================

    best_model = load_pahvit_model(
        str(config.CHECKPOINT_DIR / "pahvit_best.pth"),
        device=device,
    )
    best_model = best_model.to(device)

    final_metrics = validate_pahvit_model(best_model, val_loader, device, class_names)
    print_metrics(final_metrics, class_names)

    plot_confusion_matrix(
        final_metrics,
        class_names,
        save_path=str(config.LOG_DIR / "pahvit_confusion_matrix.png"),
    )

    plot_training_history(
        history,
        save_path=str(config.LOG_DIR / "pahvit_training_history.png"),
    )

    save_pahvit_model(
        model,
        str(config.CHECKPOINT_DIR / "pahvit_final.pth"),
        epoch=epochs_head + epochs_finetune - 1,
        best_val_acc=best_val_acc,
        best_metric=best_metric,
    )

    return model, history, final_metrics


def main():
    """Main entry point for PAH-ViT training."""
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    _, _, metrics = train_pahvit()

    print("\n" + "=" * 60)
    print("PAH-ViT Training Complete!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("=" * 60)
