"""Training module for screen detector V3.

Single-stage training with 3-class classification:
- natural, screenshot, screen_photo

Optimizations:
- Weighted Loss for class imbalance
- Focal Loss for hard examples
- Center Loss for intra-class compactness
- OHEM for hard example mining
- ArcFace angular margin classifier
- FFT/DWT attention modules
- Oversampling with WeightedRandomSampler
- Mixed Precision training
"""

# pyright: reportPrivateImportUsage=none
import time
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from . import config
from .augment import get_train_transforms, get_val_transforms
from .dataset import create_data_loaders
from .losses import CombinedLoss, create_criterion
from .model import create_model, load_model, save_model
from .threshold_optimizer import optimize_thresholds
from .validate import (
    plot_confusion_matrix,
    plot_training_history,
    print_metrics,
    validate_model,
)


def train_one_epoch(
    model: nn.Module,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str = "cpu",
    use_amp: bool = True,
    use_arcface: bool = False,
    use_center_loss: bool = False,
    center_loss_optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Train model for one epoch with Mixed Precision.

    Args:
        model: Model to train
        train_loader: Training data loader (rgb, fft, dwt, labels)
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        use_amp: Whether to use Automatic Mixed Precision
        use_arcface: Whether model uses ArcFace classifier
        use_center_loss: Whether to use combined loss with center loss
        center_loss_optimizer: Optimizer for center loss parameters

    Returns:
        Tuple of (epoch_loss, epoch_acc)
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # Mixed Precision
    amp_device = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

    for rgb, fft, dwt, labels in train_loader:
        rgb = rgb.to(device)
        fft = fft.to(device)
        dwt = dwt.to(device)
        labels = labels.to(device)

        # Forward pass with AMP
        with torch.amp.autocast("cuda" if device == "cuda" else "cpu", enabled=use_amp):
            if use_arcface:
                # ArcFace returns (logits, features)
                outputs, features = model(rgb, fft, dwt, labels)

                if use_center_loss and isinstance(criterion, CombinedLoss):
                    loss, _focal_loss, _center_loss = criterion(outputs, features, labels)
                else:
                    loss = criterion(outputs, labels)
            else:
                outputs = model(rgb, fft, dwt)

                if use_center_loss and isinstance(criterion, CombinedLoss):
                    # For non-ArcFace, extract features separately
                    features = model.get_features(rgb, fft, dwt)
                    loss, _focal_loss, _center_loss = criterion(outputs, features, labels)
                else:
                    loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        if center_loss_optimizer is not None:
            center_loss_optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        if center_loss_optimizer is not None:
            scaler.step(center_loss_optimizer)

        scaler.update()

        # Statistics
        running_loss += loss.item() * rgb.size(0)

        if use_arcface:
            # For ArcFace, get predictions from logits
            _, predicted = torch.max(outputs, 1)
        else:
            _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train_three_class(
    data_map: dict[str, list[str]] | None = None,
    class_names: list[str] | None = None,
    class_weights: list[float] | None = None,
    data_dir: Path | None = None,
    epochs_head: int = config.EPOCHS_HEAD,
    epochs_finetune: int = config.EPOCHS_FINETUNE,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    device: str | None = None,
    use_focal_loss: bool = config.USE_FOCAL_LOSS,
    use_weighted_sampler: bool = config.USE_WEIGHTED_SAMPLER,
    use_center_loss: bool = config.USE_CENTER_LOSS,
    center_loss_weight: float = config.CENTER_LOSS_WEIGHT,
    use_ohem: bool = config.USE_OHEM,
    ohem_hard_ratio: float = config.OHEM_HARD_RATIO,
    use_arcface: bool = config.USE_ARCFACE,
    arcface_scale: float = config.ARCFACE_SCALE,
    arcface_margin: float = config.ARCFACE_MARGIN,
    use_fft_attention: bool = config.USE_FFT_ATTENTION,
    attention_type: str = config.FFT_ATTENTION_TYPE,
    use_adaptive_threshold: bool = config.USE_ADAPTIVE_THRESHOLD,
) -> tuple[nn.Module, dict, dict]:
    """Train a single-stage three-class classifier.

    Args:
        data_map: Data mapping {class_name: [source_dirs]}
        class_names: Class names for three-class classification
        class_weights: Class weights for imbalanced dataset
        data_dir: Data directory
        epochs_head: Epochs for head training
        epochs_finetune: Epochs for fine-tuning
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
        use_focal_loss: Whether to use Focal Loss
        use_weighted_sampler: Whether to use WeightedRandomSampler
        use_center_loss: Whether to use Center Loss
        center_loss_weight: Weight for Center Loss
        use_ohem: Whether to use OHEM
        ohem_hard_ratio: Ratio of hard samples for OHEM
        use_arcface: Whether to use ArcFace classifier
        arcface_scale: ArcFace scale factor
        arcface_margin: ArcFace angular margin
        use_fft_attention: Whether to use FFT attention
        attention_type: Type of attention ('cbam', 'coordinate')
        use_adaptive_threshold: Whether to optimize thresholds

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
    print(f"Training Three-Class Classifier: {', '.join(class_names)}")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print("\n--- Optimization Modules ---")
    print(f"Focal Loss: {use_focal_loss}")
    print(f"Center Loss: {use_center_loss} (weight={center_loss_weight})")
    print(f"OHEM: {use_ohem} (ratio={ohem_hard_ratio})")
    print(f"ArcFace: {use_arcface} (s={arcface_scale}, m={arcface_margin})")
    print(f"FFT Attention: {use_fft_attention} (type={attention_type})")
    print(f"Adaptive Threshold: {use_adaptive_threshold}")
    print(f"Weighted Sampler: {use_weighted_sampler}")
    print(f"Class Weights: {class_weights}")

    # Create data loaders
    train_loader, val_loader, full_dataset = create_data_loaders(
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

    # Feature dimension: EfficientNet (1280) + FFT/DWT (256) = 1536
    feat_dim = 1536

    # Create model with optimization options
    model = create_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True,
        use_dwt=True,
        use_arcface=use_arcface,
        arcface_scale=arcface_scale,
        arcface_margin=arcface_margin,
        use_fft_attention=use_fft_attention,
        attention_type=attention_type,
    )
    model = model.to(device)

    # Create loss criterion
    criterion = create_criterion(
        use_focal_loss=use_focal_loss,
        class_weights=class_weights,
        focal_gamma=config.FOCAL_LOSS_GAMMA,
        use_center_loss=use_center_loss,
        center_weight=center_loss_weight,
        num_classes=config.NUM_CLASSES,
        feat_dim=feat_dim,
        use_ohem=use_ohem,
        ohem_hard_ratio=ohem_hard_ratio,
    )
    criterion = criterion.to(device)
    print(f"Loss function: {type(criterion).__name__}")

    # Separate optimizer for center loss if used
    center_loss_optimizer = None
    if use_center_loss and isinstance(criterion, CombinedLoss):
        center_loss_optimizer = optim.SGD(
            criterion.center_criterion.parameters(),
            lr=0.5,
        )

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_metric": [],
    }

    best_val_acc = 0.0
    best_metric = 0.0  # 0.5 * screen_photo_f1 + 0.3 * accuracy + 0.2 * macro_f1

    # screen_photo is class index 2
    screen_photo_class_idx = class_names.index("screen_photo") if "screen_photo" in class_names else 2

    # ==========================================
    # Stage A: Train classification head
    # ==========================================
    print(f"\n[Stage A] Training classification head ({epochs_head} epochs)")

    optimizer = optim.AdamW(
        list(model.classifier.parameters()) + list(model.freq_branch.parameters()),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_head)

    for epoch in range(epochs_head):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_arcface=use_arcface,
            use_center_loss=use_center_loss,
            center_loss_optimizer=center_loss_optimizer,
        )

        val_metrics = validate_model(model, val_loader, device, class_names)
        val_acc = val_metrics["accuracy"]

        # Calculate best_metric: 0.5 * screen_photo_f1 + 0.3 * accuracy + 0.2 * macro_f1
        screen_photo_recall = val_metrics["recall_per_class"][screen_photo_class_idx]
        screen_photo_precision = val_metrics["precision_per_class"][screen_photo_class_idx]
        screen_photo_f1 = val_metrics["f1_per_class"][screen_photo_class_idx]
        macro_f1 = val_metrics["f1_macro"]
        current_metric = (
            config.BEST_METRIC_F1_WEIGHT * screen_photo_f1
            + config.BEST_METRIC_ACCURACY_WEIGHT * val_acc
            + config.BEST_METRIC_MACRO_F1_WEIGHT * macro_f1
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(0.0)
        history["val_acc"].append(val_acc)
        history["best_metric"].append(current_metric)

        if current_metric > best_metric:
            best_metric = current_metric
            best_val_acc = val_acc
            save_model(
                model,
                str(config.CHECKPOINT_DIR / "three_class_best.pth"),
                epoch=epoch,
                optimizer_state_dict=optimizer.state_dict(),
                best_val_acc=best_val_acc,
            )

        elapsed = time.time() - start_time
        print(
            f"  Epoch {epoch + 1}/{epochs_head} - "
            f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
            f"Val Acc: {val_acc:.4f} - SP F1: {screen_photo_f1:.4f} - "
            f"SP P/R: {screen_photo_precision:.4f}/{screen_photo_recall:.4f} - "
            f"Metric: {current_metric:.4f} - Time: {elapsed:.1f}s"
        )

    # ==========================================
    # Stage B: Fine-tune with unfrozen layers
    # ==========================================
    print(f"\n[Stage B] Fine-tuning ({epochs_finetune} epochs)")

    model.unfreeze_backbone(num_layers=6)

    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": learning_rate * 0.1},
            {"params": model.freq_branch.parameters(), "lr": learning_rate * 0.1},
            {"params": model.classifier.parameters(), "lr": learning_rate},
            {"params": model.spatial_norm.parameters(), "lr": learning_rate},
            {"params": model.freq_norm.parameters(), "lr": learning_rate},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_finetune)

    for epoch in range(epochs_finetune):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_arcface=use_arcface,
            use_center_loss=use_center_loss,
            center_loss_optimizer=center_loss_optimizer,
        )

        val_metrics = validate_model(model, val_loader, device, class_names)
        val_acc = val_metrics["accuracy"]

        # Calculate best_metric: 0.5 * screen_photo_f1 + 0.3 * accuracy + 0.2 * macro_f1
        screen_photo_recall = val_metrics["recall_per_class"][screen_photo_class_idx]
        screen_photo_precision = val_metrics["precision_per_class"][screen_photo_class_idx]
        screen_photo_f1 = val_metrics["f1_per_class"][screen_photo_class_idx]
        macro_f1 = val_metrics["f1_macro"]
        current_metric = (
            config.BEST_METRIC_F1_WEIGHT * screen_photo_f1
            + config.BEST_METRIC_ACCURACY_WEIGHT * val_acc
            + config.BEST_METRIC_MACRO_F1_WEIGHT * macro_f1
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(0.0)
        history["val_acc"].append(val_acc)
        history["best_metric"].append(current_metric)

        if current_metric > best_metric:
            best_metric = current_metric
            best_val_acc = val_acc
            save_model(
                model,
                str(config.CHECKPOINT_DIR / "three_class_best.pth"),
                epoch=epochs_head + epoch,
                optimizer_state_dict=optimizer.state_dict(),
                best_val_acc=best_val_acc,
            )

        elapsed = time.time() - start_time
        print(
            f"  Epoch {epoch + 1}/{epochs_finetune} - "
            f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
            f"Val Acc: {val_acc:.4f} - SP Precision: {screen_photo_precision:.4f} - "
            f"SP Recall: {screen_photo_recall:.4f} - Metric: {current_metric:.4f} - Time: {elapsed:.1f}s"
        )

    # ==========================================
    # Final evaluation
    # ==========================================

    best_model = load_model(
        str(config.CHECKPOINT_DIR / "three_class_best.pth"),
        device=device,
    )
    best_model = best_model.to(device)

    final_metrics = validate_model(best_model, val_loader, device, class_names)
    print_metrics(final_metrics, class_names)

    # Adaptive threshold optimization
    if use_adaptive_threshold:
        print("\n" + "=" * 60)
        print("Adaptive Threshold Optimization")
        print("=" * 60)

        # Get probabilities from best model
        best_model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for rgb, fft, dwt, labels in val_loader:
                rgb = rgb.to(device)
                fft = fft.to(device)
                dwt = dwt.to(device)

                outputs = best_model(rgb, fft, dwt)
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())

        import numpy as np

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Optimize thresholds
        threshold_result = optimize_thresholds(
            probabilities=all_probs,
            labels=all_labels,
            class_names=class_names,
            target_class=config.ADAPTIVE_THRESHOLD_TARGET,
        )

        print("\nMetrics with optimized thresholds:")
        print(f"  Accuracy: {threshold_result.metrics['accuracy']:.4f}")
        print(f"  SP Precision: {threshold_result.metrics['precision_per_class'][screen_photo_class_idx]:.4f}")
        print(f"  SP Recall: {threshold_result.metrics['recall_per_class'][screen_photo_class_idx]:.4f}")
        print(f"  SP F1: {threshold_result.metrics['f1_per_class'][screen_photo_class_idx]:.4f}")

    plot_confusion_matrix(
        final_metrics,
        class_names,
        save_path=str(config.LOG_DIR / "three_class_confusion_matrix.png"),
    )

    plot_training_history(
        history,
        save_path=str(config.LOG_DIR / "three_class_training_history.png"),
    )

    save_model(
        model,
        str(config.CHECKPOINT_DIR / "three_class_final.pth"),
        epoch=epochs_head + epochs_finetune - 1,
        best_val_acc=best_val_acc,
    )

    return model, history, final_metrics


def main():
    """Main entry point for three-class training."""
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Train three-class classifier
    _, _, metrics = train_three_class()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Three-Class Accuracy: {metrics['accuracy']:.4f}")
    print(f"Best Validation Accuracy: {metrics['accuracy']:.4f}")
    print("=" * 60)
