"""Quick benchmark script for PAH-ViT model.

Runs a single training with reduced epochs to verify the model works
correctly before running the full 5-fold validation.

Usage:
    uv run python -m trainer.run_benchmark
"""

import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from . import config
from .augment import get_train_transforms, get_val_transforms
from .dataset import create_single_input_data_loaders
from .losses_pahvit import create_pahvit_criterion
from .model_pahvit import create_pahvit_model, load_pahvit_model, save_pahvit_model
from .train_pahvit import train_one_epoch_pahvit, validate_pahvit_model


def run_quick_benchmark(
    epochs_head: int = 2,
    epochs_finetune: int = 3,
    device: str | None = None,
) -> dict:
    """Run a quick benchmark with reduced epochs.

    Args:
        epochs_head: Stage A epochs (default: 2)
        epochs_finetune: Stage B epochs (default: 3)
        device: Device to use

    Returns:
        dict with benchmark results
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 60}")
    print("PAH-ViT Quick Benchmark")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Stage A epochs: {epochs_head}")
    print(f"Stage B epochs: {epochs_finetune}")

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    class_names = config.CLASS_NAMES_THREE_CLASS
    class_weights = config.CLASS_WEIGHTS_THREE_CLASS

    # Create data loaders
    print("\n[1/4] Creating data loaders...")
    train_loader, val_loader, full_dataset = create_single_input_data_loaders(
        data_map=config.THREE_CLASS_DATA_MAP,
        data_dir=config.DATA_DIR,
        transform_train=get_train_transforms(),
        transform_val=get_val_transforms(),
        batch_size=config.PAH_VIT_BATCH_SIZE,
        use_weighted_sampler=config.USE_WEIGHTED_SAMPLER,
    )

    train_size = int(len(full_dataset) * config.TRAIN_VAL_SPLIT)
    val_size = len(full_dataset) - train_size
    print(f"  Dataset: {len(full_dataset)} images ({train_size}/{val_size})")

    # Create model
    print("\n[2/4] Creating PAH-ViT model...")
    model = create_pahvit_model(
        model_name=config.PAH_VIT_MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print("\n[3/4] Testing forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        logits, anomaly_scores = model(dummy_input)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Anomaly scores shape: {anomaly_scores.shape}")

    # Create criterion
    criterion = create_pahvit_criterion(
        lambda_contrastive=config.PAH_VIT_LAMBDA_CONTRASTIVE,
        class_weights=class_weights,
        focal_gamma=config.PAH_VIT_FOCAL_GAMMA,
    )

    # Test loss computation
    dummy_labels = torch.tensor([0, 1]).to(device)
    total_loss, ce_loss, contrastive_loss = criterion(logits, anomaly_scores, dummy_labels)
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  CE loss: {ce_loss.item():.4f}")
    print(f"  Contrastive loss: {contrastive_loss.item():.4f}")

    # Quick training
    print(f"\n[4/4] Running quick training ({epochs_head}+{epochs_finetune} epochs)...")
    screen_photo_class_idx = class_names.index("screen_photo") if "screen_photo" in class_names else 2

    best_metric = 0.0
    start_time = time.time()

    # Stage A
    print(f"\n  Stage A ({epochs_head} epochs):")
    optimizer = optim.AdamW(
        list(model.token_mixer.parameters())
        + list(model.patch_branch.parameters())
        + list(model.fusion.parameters())
        + list(model.classifier.parameters()),
        lr=config.PAH_VIT_LEARNING_RATE,
        weight_decay=config.PAH_VIT_WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_head)

    for epoch in range(epochs_head):
        epoch_start = time.time()
        train_loss, train_acc, ce_loss, _ = train_one_epoch_pahvit(model, train_loader, criterion, optimizer, device)

        val_metrics = validate_pahvit_model(model, val_loader, device, class_names)
        val_acc = val_metrics["accuracy"]
        sp_f1 = val_metrics["f1_per_class"][screen_photo_class_idx]
        macro_f1 = val_metrics["f1_macro"]
        current_metric = (
            config.BEST_METRIC_F1_WEIGHT * sp_f1
            + config.BEST_METRIC_ACCURACY_WEIGHT * val_acc
            + config.BEST_METRIC_MACRO_F1_WEIGHT * macro_f1
        )

        scheduler.step()

        if current_metric > best_metric:
            best_metric = current_metric
            save_pahvit_model(
                model,
                str(config.CHECKPOINT_DIR / "pahvit_benchmark_best.pth"),
                epoch=epoch,
                best_val_acc=val_acc,
                best_metric=best_metric,
            )

        elapsed = time.time() - epoch_start
        print(
            f"    Epoch {epoch + 1}/{epochs_head} - "
            f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
            f"Val Acc: {val_acc:.4f} - SP F1: {sp_f1:.4f} - "
            f"Metric: {current_metric:.4f} - Time: {elapsed:.1f}s"
        )

    # Stage B
    print(f"\n  Stage B ({epochs_finetune} epochs):")
    model.unfreeze_backbone(num_layers=4)

    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": config.PAH_VIT_LEARNING_RATE * 0.1},
            {"params": model.token_mixer.parameters(), "lr": config.PAH_VIT_LEARNING_RATE * 0.1},
            {"params": model.patch_branch.parameters(), "lr": config.PAH_VIT_LEARNING_RATE},
            {"params": model.fusion.parameters(), "lr": config.PAH_VIT_LEARNING_RATE},
            {"params": model.classifier.parameters(), "lr": config.PAH_VIT_LEARNING_RATE},
        ],
        weight_decay=config.PAH_VIT_WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_finetune)

    for epoch in range(epochs_finetune):
        epoch_start = time.time()
        train_loss, train_acc, ce_loss, _ = train_one_epoch_pahvit(model, train_loader, criterion, optimizer, device)

        val_metrics = validate_pahvit_model(model, val_loader, device, class_names)
        val_acc = val_metrics["accuracy"]
        sp_f1 = val_metrics["f1_per_class"][screen_photo_class_idx]
        macro_f1 = val_metrics["f1_macro"]
        current_metric = (
            config.BEST_METRIC_F1_WEIGHT * sp_f1
            + config.BEST_METRIC_ACCURACY_WEIGHT * val_acc
            + config.BEST_METRIC_MACRO_F1_WEIGHT * macro_f1
        )

        scheduler.step()

        if current_metric > best_metric:
            best_metric = current_metric
            save_pahvit_model(
                model,
                str(config.CHECKPOINT_DIR / "pahvit_benchmark_best.pth"),
                epoch=epochs_head + epoch,
                best_val_acc=val_acc,
                best_metric=best_metric,
            )

        elapsed = time.time() - epoch_start
        print(
            f"    Epoch {epoch + 1}/{epochs_finetune} - "
            f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
            f"Val Acc: {val_acc:.4f} - SP F1: {sp_f1:.4f} - "
            f"Metric: {current_metric:.4f} - Time: {elapsed:.1f}s"
        )

    total_time = time.time() - start_time

    # Load best model and evaluate
    best_model = load_pahvit_model(
        str(config.CHECKPOINT_DIR / "pahvit_benchmark_best.pth"),
        device=device,
    )
    best_model = best_model.to(device)
    final_metrics = validate_pahvit_model(best_model, val_loader, device, class_names)

    print(f"\n{'=' * 60}")
    print("Benchmark Results")
    print(f"{'=' * 60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Best metric: {best_metric:.4f}")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"SP F1: {final_metrics['f1_per_class'][screen_photo_class_idx]:.4f}")
    print(f"SP Precision: {final_metrics['precision_per_class'][screen_photo_class_idx]:.4f}")
    print(f"SP Recall: {final_metrics['recall_per_class'][screen_photo_class_idx]:.4f}")
    print(f"Macro F1: {final_metrics['f1_macro']:.4f}")
    print("\n[OK] Benchmark completed successfully!")
    print("  Model is ready for full 5-fold validation.")
    print("  Run: uv run python -m trainer.validate_pahvit")

    return {
        "accuracy": final_metrics["accuracy"],
        "sp_f1": final_metrics["f1_per_class"][screen_photo_class_idx],
        "sp_precision": final_metrics["precision_per_class"][screen_photo_class_idx],
        "sp_recall": final_metrics["recall_per_class"][screen_photo_class_idx],
        "macro_f1": final_metrics["f1_macro"],
        "best_metric": best_metric,
        "total_time": total_time,
    }


def main():
    """Main entry point for quick benchmark."""
    run_quick_benchmark(epochs_head=2, epochs_finetune=3)


if __name__ == "__main__":
    main()
