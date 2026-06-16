"""Training pipeline for screen detector models.

支持三种模型架构：
1. DeiTScreenDetector - 纯 DeiT (RGB only)
2. FFTDeiT - 双流 DeiT (RGB + FFT)
3. DWTFFTDeiT - 三流 DeiT (RGB + FFT + DWT)

技术特点：
1. 使用预训练权重 (ImageNet-1k)
2. DeiT-Small 模型 (22M 参数)
3. 两阶段迁移学习
4. Mixup + CutMix 数据增强
5. Stochastic Depth (drop_path_rate=0.1)
6. Label Smoothing (0.1)
7. Layer-wise Learning Rate Decay (LLRD)
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import create_dataloaders
from .model import (
    DeiTScreenDetector,
    DWTFFTDeiT,
    FFTDeiT,
    create_deit_model,
    create_dwt_fft_deit_model,
    create_fft_deit_model,
    save_deit_model,
)
from .transforms import MixUpCutMixWrapper
from .validate import compute_metrics

# Default config
DEFAULT_CONFIG = {
    "model_type": "deit",  # "deit", "fft_deit", "dwt_fft_deit"
    "model_name": "deit_small_patch16_224",
    "num_classes": 3,
    "image_size": 224,
    "batch_size": 32,
    "num_workers": 4,
    "stage1_epochs": 10,  # Head only
    "stage2_epochs": 90,  # Fine-tune
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "val_split": 0.2,
    "seed": 42,
    "drop_path_rate": 0.1,
    "label_smoothing": 0.1,
    "mixup_prob": 0.5,
    "use_mixup": True,
}


def create_model_from_config(
    config: dict,
) -> DeiTScreenDetector | FFTDeiT | DWTFFTDeiT:
    """Create model based on config.

    Args:
        config: Training configuration

    Returns:
        Model instance
    """
    model_type = config.get("model_type", "deit")
    model_name = config.get("model_name", "deit_small_patch16_224")
    num_classes = config.get("num_classes", 3)
    drop_path_rate = config.get("drop_path_rate", 0.1)

    if model_type == "deit":
        return create_deit_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
            drop_path_rate=drop_path_rate,
        )
    if model_type == "fft_deit":
        return create_fft_deit_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
            drop_path_rate=drop_path_rate,
        )
    if model_type == "dwt_fft_deit":
        return create_dwt_fft_deit_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
            drop_path_rate=drop_path_rate,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def get_layerwise_lr_params(
    model: nn.Module, lr: float, decay: float = 0.85
) -> list[dict]:
    """Get layer-wise learning rate parameters for LLRD.

    Args:
        model: Model
        lr: Base learning rate
        decay: Learning rate decay factor

    Returns:
        List of parameter groups with different learning rates
    """
    param_groups = []

    # Get the DeiT model (handle different model types)
    if isinstance(model, DeiTScreenDetector):
        deit_model = model.model
    elif isinstance(model, FFTDeiT | DWTFFTDeiT):
        deit_model = model.rgb_stream
    else:
        # Fallback: all parameters with same learning rate
        param_groups.append(
            {"params": list(model.parameters()), "lr": lr, "name": "all"}
        )
        return param_groups

    # Get model layers (for DeiT-Small: 12 transformer blocks)
    if hasattr(deit_model, "blocks"):
        blocks = deit_model.blocks
        num_layers = len(blocks)

        # Embedding layer (lowest learning rate)
        embed_params = list(deit_model.patch_embed.parameters())
        if hasattr(deit_model, "cls_token"):
            embed_params.append(deit_model.cls_token)
        if hasattr(deit_model, "pos_embed"):
            embed_params.append(deit_model.pos_embed)
        param_groups.append(
            {
                "params": embed_params,
                "lr": lr * (decay**num_layers),
                "name": "embeddings",
            }
        )

        # Transformer blocks (increasing learning rate)
        for i, block in enumerate(blocks):
            block_lr = lr * (decay ** (num_layers - i - 1))
            param_groups.append(
                {
                    "params": list(block.parameters()),
                    "lr": block_lr,
                    "name": f"block_{i}",
                }
            )

        # Head (highest learning rate)
        if hasattr(deit_model, "head"):
            head_params = list(deit_model.head.parameters())
            param_groups.append({"params": head_params, "lr": lr, "name": "head"})
    else:
        # Fallback: all parameters with same learning rate
        param_groups.append(
            {"params": list(model.parameters()), "lr": lr, "name": "all"}
        )

    # Add FFT/DWT stream parameters if applicable
    if isinstance(model, FFTDeiT):
        param_groups.append(
            {
                "params": list(model.fft_stream.parameters()),
                "lr": lr * 0.5,
                "name": "fft_stream",
            }
        )
        param_groups.append(
            {
                "params": list(model.classifier.parameters()),
                "lr": lr,
                "name": "classifier",
            }
        )
    elif isinstance(model, DWTFFTDeiT):
        param_groups.append(
            {
                "params": list(model.fft_stream.parameters()),
                "lr": lr * 0.5,
                "name": "fft_stream",
            }
        )
        param_groups.append(
            {
                "params": list(model.dwt_stream.parameters()),
                "lr": lr * 0.5,
                "name": "dwt_stream",
            }
        )
        param_groups.append(
            {
                "params": list(model.classifier.parameters()),
                "lr": lr,
                "name": "classifier",
            }
        )

    return param_groups


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,  # noqa: ARG001
    model_type: str = "deit",
    mixup_cutmix: MixUpCutMixWrapper | None = None,
) -> dict:
    """Train model for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        model_type: Model type for input handling
        mixup_cutmix: Mixup/CutMix wrapper (optional)

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        if len(batch) == 2:
            # RGB only mode
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # Save original labels for accuracy
            labels_orig = labels.clone()

            # Apply Mixup/CutMix if enabled
            if mixup_cutmix is not None:
                images, labels_onehot = mixup_cutmix(images, labels)
            else:
                labels_onehot = labels

            # Forward pass
            outputs = model(images)
        elif len(batch) == 3:
            # RGB + FFT mode
            images, fft_input, labels = batch
            images = images.to(device)
            fft_input = fft_input.to(device)
            labels = labels.to(device)

            # Save original labels for accuracy
            labels_orig = labels.clone()

            # Apply Mixup/CutMix if enabled
            if mixup_cutmix is not None:
                images, labels_onehot = mixup_cutmix(images, labels)
                # Also mix FFT input
                fft_input = fft_input.to(device)
            else:
                labels_onehot = labels

            # Forward pass
            if model_type in ("fft_deit", "dwt_fft_deit"):
                outputs = model(images, fft_input)
            else:
                outputs = model(images)
        else:
            raise ValueError(f"Unexpected batch format: {len(batch)} elements")

        # Compute loss
        if labels_onehot.dim() > 1:
            # One-hot labels (from Mixup/CutMix)
            loss = criterion(outputs, labels_onehot)
        else:
            # Integer labels
            loss = criterion(outputs, labels_onehot)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels_orig.size(0)
        correct += predicted.eq(labels_orig).sum().item()

    return {
        "loss": total_loss / len(train_loader),
        "accuracy": correct / total,
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = "deit",
) -> dict:
    """Validate model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        model_type: Model type for input handling

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
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

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / len(val_loader)

    return metrics


def train(
    config: dict | None = None,
    data_dir: str | Path = "data/input",
    output_dir: str | Path = "trainer_vit/checkpoints",
) -> dict:
    """Main training function.

    Args:
        config: Training configuration (uses DEFAULT_CONFIG if None)
        data_dir: Data directory
        output_dir: Output directory for checkpoints

    Returns:
        Dictionary with training results
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Setup
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Get config values
    model_type = config.get("model_type", "deit")
    input_mode = config.get("input_mode", model_type)
    if input_mode == "deit":
        input_mode = "rgb"
    elif input_mode in ("fft_deit", "dwt_fft_deit"):
        input_mode = "fft"

    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    image_size = config.get("image_size", 224)
    seed = config.get("seed", 42)
    val_split = config.get("val_split", 0.2)
    stage1_epochs = config.get("stage1_epochs", 10)
    stage2_epochs = config.get("stage2_epochs", 90)
    learning_rate = config.get("learning_rate", 1e-3)
    weight_decay = config.get("weight_decay", 0.01)
    use_mixup = config.get("use_mixup", True)
    mixup_prob = config.get("mixup_prob", 0.5)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        image_size=image_size,
        seed=seed,
        input_mode=input_mode,
    )

    # Create model
    logger.info(f"Creating model: {model_type}")
    model = create_model_from_config(config)
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Mixup/CutMix
    mixup_cutmix = None
    if use_mixup:
        mixup_cutmix = MixUpCutMixWrapper(
            num_classes=config.get("num_classes", 3),
            prob=mixup_prob,
        )

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_recall": [],
    }

    best_val_acc = 0.0

    # Stage 1: Train head only
    logger.info(f"Stage 1: Training head only ({stage1_epochs} epochs)")
    model.freeze_backbone()

    # Get parameters for stage 1
    trainable_params = []
    for _name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)

    optimizer = AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=stage1_epochs, eta_min=1e-6)

    for epoch in range(stage1_epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            model_type,
            mixup_cutmix,
        )
        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            model_type,
        )
        scheduler.step()

        logger.info(
            f"Epoch {epoch + 1}/{stage1_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f} - "
            f"Train Acc: {train_metrics['accuracy']:.4f} - "
            f"Val Loss: {val_metrics['loss']:.4f} - "
            f"Val Acc: {val_metrics['accuracy']:.4f} - "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_recall"].append(val_metrics.get("recall", 0))

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_deit_model(
                model,
                str(output_dir / "best.pth"),
                epoch,
                model_type,
            )

    # Stage 2: Fine-tune all layers
    logger.info(f"Stage 2: Fine-tuning all layers ({stage2_epochs} epochs)")
    model.unfreeze_backbone()

    # Get layer-wise learning rate parameters
    param_groups = get_layerwise_lr_params(model, learning_rate, decay=0.85)

    optimizer = AdamW(
        param_groups,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs, eta_min=1e-6)

    for epoch in range(stage2_epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            model_type,
            mixup_cutmix,
        )
        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            model_type,
        )
        scheduler.step()

        logger.info(
            f"Epoch {epoch + 1}/{stage2_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f} - "
            f"Train Acc: {train_metrics['accuracy']:.4f} - "
            f"Val Loss: {val_metrics['loss']:.4f} - "
            f"Val Acc: {val_metrics['accuracy']:.4f} - "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_recall"].append(val_metrics.get("recall", 0))

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_deit_model(
                model,
                str(output_dir / "best.pth"),
                stage1_epochs + epoch,
                model_type,
            )

    # Save training history
    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    logger.info(f"Training completed. Best val acc: {best_val_acc:.4f}")
    logger.info(f"Model saved to: {output_dir / 'best.pth'}")

    return {
        "best_val_acc": best_val_acc,
        "history": history,
        "output_dir": str(output_dir),
    }


def main() -> None:
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train screen detector model")
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
        default="trainer_vit/checkpoints",
        help="Path to output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["model_type"] = args.model_type
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.learning_rate
    config["stage1_epochs"] = args.epochs // 10
    config["stage2_epochs"] = args.epochs - config["stage1_epochs"]

    train(
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
