"""5-run random validation for PAH-ViT model.

Runs 5 independent training runs with different random seeds (80/20 split),
computes average metrics and standard deviation, and compares with the
old EfficientNet+FFT+DWT architecture baseline.

Usage:
    uv run python -m trainer.validate_pahvit
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from . import config
from .augment import get_train_transforms, get_val_transforms
from .dataset import create_single_input_data_loaders
from .losses_pahvit import create_pahvit_criterion
from .model_pahvit import create_pahvit_model, load_pahvit_model, save_pahvit_model
from .train_pahvit import train_one_epoch_pahvit, validate_pahvit_model

# Old architecture baseline (from previous training)
OLD_BASELINE = {
    "accuracy": 0.8946,
    "screen_photo_f1": 0.7630,
    "screen_photo_precision": 0.7097,
    "screen_photo_recall": 0.8250,
    "screen_photo_f1_per_class": [0.9254, 0.9121, 0.7630],
    "macro_f1": 0.8668,
}


def train_and_evaluate_single_run(
    seed: int,
    data_map: dict[str, list[str]],
    class_names: list[str],
    class_weights: list[float],
    data_dir: Path,
    device: str,
    epochs_head: int = config.PAH_VIT_EPOCHS_HEAD,
    epochs_finetune: int = config.PAH_VIT_EPOCHS_FINETUNE,
    batch_size: int = config.PAH_VIT_BATCH_SIZE,
    learning_rate: float = config.PAH_VIT_LEARNING_RATE,
    lambda_contrastive: float = config.PAH_VIT_LAMBDA_CONTRASTIVE,
) -> dict:
    """Train and evaluate PAH-ViT model for a single run.

    Args:
        seed: Random seed for this run
        data_map: Data mapping
        class_names: Class names
        class_weights: Class weights
        data_dir: Data directory
        device: Device to use
        epochs_head: Stage A epochs
        epochs_finetune: Stage B epochs
        batch_size: Batch size
        learning_rate: Learning rate
        lambda_contrastive: Contrastive loss weight

    Returns:
        dict with metrics for this run
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Override random seed in config for this run
    original_seed = config.RANDOM_SEED
    config.RANDOM_SEED = seed

    try:
        # Create data loaders
        train_loader, val_loader, full_dataset = create_single_input_data_loaders(
            data_map=data_map,
            data_dir=data_dir,
            transform_train=get_train_transforms(),
            transform_val=get_val_transforms(),
            batch_size=batch_size,
            use_weighted_sampler=config.USE_WEIGHTED_SAMPLER,
        )

        train_size = int(len(full_dataset) * config.TRAIN_VAL_SPLIT)
        val_size = len(full_dataset) - train_size

        # Create model
        model = create_pahvit_model(
            model_name=config.PAH_VIT_MODEL_NAME,
            num_classes=config.NUM_CLASSES,
            pretrained=True,
            freeze_backbone=True,
        )
        model = model.to(device)

        # Create criterion
        criterion = create_pahvit_criterion(
            lambda_contrastive=lambda_contrastive,
            class_weights=class_weights,
            focal_gamma=config.PAH_VIT_FOCAL_GAMMA,
        )

        screen_photo_class_idx = (
            class_names.index("screen_photo") if "screen_photo" in class_names else 2
        )

        best_metric = 0.0
        best_val_acc = 0.0

        # Stage A: Train head
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
            train_loss, train_acc, _, _ = train_one_epoch_pahvit(
                model, train_loader, criterion, optimizer, device
            )

            val_metrics = validate_pahvit_model(model, val_loader, device, class_names)
            val_acc = val_metrics["accuracy"]

            screen_photo_f1 = val_metrics["f1_per_class"][screen_photo_class_idx]
            macro_f1 = val_metrics["f1_macro"]
            current_metric = (
                config.BEST_METRIC_F1_WEIGHT * screen_photo_f1
                + config.BEST_METRIC_ACCURACY_WEIGHT * val_acc
                + config.BEST_METRIC_MACRO_F1_WEIGHT * macro_f1
            )

            scheduler.step()

            if current_metric > best_metric:
                best_metric = current_metric
                best_val_acc = val_acc
                save_pahvit_model(
                    model,
                    str(config.CHECKPOINT_DIR / f"pahvit_run_{seed}_best.pth"),
                    epoch=epoch,
                    best_val_acc=best_val_acc,
                    best_metric=best_metric,
                )

        # Stage B: Fine-tune
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
            train_loss, train_acc, _, _ = train_one_epoch_pahvit(
                model, train_loader, criterion, optimizer, device
            )

            val_metrics = validate_pahvit_model(model, val_loader, device, class_names)
            val_acc = val_metrics["accuracy"]

            screen_photo_f1 = val_metrics["f1_per_class"][screen_photo_class_idx]
            macro_f1 = val_metrics["f1_macro"]
            current_metric = (
                config.BEST_METRIC_F1_WEIGHT * screen_photo_f1
                + config.BEST_METRIC_ACCURACY_WEIGHT * val_acc
                + config.BEST_METRIC_MACRO_F1_WEIGHT * macro_f1
            )

            scheduler.step()

            if current_metric > best_metric:
                best_metric = current_metric
                best_val_acc = val_acc
                save_pahvit_model(
                    model,
                    str(config.CHECKPOINT_DIR / f"pahvit_run_{seed}_best.pth"),
                    epoch=epochs_head + epoch,
                    best_val_acc=best_val_acc,
                    best_metric=best_metric,
                )

        # Load best model and evaluate
        best_model = load_pahvit_model(
            str(config.CHECKPOINT_DIR / f"pahvit_run_{seed}_best.pth"),
            device=device,
        )
        best_model = best_model.to(device)

        final_metrics = validate_pahvit_model(
            best_model, val_loader, device, class_names
        )

        return {
            "seed": seed,
            "accuracy": final_metrics["accuracy"],
            "screen_photo_f1": final_metrics["f1_per_class"][screen_photo_class_idx],
            "screen_photo_precision": final_metrics["precision_per_class"][
                screen_photo_class_idx
            ],
            "screen_photo_recall": final_metrics["recall_per_class"][
                screen_photo_class_idx
            ],
            "macro_f1": final_metrics["f1_macro"],
            "f1_per_class": final_metrics["f1_per_class"].tolist(),
            "best_metric": best_metric,
            "train_size": train_size,
            "val_size": val_size,
        }

    finally:
        config.RANDOM_SEED = original_seed


def run_5fold_validation(
    num_runs: int = 5,
    device: str | None = None,
) -> dict:
    """Run 5-fold random validation and compare with old architecture.

    Args:
        num_runs: Number of validation runs
        device: Device to use

    Returns:
        dict with aggregated results and comparison
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 70}")
    print("PAH-ViT 5-Fold Random Validation")
    print(f"{'=' * 70}")
    print(f"Device: {device}")
    print(f"Runs: {num_runs}")
    print(f"Stage A epochs: {config.PAH_VIT_EPOCHS_HEAD}")
    print(f"Stage B epochs: {config.PAH_VIT_EPOCHS_FINETUNE}")
    print(f"Lambda contrastive: {config.PAH_VIT_LAMBDA_CONTRASTIVE}")

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    seeds = [42 + i for i in range(num_runs)]

    for i, seed in enumerate(seeds):
        print(f"\n{'=' * 50}")
        print(f"Run {i + 1}/{num_runs} (seed={seed})")
        print(f"{'=' * 50}")

        start_time = time.time()

        run_result = train_and_evaluate_single_run(
            seed=seed,
            data_map=config.THREE_CLASS_DATA_MAP,
            class_names=config.CLASS_NAMES_THREE_CLASS,
            class_weights=config.CLASS_WEIGHTS_THREE_CLASS,
            data_dir=config.DATA_DIR,
            device=device,
        )

        elapsed = time.time() - start_time
        run_result["elapsed_seconds"] = elapsed
        results.append(run_result)

        print(f"\n  Run {i + 1} Results:")
        print(f"    Accuracy: {run_result['accuracy']:.4f}")
        print(f"    SP F1: {run_result['screen_photo_f1']:.4f}")
        print(f"    SP Precision: {run_result['screen_photo_precision']:.4f}")
        print(f"    SP Recall: {run_result['screen_photo_recall']:.4f}")
        print(f"    Macro F1: {run_result['macro_f1']:.4f}")
        print(f"    Time: {elapsed:.1f}s")

    # Aggregate results
    metrics = ["accuracy", "screen_photo_f1", "screen_photo_precision", "screen_photo_recall", "macro_f1"]
    aggregated = {}

    for metric in metrics:
        values = [r[metric] for r in results]
        aggregated[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": [float(v) for v in values],
        }

    # Compare with old baseline
    comparison = {}
    for metric in metrics:
        new_mean = aggregated[metric]["mean"]
        old_value = OLD_BASELINE.get(metric, 0.0)
        improvement = new_mean - old_value
        comparison[metric] = {
            "new_mean": new_mean,
            "old_value": old_value,
            "improvement": improvement,
            "improved": improvement > 0,
        }

    # Print summary
    print(f"\n{'=' * 70}")
    print("5-Fold Validation Summary")
    print(f"{'=' * 70}")

    print(f"\n{'Metric':<25} {'New (mean±std)':<20} {'Old':<12} {'Diff':<12} {'Status'}")
    print("-" * 70)

    all_improved = True
    for metric in metrics:
        c = comparison[metric]
        status = "[+] Better" if c["improved"] else "[-] Worse"
        if not c["improved"]:
            all_improved = False
        print(
            f"{metric:<25} "
            f"{c['new_mean']:.4f}±{aggregated[metric]['std']:.4f}  "
            f"{c['old_value']:.4f}      "
            f"{c['improvement']:+.4f}     "
            f"{status}"
        )

    print(f"\n{'=' * 70}")
    if all_improved:
        print("[+] PAH-ViT outperforms old architecture on ALL metrics!")
        print("  Recommendation: Keep PAH-ViT architecture.")
    else:
        print("[-] PAH-ViT does NOT outperform old architecture on all metrics.")
        print("  Recommendation: Rollback to old architecture.")

    # Save results
    output = {
        "pahvit_results": aggregated,
        "old_baseline": OLD_BASELINE,
        "comparison": comparison,
        "all_improved": all_improved,
        "per_run_results": results,
        "config": {
            "num_runs": num_runs,
            "epochs_head": config.PAH_VIT_EPOCHS_HEAD,
            "epochs_finetune": config.PAH_VIT_EPOCHS_FINETUNE,
            "lambda_contrastive": config.PAH_VIT_LAMBDA_CONTRASTIVE,
            "model_name": config.PAH_VIT_MODEL_NAME,
            "batch_size": config.PAH_VIT_BATCH_SIZE,
            "learning_rate": config.PAH_VIT_LEARNING_RATE,
        },
    }

    output_path = config.LOG_DIR / "pahvit_5fold_validation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return output


def main():
    """Main entry point for 5-fold validation."""
    results = run_5fold_validation(num_runs=5)

    print(f"\n{'=' * 70}")
    print("Final Decision:")
    if results["all_improved"]:
        print("  → PAH-ViT is BETTER. Proceed with architecture upgrade.")
    else:
        print("  → Old architecture is BETTER or comparable. Rollback.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
