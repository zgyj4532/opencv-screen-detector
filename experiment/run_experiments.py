"""Experiment runner for screen detector ablation study.

Runs 5 experiments with multiple trials each:
- Exp0: CNN Only (EfficientNet-B0) - control group
- Exp1: CNN+FFT (EfficientNet-B0 + FFT Branch)
- Exp2: DeiT (deit_small_patch16_224)
- Exp3: FFT+DeiT (dual-stream)
- Exp4: DWT+FFT+DeiT (triple-stream)

Outputs to data/output/:
- metrics/{exp_name}_metrics.json
- recall_ranking.md
- confusion_matrices/{exp_name}_confusion.json
- logs/{exp_name}_log.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import cast

# Add project root to path before imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader

from trainer.model import create_model as create_cnn_model
from trainer_vit.src.dataset import LABEL_NAMES, create_dataloaders
from trainer_vit.src.model import (
    create_deit_model,
    create_dwt_fft_deit_model,
    create_fft_deit_model,
)

# Experiment configurations
EXPERIMENTS = {
    "exp0_cnn_only": {
        "name": "Exp0: CNN Only",
        "description": "EfficientNet-B0 only (no FFT/DWT)",
        "input_mode": "rgb",
        "model_type": "cnn",
    },
    "exp1_cnn_fft": {
        "name": "Exp1: CNN+FFT",
        "description": "EfficientNet-B0 + FFT Branch",
        "input_mode": "fft",
        "model_type": "cnn",
    },
    "exp2_deit": {
        "name": "Exp2: DeiT",
        "description": "DeiT-Small only",
        "input_mode": "rgb",
        "model_type": "deit",
    },
    "exp3_fft_deit": {
        "name": "Exp3: FFT+DeiT",
        "description": "Dual-stream DeiT (RGB + FFT)",
        "input_mode": "fft",
        "model_type": "fft_deit",
    },
    "exp4_dwt_fft_deit": {
        "name": "Exp4: DWT+FFT+DeiT",
        "description": "Triple-stream DeiT (RGB + FFT + DWT)",
        "input_mode": "fft",
        "model_type": "dwt_fft_deit",
    },
}

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
METRICS_DIR = OUTPUT_DIR / "metrics"
CONFUSION_DIR = OUTPUT_DIR / "confusion_matrices"
LOGS_DIR = OUTPUT_DIR / "logs"


def setup_logger(log_file: Path) -> None:
    """Setup logger to write to file and stderr."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        str(log_file),
        level="DEBUG",
        rotation="10 MB",
        encoding="utf-8",
    )


def create_experiment_model(
    model_type: str,
    num_classes: int = 3,
    pretrained: bool = True,
) -> nn.Module:
    """Create model for experiment.

    Args:
        model_type: Model type ("cnn", "deit", "fft_deit", "dwt_fft_deit")
        num_classes: Number of classes
        pretrained: Use pretrained weights

    Returns:
        Model instance
    """
    if model_type == "cnn":
        return create_cnn_model(
            num_classes=num_classes,
            pretrained=pretrained,
        )
    if model_type == "deit":
        return create_deit_model(
            num_classes=num_classes,
            pretrained=pretrained,
        )
    if model_type == "fft_deit":
        return create_fft_deit_model(
            num_classes=num_classes,
            pretrained=pretrained,
        )
    if model_type == "dwt_fft_deit":
        return create_dwt_fft_deit_model(
            num_classes=num_classes,
            pretrained=pretrained,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_type: str,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        model_type: Model type for input handling

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        if len(batch) == 2:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            if model_type == "cnn":
                fft_dummy = torch.zeros_like(images[:, :1, :, :])
                outputs = model(images, fft_dummy)
            else:
                outputs = model(images)
        elif len(batch) == 3:
            images, fft_input, labels = batch
            images = images.to(device)
            fft_input = fft_input.to(device)
            labels = labels.to(device)

            if model_type in ("fft_deit", "dwt_fft_deit", "cnn"):
                outputs = model(images, fft_input)
            else:
                outputs = model(images)
        else:
            raise ValueError(f"Unexpected batch format: {len(batch)} elements")

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return {
        "loss": total_loss / len(train_loader),
        "accuracy": correct / total,
    }


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    model_type: str,
) -> dict:
    """Evaluate model on validation set.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to evaluate on
        model_type: Model type for input handling

    Returns:
        Dictionary with all metrics
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

                if model_type == "cnn":
                    fft_dummy = torch.zeros_like(images[:, :1, :, :])
                    outputs = model(images, fft_dummy)
                else:
                    outputs = model(images)
            elif len(batch) == 3:
                images, fft_input, labels = batch
                images = images.to(device)
                fft_input = fft_input.to(device)
                labels = labels.to(device)

                if model_type in (
                    "fft_deit",
                    "dwt_fft_deit",
                    "cnn",
                ):
                    outputs = model(images, fft_input)
                else:
                    outputs = model(images)
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average="macro")
    recall_macro = recall_score(all_labels, all_preds, average="macro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    precision_per_class = cast(np.ndarray, precision_score(all_labels, all_preds, average=None))
    recall_per_class = cast(np.ndarray, recall_score(all_labels, all_preds, average=None))
    f1_per_class = cast(np.ndarray, f1_score(all_labels, all_preds, average=None))

    cm = confusion_matrix(all_labels, all_preds)

    classes = {}
    for i, name in enumerate(LABEL_NAMES):
        classes[name] = {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
        }

    confusion = {}
    for i, name in enumerate(LABEL_NAMES):
        confusion[name] = {}
        for j, name2 in enumerate(LABEL_NAMES):
            confusion[name][name2] = int(cm[i][j])

    return {
        "accuracy": float(accuracy),
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1_score": float(f1_macro),
        "classes": classes,
        "confusion_matrix": confusion,
        "screen_photo_recall": float(recall_per_class[2]),
        "screen_photo_precision": float(precision_per_class[2]),
        "screen_photo_f1": float(f1_per_class[2]),
        "total_samples": len(all_labels),
    }


def run_single_trial(
    exp_config: dict,
    data_dir: Path,
    output_dir: Path,
    trial_idx: int,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: torch.device | None = None,
    early_stopping_patience: int = 2,
) -> dict:
    """Run a single trial of an experiment with early stopping.

    Args:
        exp_config: Experiment configuration
        data_dir: Data directory
        trial_idx: Trial index
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
        early_stopping_patience: Stop if no improvement for N checkpoints

    Returns:
        Dictionary with trial results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = exp_config["name"]
    model_type = exp_config["model_type"]
    input_mode = exp_config["input_mode"]

    logger.info(f"Trial {trial_idx + 1} - {exp_name}")
    logger.info(f"Model: {model_type}, Input: {input_mode}, Device: {device}")

    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
        input_mode=input_mode,
    )

    model = create_experiment_model(model_type, num_classes=3, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_acc = 0.0
    best_metrics = None
    train_log = []
    checkpoint_best_acc = 0.0  # Best acc at last checkpoint
    no_improvement_count = 0  # Count of checkpoints without improvement

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, model_type)

        val_metrics = evaluate(model, val_loader, device, model_type)

        scheduler.step()

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["f1_score"],
            "screen_photo_recall": val_metrics["screen_photo_recall"],
        }
        train_log.append(log_entry)

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_metrics = val_metrics

            # Save best model checkpoint
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save(
                {
                    "model_type": model_type,
                    "model_state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Loss: {train_metrics['loss']:.4f} - "
                f"Acc: {val_metrics['accuracy']:.4f} - "
                f"F1: {val_metrics['f1_score']:.4f} - "
                f"SP_Recall: {val_metrics['screen_photo_recall']:.4f}"
            )
            logger.info(f"Checkpoint - Best Acc: {best_acc:.4f} (prev: {checkpoint_best_acc:.4f})")

            # Early stopping check
            if best_acc <= checkpoint_best_acc:
                no_improvement_count += 1
                logger.info(f"No improvement for {no_improvement_count} checkpoint(s)")
                if no_improvement_count >= early_stopping_patience:
                    logger.info("Early stopping triggered - no improvement for 2 consecutive checkpoints")
                    break
            else:
                no_improvement_count = 0

            checkpoint_best_acc = best_acc

    return {
        "trial": trial_idx + 1,
        "best_val_acc": best_acc,
        "metrics": best_metrics,
        "train_log": train_log,
        "epochs_trained": len(train_log),
    }


def run_experiment(
    exp_name: str,
    data_dir: Path,
    num_trials: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    early_stopping_patience: int = 2,
) -> dict:
    """Run an experiment with multiple trials.

    Args:
        exp_name: Experiment name key
        data_dir: Data directory
        num_trials: Number of trials
        epochs: Number of training epochs per trial
        batch_size: Batch size
        learning_rate: Learning rate

    Returns:
        Dictionary with experiment results
    """
    exp_config = EXPERIMENTS[exp_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running {exp_config['name']}")
    logger.info(f"Description: {exp_config['description']}")
    logger.info(f"Trials: {num_trials}, Epochs: {epochs}, Device: {device}")

    trials = []
    for trial_idx in range(num_trials):
        trial_result = run_single_trial(
            exp_config=exp_config,
            data_dir=data_dir,
            output_dir=OUTPUT_DIR / exp_name,
            trial_idx=trial_idx,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            early_stopping_patience=early_stopping_patience,
        )
        trials.append(trial_result)

    metrics_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "screen_photo_recall",
        "screen_photo_precision",
        "screen_photo_f1",
    ]

    aggregated = {}
    for key in metrics_keys:
        values = [t["metrics"][key] for t in trials]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    # Add epochs trained
    epochs_trained = [t.get("epochs_trained", epochs) for t in trials]
    aggregated["epochs_trained_mean"] = float(np.mean(epochs_trained))
    aggregated["epochs_trained_std"] = float(np.std(epochs_trained))

    return {
        "experiment": exp_name,
        "name": exp_config["name"],
        "description": exp_config["description"],
        "model_type": exp_config["model_type"],
        "input_mode": exp_config["input_mode"],
        "num_trials": num_trials,
        "epochs": epochs,
        "trials": trials,
        "aggregated": aggregated,
    }


def generate_recall_ranking(
    all_results: dict[str, dict],
) -> str:
    """Generate Recall Ranking markdown table.

    Args:
        all_results: Dictionary mapping experiment name to results

    Returns:
        Markdown string with recall ranking table
    """
    ranked = sorted(
        all_results.values(),
        key=lambda x: x["aggregated"]["screen_photo_recall_mean"],
        reverse=True,
    )

    lines = [
        "# Recall Ranking",
        "",
        "Screen Photo Recall is the most critical metric.",
        "Higher recall means fewer missed screen photos.",
        "",
        "## Ranking Table",
        "",
        "| Rank | Model | SP Recall | Precision | F1 | Accuracy |",
        "|------|-------|-----------|-----------|----|----------|",
    ]

    for rank, result in enumerate(ranked, 1):
        agg = result["aggregated"]
        sp_recall = agg["screen_photo_recall_mean"]
        sp_recall_std = agg["screen_photo_recall_std"]
        sp_prec = agg["screen_photo_precision_mean"]
        sp_prec_std = agg["screen_photo_precision_std"]
        sp_f1 = agg["screen_photo_f1_mean"]
        sp_f1_std = agg["screen_photo_f1_std"]
        acc = agg["accuracy_mean"]
        acc_std = agg["accuracy_std"]

        lines.append(
            f"| {rank} | {result['name']} | "
            f"{sp_recall:.4f}±{sp_recall_std:.4f} | "
            f"{sp_prec:.4f}±{sp_prec_std:.4f} | "
            f"{sp_f1:.4f}±{sp_f1_std:.4f} | "
            f"{acc:.4f}±{acc_std:.4f} |"
        )

    lines.extend(["", "## Key Findings", ""])

    best = ranked[0]
    worst = ranked[-1]
    best_recall = best["aggregated"]["screen_photo_recall_mean"]
    worst_recall = worst["aggregated"]["screen_photo_recall_mean"]

    lines.extend(
        [
            f"- **Best**: {best['name']} with SP Recall = {best_recall:.4f}",
            f"- **Worst**: {worst['name']} with SP Recall = {worst_recall:.4f}",
            "",
            "## Recommendations",
            "",
            "- For maximum recall: use the top-ranked model",
            "- For balanced performance: consider high F1 score",
            "- For production: consider speed vs accuracy trade-off",
        ]
    )

    return "\n".join(lines)


def save_results(
    all_results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Save experiment results to files.

    Args:
        all_results: Dictionary mapping experiment name to results
        output_dir: Output directory
    """
    metrics_dir = output_dir / "metrics"
    confusion_dir = output_dir / "confusion_matrices"
    logs_dir = output_dir / "logs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    for exp_name, result in all_results.items():
        metrics_output = {
            "experiment": result["experiment"],
            "name": result["name"],
            "description": result["description"],
            "model_type": result["model_type"],
            "input_mode": result["input_mode"],
            "num_trials": result["num_trials"],
            "epochs": result["epochs"],
            "aggregated": result["aggregated"],
            "trials_summary": [
                {
                    "trial": t["trial"],
                    "best_val_acc": t["best_val_acc"],
                    "metrics": t["metrics"],
                }
                for t in result["trials"]
            ],
        }

        metrics_path = metrics_dir / f"{exp_name}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics_output, f, indent=2, ensure_ascii=False)

        best_trial = max(result["trials"], key=lambda t: t["best_val_acc"])
        confusion_path = confusion_dir / f"{exp_name}_confusion.json"
        with confusion_path.open("w", encoding="utf-8") as f:
            json.dump(
                best_trial["metrics"]["confusion_matrix"],
                f,
                indent=2,
                ensure_ascii=False,
            )

        logs_path = logs_dir / f"{exp_name}_log.json"
        with logs_path.open("w", encoding="utf-8") as f:
            json.dump(
                [t["train_log"] for t in result["trials"]],
                f,
                indent=2,
                ensure_ascii=False,
            )

    ranking_md = generate_recall_ranking(all_results)
    if len(all_results) == 1:
        exp_name = next(iter(all_results.keys()))
        ranking_filename = f"{exp_name}_recall_ranking.md"
    else:
        ranking_filename = "all_experiments_recall_ranking.md"
    ranking_path = output_dir / ranking_filename
    with ranking_path.open("w", encoding="utf-8") as f:
        f.write(ranking_md)

    # Save summary JSON with experiment-specific filename
    if len(all_results) == 1:
        # Single experiment: use experiment name in filename
        exp_name = next(iter(all_results.keys()))
        summary_filename = f"{exp_name}_summary.json"
    else:
        # Multiple experiments: use combined summary
        summary_filename = "all_experiments_summary.json"

    summary = {
        "experiments": list(all_results.keys()),
        "ranking": [
            {
                "rank": i + 1,
                "name": r["name"],
                "screen_photo_recall": r["aggregated"]["screen_photo_recall_mean"],
                "accuracy": r["aggregated"]["accuracy_mean"],
                "f1_score": r["aggregated"]["f1_score_mean"],
            }
            for i, r in enumerate(
                sorted(
                    all_results.values(),
                    key=lambda x: x["aggregated"]["screen_photo_recall_mean"],
                    reverse=True,
                )
            )
        ],
    }
    summary_path = output_dir / summary_filename
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary: {summary_path}")
    logger.info(f"Recall Ranking: {ranking_path}")

    # Export best model to ONNX
    best_exp = max(
        all_results.values(),
        key=lambda x: x["aggregated"]["screen_photo_recall_mean"],
    )
    best_exp_name = best_exp["experiment"]
    best_checkpoint = output_dir / best_exp_name / "checkpoints" / "best_model.pth"

    if best_checkpoint.exists():
        import shutil

        inference_dir = PROJECT_ROOT / "inference" / "models"
        inference_dir.mkdir(parents=True, exist_ok=True)

        # Copy checkpoint
        dest_checkpoint = inference_dir / "best_model.pth"
        shutil.copy2(best_checkpoint, dest_checkpoint)
        logger.info(f"Best model checkpoint saved to: {dest_checkpoint}")

        # Export to ONNX
        try:
            from trainer_vit.src.export_onnx import export_from_checkpoint

            onnx_path = inference_dir / "best_model.onnx"
            export_from_checkpoint(
                checkpoint_path=str(dest_checkpoint),
                output_path=str(onnx_path),
            )
            logger.info(f"Best model ONNX exported to: {onnx_path}")
        except Exception as e:
            logger.error(f"Failed to export ONNX: {e}")
    else:
        logger.warning(f"Best model checkpoint not found: {best_checkpoint}")


def main() -> None:
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(description="Run screen detector ablation experiments")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/input",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output",
        help="Path to output directory",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of trials per experiment",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs per trial",
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
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=list(EXPERIMENTS.keys()),
        help="Experiments to run",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # Setup logger
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(output_dir / "run_experiments.log")

    logger.info(f"Running experiments: {args.experiments}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    start_time = time.time()
    all_results = {}

    for exp_name in args.experiments:
        if exp_name not in EXPERIMENTS:
            logger.warning(f"Unknown experiment '{exp_name}', skipping")
            continue

        result = run_experiment(
            exp_name=exp_name,
            data_dir=data_dir,
            num_trials=args.num_trials,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            early_stopping_patience=2,
        )
        all_results[exp_name] = result

    save_results(all_results, output_dir)

    elapsed = time.time() - start_time
    logger.info(f"All experiments completed in {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
