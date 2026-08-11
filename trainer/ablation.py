"""Ablation study for screen detector optimization modules.

Trains and evaluates the model with different combinations of optimization
modules to measure the impact of each component.

Usage:
    # Run full ablation study
    uv run python -m trainer ablation

    # Run specific ablation
    uv run python -m trainer ablation --modules center_loss,arcface
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import config
from .train import train_three_class


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""

    name: str
    description: str
    use_center_loss: bool = False
    use_ohem: bool = False
    use_arcface: bool = False
    use_fft_attention: bool = False
    use_adaptive_threshold: bool = False


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""

    config_name: str
    accuracy: float = 0.0
    sp_precision: float = 0.0
    sp_recall: float = 0.0
    sp_f1: float = 0.0
    macro_f1: float = 0.0
    training_time: float = 0.0
    details: dict = field(default_factory=dict)


# Define ablation experiments
ABLATION_CONFIGS = [
    AblationConfig(
        name="baseline",
        description="Baseline: Focal Loss only (current best)",
        use_center_loss=False,
        use_ohem=False,
        use_arcface=False,
        use_fft_attention=False,
        use_adaptive_threshold=False,
    ),
    AblationConfig(
        name="+center_loss",
        description="Baseline + Center Loss",
        use_center_loss=True,
        use_ohem=False,
        use_arcface=False,
        use_fft_attention=False,
        use_adaptive_threshold=False,
    ),
    AblationConfig(
        name="+ohem",
        description="Baseline + OHEM",
        use_center_loss=False,
        use_ohem=True,
        use_arcface=False,
        use_fft_attention=False,
        use_adaptive_threshold=False,
    ),
    AblationConfig(
        name="+arcface",
        description="Baseline + ArcFace",
        use_center_loss=False,
        use_ohem=False,
        use_arcface=True,
        use_fft_attention=False,
        use_adaptive_threshold=False,
    ),
    AblationConfig(
        name="+attention",
        description="Baseline + FFT Attention (CBAM)",
        use_center_loss=False,
        use_ohem=False,
        use_arcface=False,
        use_fft_attention=True,
        use_adaptive_threshold=False,
    ),
    AblationConfig(
        name="+threshold",
        description="Baseline + Adaptive Threshold",
        use_center_loss=False,
        use_ohem=False,
        use_arcface=False,
        use_fft_attention=False,
        use_adaptive_threshold=True,
    ),
    AblationConfig(
        name="full",
        description="All optimizations enabled",
        use_center_loss=True,
        use_ohem=True,
        use_arcface=True,
        use_fft_attention=True,
        use_adaptive_threshold=True,
    ),
]


def run_ablation(
    ablation_config: AblationConfig,
    device: str | None = None,
    epochs_head: int = 5,  # Reduced for faster ablation
    epochs_finetune: int = 10,  # Reduced for faster ablation
) -> AblationResult:
    """Run a single ablation experiment.

    Args:
        ablation_config: Configuration for this ablation
        device: Device to use
        epochs_head: Epochs for head training (reduced for ablation)
        epochs_finetune: Epochs for fine-tuning (reduced for ablation)

    Returns:
        AblationResult with metrics
    """
    print(f"\n{'#' * 60}")
    print(f"# Ablation: {ablation_config.name}")
    print(f"# {ablation_config.description}")
    print(f"{'#' * 60}")

    start_time = time.time()

    try:
        _, _, metrics = train_three_class(
            use_center_loss=ablation_config.use_center_loss,
            use_ohem=ablation_config.use_ohem,
            use_arcface=ablation_config.use_arcface,
            use_fft_attention=ablation_config.use_fft_attention,
            use_adaptive_threshold=ablation_config.use_adaptive_threshold,
            epochs_head=epochs_head,
            epochs_finetune=epochs_finetune,
            device=device,
        )

        training_time = time.time() - start_time

        # Extract screen_photo metrics (class index 2)
        sp_idx = 2
        result = AblationResult(
            config_name=ablation_config.name,
            accuracy=metrics["accuracy"],
            sp_precision=metrics["precision_per_class"][sp_idx],
            sp_recall=metrics["recall_per_class"][sp_idx],
            sp_f1=metrics["f1_per_class"][sp_idx],
            macro_f1=metrics["f1_macro"],
            training_time=training_time,
        )

    except Exception as e:
        print(f"Error in ablation {ablation_config.name}: {e}")
        result = AblationResult(
            config_name=ablation_config.name,
            training_time=time.time() - start_time,
            details={"error": str(e)},
        )

    return result


def print_results(results: list[AblationResult]) -> None:
    """Print ablation results in a formatted table.

    Args:
        results: List of ablation results
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print(
        f"{'Config':<20} {'Accuracy':<10} {'SP Prec':<10} {'SP Recall':<10} {'SP F1':<10} {'Macro F1':<10} {'Time':<10}"
    )
    print("-" * 80)

    for result in results:
        print(
            f"{result.config_name:<20} "
            f"{result.accuracy:<10.4f} "
            f"{result.sp_precision:<10.4f} "
            f"{result.sp_recall:<10.4f} "
            f"{result.sp_f1:<10.4f} "
            f"{result.macro_f1:<10.4f} "
            f"{result.training_time:<10.1f}s"
        )

    print("=" * 80)

    # Find best configuration
    valid_results = [r for r in results if r.accuracy > 0]
    if valid_results:
        best = max(valid_results, key=lambda r: r.sp_f1)
        print(f"\nBest configuration (by SP F1): {best.config_name}")
        print(f"  SP F1: {best.sp_f1:.4f}")
        print(f"  SP Precision: {best.sp_precision:.4f}")
        print(f"  SP Recall: {best.sp_recall:.4f}")
        print(f"  Accuracy: {best.accuracy:.4f}")


def save_results(
    results: list[AblationResult],
    save_path: str | None = None,
) -> None:
    """Save ablation results to JSON file.

    Args:
        results: List of ablation results
        save_path: Path to save results
    """
    if save_path is None:
        save_path = str(config.LOG_DIR / "ablation_results.json")

    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [],
    }

    for result in results:
        data["results"].append(
            {
                "config_name": result.config_name,
                "accuracy": result.accuracy,
                "sp_precision": result.sp_precision,
                "sp_recall": result.sp_recall,
                "sp_f1": result.sp_f1,
                "macro_f1": result.macro_f1,
                "training_time": result.training_time,
                "details": result.details,
            }
        )

    with Path(save_path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {save_path}")


def run_full_ablation(
    device: str | None = None,
    epochs_head: int = 5,
    epochs_finetune: int = 10,
    configs: list[AblationConfig] | None = None,
) -> list[AblationResult]:
    """Run full ablation study.

    Args:
        device: Device to use
        epochs_head: Epochs for head training
        epochs_finetune: Epochs for fine-tuning
        configs: Specific configs to run (default: all)

    Returns:
        List of ablation results
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if configs is None:
        configs = ABLATION_CONFIGS

    print(f"\n{'=' * 60}")
    print("ABLATION STUDY")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs_head} (head) + {epochs_finetune} (finetune)")
    print(f"Experiments: {len(configs)}")

    results = []
    for i, ablation_config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running: {ablation_config.name}")
        result = run_ablation(
            ablation_config,
            device=device,
            epochs_head=epochs_head,
            epochs_finetune=epochs_finetune,
        )
        results.append(result)

    # Print and save results
    print_results(results)
    save_results(results)

    return results


def parse_modules(module_str: str) -> list[str]:
    """Parse comma-separated module names.

    Args:
        module_str: Comma-separated module names

    Returns:
        List of module names
    """
    return [m.strip() for m in module_str.split(",")]


def get_configs_for_modules(module_names: list[str]) -> list[AblationConfig]:
    """Get ablation configs for specific modules.

    Args:
        module_names: List of module names to test

    Returns:
        List of ablation configs
    """
    all_configs = {
        "baseline": ABLATION_CONFIGS[0],
        "center_loss": ABLATION_CONFIGS[1],
        "ohem": ABLATION_CONFIGS[2],
        "arcface": ABLATION_CONFIGS[3],
        "attention": ABLATION_CONFIGS[4],
        "threshold": ABLATION_CONFIGS[5],
        "full": ABLATION_CONFIGS[6],
    }

    configs = []
    for name in module_names:
        if name in all_configs:
            configs.append(all_configs[name])
        else:
            print(f"Warning: Unknown module '{name}', skipping")

    return configs


def main():
    """Main entry point for ablation study."""
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--modules",
        type=str,
        default=None,
        help="Comma-separated list of modules to test (default: all)",
    )
    parser.add_argument(
        "--epochs-head",
        type=int,
        default=5,
        help="Epochs for head training (default: 5)",
    )
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=10,
        help="Epochs for fine-tuning (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto)",
    )

    args = parser.parse_args()

    # Get configs
    if args.modules:
        module_names = parse_modules(args.modules)
        configs = get_configs_for_modules(module_names)
    else:
        configs = None  # Run all

    # Run ablation
    run_full_ablation(
        device=args.device,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        configs=configs,
    )


if __name__ == "__main__":
    main()
