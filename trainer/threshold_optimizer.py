"""Adaptive Threshold Optimizer for screen detector.

Instead of using fixed thresholds (e.g., argmax or fixed 0.60 for screen_photo),
this module searches for class-specific confidence thresholds that maximize
a target metric (default: screen_photo F1).

This is particularly important for the screen_photo class which has:
- Lower precision (confused with screenshot)
- Lower recall (hard to detect)

Usage:
    optimizer = ThresholdOptimizer(class_names=["natural", "screenshot", "screen_photo"])
    best_thresholds = optimizer.search(probabilities, labels)

    # Apply thresholds
    predictions = optimizer.predict(probabilities)
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass
class ThresholdResult:
    """Result of threshold optimization."""

    thresholds: dict[str, float]
    metrics: dict[str, float]
    predictions: np.ndarray


class ThresholdOptimizer:
    """Search optimal class-specific thresholds on validation set.

    The optimizer performs grid search over thresholds for each class,
    optimizing for the target metric (default: screen_photo F1).

    Args:
        class_names: List of class names
        target_class: Class to optimize for (default: 'screen_photo')
        metric: Metric to optimize ('f1', 'precision', 'recall')
    """

    def __init__(
        self,
        class_names: list[str],
        target_class: str = "screen_photo",
        metric: str = "f1",
    ) -> None:
        self.class_names = class_names
        self.target_class = target_class
        self.target_class_idx = class_names.index(target_class)
        self.metric = metric
        self.optimal_thresholds: dict[str, float] | None = None

    def search(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        threshold_range: tuple[float, float] = (0.30, 0.80),
        step: float = 0.05,
    ) -> ThresholdResult:
        """Grid search for optimal thresholds.

        Args:
            probabilities: Model output probabilities (N, num_classes)
            labels: Ground truth labels (N,)
            threshold_range: Range to search (min, max)
            step: Search step size

        Returns:
            ThresholdResult with optimal thresholds and metrics
        """
        best_thresholds = dict.fromkeys(self.class_names, 0.5)
        best_score = 0.0
        best_predictions = np.argmax(probabilities, axis=1)

        # Search target class threshold
        for sp_threshold in np.arange(threshold_range[0], threshold_range[1] + step, step):
            predictions = self._apply_thresholds(probabilities, sp_threshold)

            score = self._compute_metric(labels, predictions)

            if score > best_score:
                best_score = score
                best_thresholds[self.class_names[self.target_class_idx]] = sp_threshold
                best_predictions = predictions

        self.optimal_thresholds = best_thresholds

        # Compute final metrics
        final_metrics = self._compute_all_metrics(labels, best_predictions)

        return ThresholdResult(
            thresholds=best_thresholds,
            metrics=final_metrics,
            predictions=best_predictions,
        )

    def _apply_thresholds(
        self,
        probabilities: np.ndarray,
        target_threshold: float,
    ) -> np.ndarray:
        """Apply threshold-based classification.

        If target class probability >= threshold, classify as target class.
        Otherwise, use argmax.

        Args:
            probabilities: Model output probabilities (N, num_classes)
            target_threshold: Threshold for target class

        Returns:
            Predicted class indices (N,)
        """
        # Default: argmax
        predictions = np.argmax(probabilities, axis=1)

        # Override: if target class prob >= threshold
        target_probs = probabilities[:, self.target_class_idx]
        target_mask = target_probs >= target_threshold
        predictions[target_mask] = self.target_class_idx

        return predictions

    def _compute_metric(self, labels: np.ndarray, predictions: np.ndarray) -> float:
        """Compute the target metric.

        Args:
            labels: Ground truth labels
            predictions: Predicted labels

        Returns:
            Metric value
        """
        if self.metric == "f1":
            return f1_score(
                labels,
                predictions,
                labels=[self.target_class_idx],
                average="macro",
                zero_division=0,
            )
        if self.metric == "precision":
            return precision_score(
                labels,
                predictions,
                labels=[self.target_class_idx],
                average="macro",
                zero_division=0,
            )
        if self.metric == "recall":
            return recall_score(
                labels,
                predictions,
                labels=[self.target_class_idx],
                average="macro",
                zero_division=0,
            )
        raise ValueError(f"Unknown metric: {self.metric}")

    def _compute_all_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
    ) -> dict[str, float]:
        """Compute all metrics for the predictions.

        Args:
            labels: Ground truth labels
            predictions: Predicted labels

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score

        # Per-class metrics
        precision_per_class = precision_score(
            labels, predictions, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            labels, predictions, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            labels, predictions, average=None, zero_division=0
        )

        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision_macro": np.mean(precision_per_class),
            "recall_macro": np.mean(recall_per_class),
            "f1_macro": np.mean(f1_per_class),
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "f1_per_class": f1_per_class,
        }

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply optimal thresholds to new probabilities.

        Args:
            probabilities: Model output probabilities (N, num_classes)

        Returns:
            Predicted class indices (N,)
        """
        if self.optimal_thresholds is None:
            raise ValueError("Must call search() before predict()")

        target_threshold = self.optimal_thresholds.get(
            self.target_class, 0.5
        )
        return self._apply_thresholds(probabilities, target_threshold)

    def print_thresholds(self) -> None:
        """Print the optimal thresholds."""
        if self.optimal_thresholds is None:
            print("No thresholds optimized yet. Call search() first.")
            return

        print("\n" + "=" * 50)
        print("Optimal Thresholds")
        print("=" * 50)
        for class_name, threshold in self.optimal_thresholds.items():
            marker = " <-- optimized" if class_name == self.target_class else ""
            print(f"  {class_name}: {threshold:.4f}{marker}")
        print("=" * 50)


def optimize_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
    target_class: str = "screen_photo",
) -> ThresholdResult:
    """Convenience function to optimize thresholds.

    Args:
        probabilities: Model output probabilities (N, num_classes)
        labels: Ground truth labels (N,)
        class_names: List of class names
        target_class: Class to optimize for

    Returns:
        ThresholdResult with optimal thresholds and metrics
    """
    if class_names is None:
        class_names = ["natural", "screenshot", "screen_photo"]

    optimizer = ThresholdOptimizer(
        class_names=class_names,
        target_class=target_class,
    )
    result = optimizer.search(probabilities, labels)
    optimizer.print_thresholds()

    return result
