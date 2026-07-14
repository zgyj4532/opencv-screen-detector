"""Custom loss functions for screen detector training.

Includes:
- FocalLoss: For handling hard examples and class imbalance
- CenterLoss: For enhancing intra-class compactness
- OHEMLoss: Online Hard Example Mining
- CombinedLoss: FocalLoss + alpha * CenterLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter, higher value focuses more on hard examples.
               Default: 2.0
        alpha: Class weights tensor. If None, no class weighting.
        reduction: Reduction mode ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str | None = None,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Model output logits (B, C)
            targets: Ground truth labels (B,)
            reduction: Override reduction mode (optional)

        Returns:
            Computed focal loss
        """
        red = reduction if reduction is not None else self.reduction

        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        loss = focal_weight * ce_loss

        if self.alpha is not None:
            # Apply class weights
            alpha_t = self.alpha.to(inputs.device)[targets]
            loss = alpha_t * loss

        if red == "mean":
            return loss.mean()
        if red == "sum":
            return loss.sum()
        return loss


class CenterLoss(nn.Module):
    """Center Loss for enhancing intra-class compactness.

    Maintains learnable class centers and minimizes the distance between
    features and their corresponding class centers.

    Reference: Wen et al., "A Discriminative Feature Learning Approach for
    Deep Face Recognition", ECCV 2016

    Args:
        num_classes: Number of classes
        feat_dim: Feature dimension
    """

    def __init__(self, num_classes: int, feat_dim: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute center loss.

        Args:
            features: Feature embeddings (B, feat_dim)
            labels: Ground truth labels (B,)

        Returns:
            Center loss value
        """
        batch_size = features.size(0)
        centers_batch = self.centers[labels]
        return 0.5 * torch.sum((features - centers_batch) ** 2) / batch_size


class OHEMLoss(nn.Module):
    """Online Hard Example Mining.

    Computes per-sample loss, sorts by difficulty (descending),
    and backpropagates only the hardest K% samples.

    This focuses training on the most confusing samples, particularly
    the screenshot <-> screen_photo boundary.

    Args:
        base_criterion: Base loss function (must support reduction='none')
        hard_ratio: Ratio of hard samples to keep (default: 0.3)
    """

    def __init__(self, base_criterion: nn.Module, hard_ratio: float = 0.3) -> None:
        super().__init__()
        self.base_criterion = base_criterion
        self.hard_ratio = hard_ratio

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute OHEM loss.

        Args:
            inputs: Model output logits (B, C)
            targets: Ground truth labels (B,)

        Returns:
            Loss computed on hardest samples only
        """
        # Get per-sample loss
        if isinstance(self.base_criterion, FocalLoss):
            per_sample_loss = self.base_criterion(inputs, targets, reduction="none")
        else:
            per_sample_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Sort by loss (descending = hardest first)
        sorted_loss, _ = torch.sort(per_sample_loss, descending=True)

        # Select hardest K%
        num_hard = max(1, int(len(sorted_loss) * self.hard_ratio))
        hard_loss = sorted_loss[:num_hard]

        return hard_loss.mean()


class CombinedLoss(nn.Module):
    """Combined Loss: FocalLoss + alpha * CenterLoss.

    Total = FocalLoss + center_weight * CenterLoss

    Args:
        focal_criterion: Focal Loss instance
        center_criterion: Center Loss instance
        center_weight: Weight for center loss (default: 0.02)
    """

    def __init__(
        self,
        focal_criterion: FocalLoss,
        center_criterion: CenterLoss,
        center_weight: float = 0.02,
    ) -> None:
        super().__init__()
        self.focal_criterion = focal_criterion
        self.center_criterion = center_criterion
        self.center_weight = center_weight

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss.

        Args:
            logits: Classification logits (B, C)
            features: Feature embeddings (B, feat_dim)
            targets: Ground truth labels (B,)

        Returns:
            Tuple of (total_loss, focal_loss, center_loss)
        """
        focal_loss = self.focal_criterion(logits, targets)
        center_loss = self.center_criterion(features, targets)
        total_loss = focal_loss + self.center_weight * center_loss
        return total_loss, focal_loss, center_loss


def create_criterion(
    use_focal_loss: bool = True,
    class_weights: list[float] | None = None,
    focal_gamma: float = 2.0,
    use_center_loss: bool = False,
    center_weight: float = 0.02,
    num_classes: int = 3,
    feat_dim: int = 1536,
    use_ohem: bool = False,
    ohem_hard_ratio: float = 0.3,
) -> nn.Module:
    """Create loss criterion based on configuration.

    Args:
        use_focal_loss: Whether to use Focal Loss
        class_weights: Class weights for imbalanced dataset
        focal_gamma: Focal loss gamma parameter
        use_center_loss: Whether to combine with Center Loss
        center_weight: Weight for Center Loss
        num_classes: Number of classes (for Center Loss)
        feat_dim: Feature dimension (for Center Loss)
        use_ohem: Whether to use Online Hard Example Mining
        ohem_hard_ratio: Ratio of hard samples for OHEM

    Returns:
        Loss criterion module
    """
    alpha = torch.tensor(class_weights) if class_weights else None

    # Base loss
    if use_focal_loss:
        base_criterion = FocalLoss(gamma=focal_gamma, alpha=alpha)
    else:
        base_criterion = nn.CrossEntropyLoss(weight=alpha)

    # Apply OHEM if requested
    if use_ohem:
        base_criterion = OHEMLoss(base_criterion, hard_ratio=ohem_hard_ratio)

    # Combine with Center Loss if requested
    if use_center_loss:
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
        return CombinedLoss(
            focal_criterion=base_criterion,
            center_criterion=center_criterion,
            center_weight=center_weight,
        )

    return base_criterion
