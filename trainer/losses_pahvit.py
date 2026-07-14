"""Loss functions for PAH-ViT training.

Includes:
- FocalLoss: For handling hard examples and class imbalance
- PatchContrastiveLoss: For patch-level contrastive learning between
  screen_photo and screenshot classes

The total loss is:
    L = CE_Loss + λ * PatchContrastiveLoss
where λ = 0.3 (configurable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 3.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class PatchContrastiveLoss(nn.Module):
    """Patch-level Contrastive Loss for distinguishing screen_photo vs screenshot.

    Pulls together patches from same class, pushes apart patches from different classes.
    Only applied to screen_photo (class 2) and screenshot (class 1) patches.

    Uses supervised contrastive learning on patch features:
    L = -log(exp(sim(z_i, z_j)/τ) / Σ exp(sim(z_i, z_k)/τ))

    where z_i, z_j are patches from same class, z_k from different class.

    Args:
        temperature: Temperature parameter for softmax (default: 0.07)
        screen_photo_idx: Index of screen_photo class (default: 2)
        screenshot_idx: Index of screenshot class (default: 1)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        screen_photo_idx: int = 2,
        screenshot_idx: int = 1,
    ):
        super().__init__()
        self.temperature = temperature
        self.screen_photo_idx = screen_photo_idx
        self.screenshot_idx = screenshot_idx

    def forward(
        self,
        anomaly_scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute patch contrastive loss on anomaly scores.

        We use anomaly_scores as a proxy for patch-level features.
        Patches with similar anomaly patterns should be from the same class.

        Args:
            anomaly_scores: Per-patch anomaly scores (B, N)
            labels: Ground truth labels (B,)

        Returns:
            Scalar loss value
        """
        device = anomaly_scores.device
        B, N = anomaly_scores.shape

        # Only consider screen_photo and screenshot samples
        mask = (labels == self.screen_photo_idx) | (labels == self.screenshot_idx)
        if mask.sum() < 2:
            return torch.tensor(0.0, device=device)

        # Filter to relevant classes
        scores = anomaly_scores[mask]  # (B', N)
        filtered_labels = labels[mask]  # (B',)
        B_prime = scores.shape[0]

        if B_prime < 2:
            return torch.tensor(0.0, device=device)

        # Compute pairwise similarity matrix
        # Normalize scores for cosine-like similarity
        scores_norm = F.normalize(scores, p=2, dim=1)  # (B', N)
        sim_matrix = torch.mm(scores_norm, scores_norm.t()) / self.temperature  # (B', B')

        # Create positive mask: same class
        labels_expanded = filtered_labels.unsqueeze(0) == filtered_labels.unsqueeze(1)  # (B', B')
        # Remove self-similarity
        self_mask = ~torch.eye(B_prime, dtype=torch.bool, device=device)
        pos_mask = labels_expanded & self_mask  # (B', B')
        neg_mask = ~labels_expanded  # (B', B')

        # For each anchor, compute loss
        # L = -log(exp(sim_pos) / (exp(sim_pos) + Σ exp(sim_neg)))
        loss = torch.tensor(0.0, device=device)
        count = 0

        for i in range(B_prime):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = neg_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            # Positive similarities
            pos_sim = sim_matrix[i, pos_indices]  # (num_pos,)
            # Negative similarities
            neg_sim = sim_matrix[i, neg_indices]  # (num_neg,)

            # Compute log-softmax
            # Concatenate pos and neg, pos at index 0
            logits = torch.cat([pos_sim, neg_sim])  # (1 + num_neg,)
            # Target: 0 (first position is positive)
            target = torch.zeros(1, dtype=torch.long, device=device)

            loss += F.cross_entropy(logits.unsqueeze(0), target)
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)

        return loss / count


class PAHVitLoss(nn.Module):
    """Combined loss for PAH-ViT training.

    L = CE_Loss + λ * PatchContrastiveLoss

    Args:
        lambda_contrastive: Weight for contrastive loss (default: 0.3)
        class_weights: Class weights for imbalanced dataset
        focal_gamma: Focal loss gamma parameter
    """

    def __init__(
        self,
        lambda_contrastive: float = 0.3,
        class_weights: list[float] | None = None,
        focal_gamma: float = 3.0,
    ):
        super().__init__()
        self.lambda_contrastive = lambda_contrastive

        alpha = torch.tensor(class_weights) if class_weights else None
        self.ce_loss = FocalLoss(gamma=focal_gamma, alpha=alpha)
        self.contrastive_loss = PatchContrastiveLoss()

    def forward(
        self,
        logits: torch.Tensor,
        anomaly_scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss.

        Args:
            logits: Classification logits (B, num_classes)
            anomaly_scores: Per-patch anomaly scores (B, N)
            labels: Ground truth labels (B,)

        Returns:
            Tuple of (total_loss, ce_loss, contrastive_loss)
        """
        ce = self.ce_loss(logits, labels)
        contrastive = self.contrastive_loss(anomaly_scores, labels)
        total = ce + self.lambda_contrastive * contrastive
        return total, ce, contrastive


def create_pahvit_criterion(
    lambda_contrastive: float = 0.3,
    class_weights: list[float] | None = None,
    focal_gamma: float = 3.0,
) -> PAHVitLoss:
    """Create PAH-ViT loss criterion.

    Args:
        lambda_contrastive: Weight for contrastive loss
        class_weights: Class weights for imbalanced dataset
        focal_gamma: Focal loss gamma parameter
    """
    return PAHVitLoss(
        lambda_contrastive=lambda_contrastive,
        class_weights=class_weights,
        focal_gamma=focal_gamma,
    )
