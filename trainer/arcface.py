"""ArcFace Margin Classifier for screen detector training.

ArcFace adds an additive angular margin penalty to the target logits,
enlarging the inter-class separation and enhancing intra-class compactness.

Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep
Face Recognition", CVPR 2019

Usage:
    # In model __init__:
    self.classifier = ArcMarginProduct(feat_dim, num_classes, s=30.0, m=0.50)

    # In forward (training):
    logits = self.classifier(features, labels)

    # In forward (inference):
    logits = self.classifier(features)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """ArcFace: Additive Angular Margin Loss.

    Replaces the standard Linear classifier with an angular margin
    to increase inter-class separation in the embedding space.

    The decision boundary becomes cos(theta + m) for the target class,
    which is stricter than the standard cos(theta).

    Args:
        in_features: Size of input features (e.g., 1536 for EfficientNet + FFT)
        out_features: Number of classes (e.g., 3)
        s: Scale factor for logits (default: 30.0)
        m: Angular margin in radians (default: 0.50, ~28.6 degrees)
        easy_margin: Whether to use easy margin variant (default: False)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.50,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        # Learnable class weight vectors
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin terms
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Feature embeddings (B, in_features)
            labels: Ground truth labels (B,) - required for training,
                    None for inference

        Returns:
            Logits scaled by s (B, out_features)
        """
        # Normalize features and weights
        features_norm = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: cos(theta)
        cosine = F.linear(features_norm, weight_norm)

        if labels is None:
            # Inference: no margin applied
            return cosine * self.s

        # Training: add angular margin to target class
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine = torch.sqrt(1.0 - torch.clamp(cosine.pow(2), 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            # Easy margin: only apply when cos(theta) > 0
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Hard margin: use threshold to avoid gradient issues
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to target class
        output = cosine * (1.0 - one_hot) + phi * one_hot

        return output * self.s


class ArcFaceClassifier(nn.Module):
    """ArcFace Classifier wrapper that manages features and classification.

    This wrapper handles the full pipeline:
    1. Feature normalization (optional)
    2. ArcFace margin classification

    Args:
        in_features: Size of input features
        out_features: Number of classes
        s: Scale factor
        m: Angular margin
        use_feature_norm: Whether to apply LayerNorm to features
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.50,
        use_feature_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Optional feature normalization
        self.feature_norm = nn.LayerNorm(in_features) if use_feature_norm else nn.Identity()

        # ArcFace classifier
        self.arcface = ArcMarginProduct(in_features, out_features, s=s, m=m)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Raw feature embeddings (B, in_features)
            labels: Ground truth labels (B,) - None for inference

        Returns:
            Logits (B, out_features)
        """
        features = self.feature_norm(features)
        return self.arcface(features, labels)


def create_arcface_classifier(
    in_features: int = 1536,
    num_classes: int = 3,
    s: float = 30.0,
    m: float = 0.50,
) -> ArcFaceClassifier:
    """Create ArcFace classifier with default settings.

    Args:
        in_features: Feature dimension (EfficientNet 1280 + FFT 256 = 1536)
        num_classes: Number of classes
        s: Scale factor (default: 30.0)
        m: Angular margin (default: 0.50)

    Returns:
        ArcFaceClassifier instance
    """
    return ArcFaceClassifier(
        in_features=in_features,
        out_features=num_classes,
        s=s,
        m=m,
    )
