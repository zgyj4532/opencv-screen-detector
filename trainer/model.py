"""EfficientNet + FFT Branch model for screen detector V3.

Single-stage 3-class CNN architecture with frequency domain analysis.
"""

# pyright: reportPrivateImportUsage=none
from typing import cast

import timm
import torch
import torch.nn as nn

from . import config
from .fft_branch import FrequencyBranch, FrequencyBranchWithDWT


class ScreenDetectorModel(nn.Module):
    """EfficientNet + FFT Branch fusion model.

    Architecture:
    - Spatial Branch: EfficientNet-B0 -> spatial_features (1280,)
    - Frequency Branch: FFT CNN -> freq_features (256,)
    - Fusion: Concat -> LayerNorm -> Classifier
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Spatial Branch (RGB)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.spatial_dim = cast("int", self.backbone.num_features)  # 1280

        # Frequency Branch (FFT)
        self.freq_branch = FrequencyBranch(out_features=256)

        # Feature Normalization (修正 #5)
        self.spatial_norm = nn.LayerNorm(self.spatial_dim)
        self.freq_norm = nn.LayerNorm(256)

        # Fusion Classifier
        fused_dim = self.spatial_dim + 256  # 1536
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers: int = 2) -> None:
        """Unfreeze last N layers of backbone."""
        children = list(self.backbone.children())
        for child in children[-num_layers:]:
            for param in child.parameters():
                param.requires_grad = True

    def forward(self, rgb_input: torch.Tensor, fft_input: torch.Tensor) -> torch.Tensor:
        """Forward pass with dual inputs.

        Args:
            rgb_input: RGB image tensor (B, 3, H, W)
            fft_input: FFT spectrum tensor (B, 1, H, W)

        Returns:
            Classification logits (B, num_classes)
        """
        spatial_feat = self.spatial_norm(self.backbone(rgb_input))
        freq_feat = self.freq_norm(self.freq_branch(fft_input))
        fused = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.classifier(fused)

    def get_features(
        self, rgb_input: torch.Tensor, fft_input: torch.Tensor
    ) -> torch.Tensor:
        """Extract fused features without classification."""
        spatial_feat = self.spatial_norm(self.backbone(rgb_input))
        freq_feat = self.freq_norm(self.freq_branch(fft_input))
        return torch.cat([spatial_feat, freq_feat], dim=1)


class ScreenDetectorModelWithDWT(nn.Module):
    """EfficientNet + FFT + DWT Branch fusion model.

    Architecture:
    - Spatial Branch: EfficientNet-B0 -> spatial_features (1280,)
    - Frequency Branch: FFT + DWT CNN -> freq_features (256,)
    - Fusion: Concat -> LayerNorm -> Classifier

    DWT 提供多尺度小波特征:
    - LL: 低频近似 (结构信息)
    - LH/HL/HH: 高频细节 (边缘、纹理、噪声)
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Spatial Branch (RGB)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.spatial_dim = cast("int", self.backbone.num_features)  # 1280

        # Frequency Branch (FFT + DWT)
        self.freq_branch = FrequencyBranchWithDWT(out_features=256)

        # Feature Normalization
        self.spatial_norm = nn.LayerNorm(self.spatial_dim)
        self.freq_norm = nn.LayerNorm(256)

        # Fusion Classifier
        fused_dim = self.spatial_dim + 256  # 1536
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers: int = 2) -> None:
        """Unfreeze last N layers of backbone."""
        children = list(self.backbone.children())
        for child in children[-num_layers:]:
            for param in child.parameters():
                param.requires_grad = True

    def forward(
        self,
        rgb_input: torch.Tensor,
        fft_input: torch.Tensor,
        dwt_input: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with triple inputs.

        Args:
            rgb_input: RGB image tensor (B, 3, H, W)
            fft_input: FFT spectrum tensor (B, 1, H, W)
            dwt_input: DWT features tensor (B, 4, H/2, W/2)

        Returns:
            Classification logits (B, num_classes)
        """
        spatial_feat = self.spatial_norm(self.backbone(rgb_input))
        freq_feat = self.freq_norm(self.freq_branch(fft_input, dwt_input))
        fused = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.classifier(fused)

    def get_features(
        self,
        rgb_input: torch.Tensor,
        fft_input: torch.Tensor,
        dwt_input: torch.Tensor,
    ) -> torch.Tensor:
        """Extract fused features without classification."""
        spatial_feat = self.spatial_norm(self.backbone(rgb_input))
        freq_feat = self.freq_norm(self.freq_branch(fft_input, dwt_input))
        return torch.cat([spatial_feat, freq_feat], dim=1)


def create_model(
    model_name: str = config.MODEL_NAME,
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    use_dwt: bool = True,
) -> ScreenDetectorModel | ScreenDetectorModelWithDWT:
    """Create a screen detector model.

    Args:
        model_name: Backbone model name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained backbone
        freeze_backbone: Whether to freeze backbone initially
        use_dwt: Whether to use DWT branch (default: True)
    """
    if use_dwt:
        return ScreenDetectorModelWithDWT(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    return ScreenDetectorModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )


def load_model(
    checkpoint_path: str,
    device: str = "cpu",
    use_dwt: bool = True,
) -> ScreenDetectorModel | ScreenDetectorModelWithDWT:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        use_dwt: Whether to load DWT variant (default: True)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = create_model(
        model_name=checkpoint.get("model_name", config.MODEL_NAME),
        num_classes=checkpoint.get("num_classes", config.NUM_CLASSES),
        pretrained=False,
        use_dwt=use_dwt,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def save_model(
    model: ScreenDetectorModel | ScreenDetectorModelWithDWT,
    checkpoint_path: str,
    epoch: int,
    optimizer_state_dict: dict | None = None,
    best_val_acc: float = 0.0,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "model_name": model.model_name,
        "num_classes": model.num_classes,
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "use_dwt": isinstance(model, ScreenDetectorModelWithDWT),
    }

    if optimizer_state_dict:
        checkpoint["optimizer_state_dict"] = optimizer_state_dict

    torch.save(checkpoint, checkpoint_path)
