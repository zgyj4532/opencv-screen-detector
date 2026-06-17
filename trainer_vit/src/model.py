"""DeiT model variants for screen detector.

Three model variants for ablation study:
1. DeiTScreenDetector: Pure DeiT (RGB only)
2. FFTDeiT: Dual-stream (RGB-DeiT + FFT-DeiT)
3. DWTFFTDeiT: Triple-stream (RGB-DeiT + FFT-DeiT + DWT-CNN)

Uses DeiT-Small (deit_small_patch16_224) from timm.
"""

from typing import cast

import timm
import torch
import torch.nn as nn

from .dwt_branch import DWTCNNBranch

# Default config - 使用 DeiT-Small
MODEL_NAME = "deit_small_patch16_224"
NUM_CLASSES = 3


class DeiTScreenDetector(nn.Module):
    """DeiT-Small model for screen detection (Experiment 2).

    Architecture:
    - DeiT-Small backbone (ImageNet pretrained, 22M params)
    - Stochastic Depth (drop_path_rate=0.1)
    - Classification head (3 classes)

    Args:
        model_name: timm model name
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone parameters
        drop_path_rate: Stochastic depth rate
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Create DeiT model with stochastic depth
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters except classifier head."""
        for name, param in self.model.named_parameters():
            if "head" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor (B, 3, 224, 224)

        Returns:
            Classification logits (B, num_classes)
        """
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head.

        Args:
            x: Input image tensor (B, 3, 224, 224)

        Returns:
            Feature tensor (B, hidden_dim)
        """
        features = cast(nn.Module, self.model.forward_features)(x)
        return features[:, 0]  # CLS token


class FFTDeiT(nn.Module):
    """Dual-stream DeiT for RGB + FFT fusion (Experiment 3).

    Architecture:
    - RGB Stream: DeiT-Small -> rgb_features (384,)
    - FFT Stream: DeiT-Small -> fft_features (384,)
    - Fusion: Concat -> LayerNorm -> Classifier

    Args:
        model_name: timm model name for both streams
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone parameters
        drop_path_rate: Stochastic depth rate
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # RGB Stream (DeiT-Small)
        self.rgb_stream = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            drop_path_rate=drop_path_rate,
        )

        # FFT Stream (DeiT-Small) - processes FFT spectrum as 3-channel image
        self.fft_stream = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            drop_path_rate=drop_path_rate,
        )

        # Get feature dimension from DeiT-Small
        self.feat_dim = cast(int, self.rgb_stream.num_features)  # 384 for DeiT-Small

        # Feature normalization
        self.rgb_norm = nn.LayerNorm(self.feat_dim)
        self.fft_norm = nn.LayerNorm(self.feat_dim)

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feat_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze both stream backbones."""
        for param in self.rgb_stream.parameters():
            param.requires_grad = False
        for param in self.fft_stream.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
        for param in self.rgb_stream.parameters():
            param.requires_grad = True
        for param in self.fft_stream.parameters():
            param.requires_grad = True

    def forward(
        self,
        rgb_input: torch.Tensor,
        fft_input: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with dual inputs.

        Args:
            rgb_input: RGB image tensor (B, 3, 224, 224)
            fft_input: FFT spectrum tensor (B, 1, 224, 224)

        Returns:
            Classification logits (B, num_classes)
        """
        # RGB stream
        rgb_feat = cast(nn.Module, self.rgb_stream.forward_features)(rgb_input)  # (B, 384)
        rgb_feat = self.rgb_norm(rgb_feat[:, 0])  # CLS token

        # FFT stream: convert 1-channel to 3-channel for DeiT
        fft_3ch = fft_input.repeat(1, 3, 1, 1)  # (B, 3, 224, 224)
        fft_feat = cast(nn.Module, self.fft_stream.forward_features)(fft_3ch)  # (B, 384)
        fft_feat = self.fft_norm(fft_feat[:, 0])  # CLS token

        # Fusion
        fused = torch.cat([rgb_feat, fft_feat], dim=1)  # (B, 768)
        return self.classifier(fused)

    def get_features(
        self,
        rgb_input: torch.Tensor,
        fft_input: torch.Tensor,
    ) -> torch.Tensor:
        """Extract fused features without classification."""
        rgb_feat = cast(nn.Module, self.rgb_stream.forward_features)(rgb_input)[:, 0]
        fft_3ch = fft_input.repeat(1, 3, 1, 1)
        fft_feat = cast(nn.Module, self.fft_stream.forward_features)(fft_3ch)[:, 0]
        return torch.cat([rgb_feat, fft_feat], dim=1)


class DWTFFTDeiT(nn.Module):
    """Triple-stream DeiT for RGB + FFT + DWT fusion (Experiment 4).

    Architecture:
    - RGB Stream: DeiT-Small -> rgb_features (384,)
    - FFT Stream: DeiT-Small -> fft_features (384,)
    - DWT Stream: CNN -> dwt_features (256,)
    - Fusion: Concat -> LayerNorm -> Classifier

    Args:
        model_name: timm model name for RGB/FFT streams
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone parameters
        drop_path_rate: Stochastic depth rate
        dwt_out_features: DWT branch output dimension
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        drop_path_rate: float = 0.1,
        dwt_out_features: int = 256,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.dwt_out_features = dwt_out_features

        # RGB Stream (DeiT-Small)
        self.rgb_stream = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )

        # FFT Stream (DeiT-Small)
        self.fft_stream = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )

        # DWT Stream (CNN)
        self.dwt_stream = DWTCNNBranch(in_channels=12, out_features=dwt_out_features)

        # Get feature dimension from DeiT-Small
        self.feat_dim = cast(int, self.rgb_stream.num_features)  # 384

        # Feature normalization
        self.rgb_norm = nn.LayerNorm(self.feat_dim)
        self.fft_norm = nn.LayerNorm(self.feat_dim)
        self.dwt_norm = nn.LayerNorm(dwt_out_features)

        # Fusion classifier
        fused_dim = self.feat_dim * 2 + dwt_out_features  # 384*2 + 256 = 1024
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze all stream backbones."""
        for param in self.rgb_stream.parameters():
            param.requires_grad = False
        for param in self.fft_stream.parameters():
            param.requires_grad = False
        for param in self.dwt_stream.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
        for param in self.rgb_stream.parameters():
            param.requires_grad = True
        for param in self.fft_stream.parameters():
            param.requires_grad = True
        for param in self.dwt_stream.parameters():
            param.requires_grad = True

    def forward(
        self,
        rgb_input: torch.Tensor,
        fft_input: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with dual inputs (DWT computed from RGB).

        Args:
            rgb_input: RGB image tensor (B, 3, 224, 224)
            fft_input: FFT spectrum tensor (B, 1, 224, 224)

        Returns:
            Classification logits (B, num_classes)
        """
        # RGB stream
        rgb_feat = cast(nn.Module, self.rgb_stream.forward_features)(rgb_input)  # (B, 384)
        rgb_feat = self.rgb_norm(rgb_feat[:, 0])  # CLS token

        # FFT stream: convert 1-channel to 3-channel for DeiT
        fft_3ch = fft_input.repeat(1, 3, 1, 1)  # (B, 3, 224, 224)
        fft_feat = cast(nn.Module, self.fft_stream.forward_features)(fft_3ch)  # (B, 384)
        fft_feat = self.fft_norm(fft_feat[:, 0])  # CLS token

        # DWT stream: computed from RGB input
        dwt_feat = self.dwt_stream(rgb_input)  # (B, 256)
        dwt_feat = self.dwt_norm(dwt_feat)

        # Fusion
        fused = torch.cat([rgb_feat, fft_feat, dwt_feat], dim=1)  # (B, 1024)
        return self.classifier(fused)

    def get_features(
        self,
        rgb_input: torch.Tensor,
        fft_input: torch.Tensor,
    ) -> torch.Tensor:
        """Extract fused features without classification."""
        rgb_feat = cast(nn.Module, self.rgb_stream.forward_features)(rgb_input)[:, 0]
        fft_3ch = fft_input.repeat(1, 3, 1, 1)
        fft_feat = cast(nn.Module, self.fft_stream.forward_features)(fft_3ch)[:, 0]
        dwt_feat = self.dwt_stream(rgb_input)
        return torch.cat([rgb_feat, fft_feat, dwt_feat], dim=1)


# Factory functions
def create_deit_model(
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    drop_path_rate: float = 0.1,
) -> DeiTScreenDetector:
    """Create a DeiT screen detector model (Experiment 2)."""
    return DeiTScreenDetector(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        drop_path_rate=drop_path_rate,
    )


def create_fft_deit_model(
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    drop_path_rate: float = 0.1,
) -> FFTDeiT:
    """Create a dual-stream FFT+DeiT model (Experiment 3)."""
    return FFTDeiT(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        drop_path_rate=drop_path_rate,
    )


def create_dwt_fft_deit_model(
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    drop_path_rate: float = 0.1,
    dwt_out_features: int = 256,
) -> DWTFFTDeiT:
    """Create a triple-stream DWT+FFT+DeiT model (Experiment 4)."""
    return DWTFFTDeiT(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        drop_path_rate=drop_path_rate,
        dwt_out_features=dwt_out_features,
    )


def load_deit_model(
    checkpoint_path: str,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load DeiT model from checkpoint.

    Automatically detects model type from checkpoint metadata.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model (DeiTScreenDetector, FFTDeiT, or DWTFFTDeiT)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_type = checkpoint.get("model_type", "deit")
    model_name = checkpoint.get("model_name", MODEL_NAME)
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)

    if model_type == "deit":
        model = create_deit_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
        )
    elif model_type == "fft_deit":
        model = create_fft_deit_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
        )
    elif model_type == "dwt_fft_deit":
        model = create_dwt_fft_deit_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def save_deit_model(
    model: nn.Module,
    checkpoint_path: str,
    epoch: int,
    model_type: str = "deit",
    optimizer_state_dict: dict | None = None,
    best_val_acc: float = 0.0,
) -> None:
    """Save DeiT model checkpoint.

    Args:
        model: Model to save
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch
        model_type: Model type identifier ("deit", "fft_deit", "dwt_fft_deit")
        optimizer_state_dict: Optimizer state dict
        best_val_acc: Best validation accuracy
    """
    checkpoint = {
        "model_type": model_type,
        "model_name": model.model_name,
        "num_classes": model.num_classes,
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
    }

    if optimizer_state_dict:
        checkpoint["optimizer_state_dict"] = optimizer_state_dict

    torch.save(checkpoint, checkpoint_path)
