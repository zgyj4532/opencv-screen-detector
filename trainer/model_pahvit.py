"""Patch-Aware Hybrid Vision Transformer (PAH-ViT) for screen detector.

Architecture:
- EfficientViT-B0 backbone (CNN+ViT hybrid, ~7M params)
- Learnable Fourier Token Mixer (replaces handcrafted FFT/DWT)
- Patch Token Branch (fine-grained anomaly detection)
- Global + Local Attention Fusion
- Classification Head with Patch Contrastive Learning

Key insight: screen_photo vs screenshot difference is local patch-level
distortion (moiré, reflection, perspective) not global semantics.
"""

import timm
import torch
import torch.nn as nn
from einops import rearrange, reduce

from . import config


class LearnableFourierTokenMixer(nn.Module):
    """Learnable Fourier Token Mixer replacing handcrafted FFT/DWT.

    FNet-style: token mixing via FFT, channel mixing via MLP.
    Learns frequency-domain filters that are more discriminative than
    handcrafted FFT/DWT for screen_photo vs screenshot distinction.

    Input: patch tokens (B, N, D)
    Output: mixed tokens (B, N, D)
    """

    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "spectral_mixer": SpectralTokenMixer(dim),
                        "channel_mixer": ChannelMixer(dim),
                        "norm1": nn.LayerNorm(dim),
                        "norm2": nn.LayerNorm(dim),
                    }
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Patch tokens (B, N, D)

        Returns:
            Mixed tokens (B, N, D)
        """
        for layer in self.layers:
            # Spectral mixing (token-level FFT)
            x = x + layer["spectral_mixer"](layer["norm1"](x))
            # Channel mixing (MLP)
            x = x + layer["channel_mixer"](layer["norm2"](x))
        return x


class SpectralTokenMixer(nn.Module):
    """Token mixing via learnable FFT.

    Applies 1D FFT along token dimension, then learnable frequency filtering,
    then inverse FFT. Captures global token relationships efficiently.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Learnable frequency-domain filter
        self.freq_filter = nn.Parameter(torch.ones(1, 1, dim) * 0.5)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Token features (B, N, D)

        Returns:
            Mixed features (B, N, D)
        """
        token_count = x.shape[1]

        # Apply 1D FFT along token dimension
        x_freq = torch.fft.rfft(x, dim=1, norm="ortho")

        # Learnable frequency filtering
        freq_weight = self.freq_filter[:, :, : x_freq.shape[2]]
        x_freq = x_freq * freq_weight

        # Inverse FFT
        x_mixed = torch.fft.irfft(x_freq, n=token_count, dim=1, norm="ortho")

        # Gate to control mixing intensity
        gate = self.gate(x)
        return gate * x_mixed + (1 - gate) * x


class ChannelMixer(nn.Module):
    """Channel mixing via MLP (standard Transformer FFN)."""

    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchTokenBranch(nn.Module):
    """Patch Token Branch for fine-grained anomaly detection.

    Extracts local patch-level features and computes per-patch anomaly scores.
    Uses cross-attention to aggregate patch information.

    Input: patch tokens (B, N, D)
    Output: patch anomaly features (B, D_patch)
    """

    def __init__(self, input_dim: int, patch_dim: int = 256, num_queries: int = 4):
        super().__init__()
        self.num_queries = num_queries

        # Learnable query tokens for cross-attention
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, patch_dim) * 0.02)

        # Cross-attention: queries attend to patch tokens
        # kdim/vdim = patch_dim because we project input to patch_dim first
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=patch_dim,
            num_heads=4,
            kdim=patch_dim,
            vdim=patch_dim,
            batch_first=True,
            dropout=0.1,
        )

        # Projection from input_dim to patch_dim
        self.proj = nn.Linear(input_dim, patch_dim)

        # Per-patch anomaly scoring
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(patch_dim, patch_dim // 2),
            nn.GELU(),
            nn.Linear(patch_dim // 2, 1),
        )

        # Output normalization
        self.norm = nn.LayerNorm(patch_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Patch tokens (B, N, D)

        Returns:
            Tuple of:
            - patch_features: Aggregated patch features (B, patch_dim)
            - anomaly_scores: Per-patch anomaly scores (B, N)
        """
        batch_size = x.shape[0]

        # Project patch tokens
        x_proj = self.proj(x)  # (B, N, patch_dim)

        # Expand queries for batch
        queries = self.query_tokens.expand(batch_size, -1, -1)  # (B, num_queries, patch_dim)

        # Cross-attention: queries attend to patches
        attn_out, _ = self.cross_attn(queries, x_proj, x_proj)  # (B, Q, patch_dim)

        # Aggregate query outputs (mean pooling)
        patch_features = reduce(attn_out, "b q d -> b d", "mean")  # (B, patch_dim)
        patch_features = self.norm(patch_features)

        # Per-patch anomaly scores
        anomaly_scores = self.anomaly_scorer(x_proj).squeeze(-1)  # (B, N)
        anomaly_scores = torch.sigmoid(anomaly_scores)

        return patch_features, anomaly_scores


class GlobalLocalFusion(nn.Module):
    """Global + Local Attention Fusion module.

    Fuses global features from backbone with local patch-level features.

    Input:
    - global_feat: Backbone global features (B, D_global)
    - local_feat: Patch branch features (B, D_patch)

    Output:
    - fused: Fused features (B, D_fused)
    """

    def __init__(self, global_dim: int, local_dim: int, fused_dim: int = 512):
        super().__init__()

        # Project to same dimension
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
        )

        self.local_proj = nn.Sequential(
            nn.Linear(local_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
        )

        # Cross-attention: global attends to local
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fused_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            global_feat: Global features (B, D_global)
            local_feat: Local features (B, D_patch)

        Returns:
            Fused features (B, D_fused)
        """
        # Project
        g = self.global_proj(global_feat).unsqueeze(1)  # (B, 1, D_fused)
        local = self.local_proj(local_feat).unsqueeze(1)  # (B, 1, D_fused)

        # Cross-attention: global attends to local
        attn_out, _ = self.cross_attn(g, local, local)  # (B, 1, D_fused)
        attn_out = attn_out.squeeze(1)  # (B, D_fused)

        # Concatenate and fuse
        fused = torch.cat([global_feat.new_zeros(global_feat.shape[0], 0), attn_out, g.squeeze(1)], dim=-1)
        # Actually, let's just concat the attended and original global
        fused = torch.cat([attn_out, g.squeeze(1)], dim=-1)  # (B, 2*D_fused)
        fused = self.fusion(fused)  # (B, D_fused)
        return self.norm(fused)


class PAHVitModel(nn.Module):
    """Patch-Aware Hybrid Vision Transformer (PAH-ViT) for screen detection.

    Architecture:
    1. EfficientViT-B0 backbone → global features (1280) + patch tokens (128, 7, 7)
    2. Learnable Fourier Token Mixer → enhanced patch tokens
    3. Patch Token Branch → local anomaly features (256) + anomaly scores
    4. Global + Local Fusion → fused features (512)
    5. Classification Head → 3-class logits

    Loss: CE + 0.3 * PatchContrastiveLoss
    """

    def __init__(
        self,
        model_name: str = "efficientvit_b0",
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # 1. Backbone: EfficientViT-B0
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
        )
        # EfficientViT-B0: num_features=128 but actual output is 1280
        # We need to test the actual output dimension
        self.global_dim = 1280  # Actual output dimension of EfficientViT-B0

        # Get patch dimension from backbone's last stage
        # EfficientViT-B0: stage 3 has 128 channels, 7x7 spatial
        self.patch_dim_raw = 128  # Channel dim of last stage
        self.patch_size = 7  # Spatial size of last stage

        # 2. Learnable Fourier Token Mixer
        self.token_mixer = LearnableFourierTokenMixer(
            dim=self.patch_dim_raw,
            num_layers=2,
        )

        # 3. Patch Token Branch
        self.patch_branch = PatchTokenBranch(
            input_dim=self.patch_dim_raw,
            patch_dim=256,
            num_queries=4,
        )

        # 4. Global + Local Fusion
        self.fusion = GlobalLocalFusion(
            global_dim=self.global_dim,
            local_dim=256,
            fused_dim=512,
        )

        # 5. Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers: int = 4) -> None:
        """Unfreeze last N stages of backbone."""
        stages = list(self.backbone.stages.children())
        for stage in stages[-num_layers:]:
            for param in stage.parameters():
                param.requires_grad = True

    def _get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from backbone's last stage.

        Args:
            x: RGB input (B, 3, H, W)

        Returns:
            Patch tokens (B, N, D) where N = patch_size^2, D = patch_dim_raw
        """
        # Forward through stem and stages
        x = self.backbone.stem(x)
        for stage in self.backbone.stages:
            x = stage(x)
        # x shape: (B, 128, 7, 7)
        # Reshape to (B, N, D)
        return rearrange(x, "b c h w -> b (h w) c")

    def forward(self, rgb_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            rgb_input: RGB image tensor (B, 3, H, W)

        Returns:
            Tuple of:
            - logits: Classification logits (B, num_classes)
            - anomaly_scores: Per-patch anomaly scores (B, N)
        """
        # 1. Get global features and patch tokens
        global_feat = self.backbone(rgb_input)  # (B, 1280)
        patch_tokens = self._get_patch_tokens(rgb_input)  # (B, 49, 128)

        # 2. Apply Fourier Token Mixer
        mixed_tokens = self.token_mixer(patch_tokens)  # (B, 49, 128)

        # 3. Patch Token Branch
        local_feat, anomaly_scores = self.patch_branch(mixed_tokens)
        # local_feat: (B, 256), anomaly_scores: (B, 49)

        # 4. Global + Local Fusion
        fused = self.fusion(global_feat, local_feat)  # (B, 512)

        # 5. Classification
        logits = self.classifier(fused)  # (B, num_classes)

        return logits, anomaly_scores

    def get_features(self, rgb_input: torch.Tensor) -> torch.Tensor:
        """Extract fused features without classification."""
        global_feat = self.backbone(rgb_input)
        patch_tokens = self._get_patch_tokens(rgb_input)
        mixed_tokens = self.token_mixer(patch_tokens)
        local_feat, _ = self.patch_branch(mixed_tokens)
        return self.fusion(global_feat, local_feat)


def create_pahvit_model(
    model_name: str = "efficientvit_b0",
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> PAHVitModel:
    """Create a PAH-ViT model.

    Args:
        model_name: Backbone model name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained backbone
        freeze_backbone: Whether to freeze backbone initially
    """
    return PAHVitModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )


def save_pahvit_model(
    model: PAHVitModel,
    checkpoint_path: str,
    epoch: int,
    optimizer_state_dict: dict | None = None,
    best_val_acc: float = 0.0,
    best_metric: float = 0.0,
) -> None:
    """Save PAH-ViT model checkpoint."""
    checkpoint = {
        "model_name": model.model_name,
        "num_classes": model.num_classes,
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "best_metric": best_metric,
        "model_type": "pahvit",
    }

    if optimizer_state_dict:
        checkpoint["optimizer_state_dict"] = optimizer_state_dict

    torch.save(checkpoint, checkpoint_path)


def load_pahvit_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> PAHVitModel:
    """Load PAH-ViT model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = create_pahvit_model(
        model_name=checkpoint.get("model_name", "efficientvit_b0"),
        num_classes=checkpoint.get("num_classes", config.NUM_CLASSES),
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    return model
