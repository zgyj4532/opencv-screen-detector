"""DWT (Discrete Wavelet Transform) Branch for screen photo detection.

Uses Haar wavelet to decompose images into 4 subbands:
- LL: Low-Low (approximation, contains structure/edges)
- LH: Low-High (horizontal details, moiré patterns)
- HL: High-Low (vertical details, screen bezels)
- HH: High-High (diagonal details, screen textures)

All 4 subbands are retained and processed through a CNN for feature extraction.
This is critical for screen photo detection as moiré patterns and screen textures
are captured in the LH/HL/HH subbands.
"""

from typing import cast

import torch
import torch.nn as nn


class DWTForward(nn.Module):
    """Haar DWT 2D transform module.

    Decomposes input image into 4 subbands (LL, LH, HL, HH).
    """

    def __init__(self) -> None:
        super().__init__()
        # Haar wavelet filter coefficients
        # Low-pass filter
        lo = torch.tensor([1.0, 1.0], dtype=torch.float32) / 2.0
        # High-pass filter
        hi = torch.tensor([1.0, -1.0], dtype=torch.float32) / 2.0

        # 2D filters: outer product
        ll = lo[:, None] * lo[None, :]  # (2, 2)
        lh = lo[:, None] * hi[None, :]
        hl = hi[:, None] * lo[None, :]
        hh = hi[:, None] * hi[None, :]

        # Stack as (4, 1, 2, 2) for convolution
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 2D Haar DWT.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            DWT coefficients (B, C*4, H/2, W/2)
        """
        b, c, h, w = x.shape
        filters = cast(torch.Tensor, self.filters.to(x.device))

        # Reshape for grouped convolution: (B*C, 1, H, W)
        x_grouped = x.reshape(b * c, 1, h, w)

        # Apply DWT filters with stride 2
        # filters shape: (4, 1, 2, 2)
        out = nn.functional.conv2d(x_grouped, filters, stride=2, padding=0)

        # Reshape back: (B, C*4, H/2, W/2)
        return out.reshape(b, c * 4, h // 2, w // 2)


class DWTCNNBranch(nn.Module):
    """CNN branch for processing DWT subbands.

    Architecture:
    - DWT decomposition: (B, 3, H, W) -> (B, 12, H/2, W/2)
    - 4 conv blocks to extract features from subbands
    - Global average pooling + projection
    """

    def __init__(self, in_channels: int = 12, out_features: int = 256) -> None:
        """Initialize DWT CNN branch.

        Args:
            in_channels: Number of input channels (3*4=12 for RGB with 4 subbands)
            out_features: Output feature dimension
        """
        super().__init__()

        self.dwt = DWTForward()

        self.features = nn.Sequential(
            # Block 1: (12, 112, 112) -> (64, 56, 56)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Block 2: (64, 56, 56) -> (128, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # Block 3: (128, 28, 28) -> (256, 14, 14)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # Block 4: (256, 14, 14) -> (256, 7, 7)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Projection to output dimension
        self.proj = nn.Linear(256, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: RGB image tensor (B, 3, H, W)

        Returns:
            DWT features (B, out_features)
        """
        # Apply DWT: (B, 3, H, W) -> (B, 12, H/2, W/2)
        x = self.dwt(x)

        # CNN features: (B, 12, H/2, W/2) -> (B, 256, 1, 1)
        x = self.features(x)
        x = self.pool(x)

        # Flatten and project: (B, 256) -> (B, out_features)
        x = x.flatten(1)
        return self.proj(x)


class DWTFeatureExtractor(nn.Module):
    """Extract DWT features for visualization and analysis.

    Returns intermediate subband features for Grad-CAM visualization.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dwt = DWTForward()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract DWT subbands.

        Args:
            x: RGB image tensor (B, 3, H, W)

        Returns:
            Dictionary with subband tensors:
            - 'll': Low-Low (approximation) (B, 3, H/2, W/2)
            - 'lh': Low-High (horizontal) (B, 3, H/2, W/2)
            - 'hl': High-Low (vertical) (B, 3, H/2, W/2)
            - 'hh': High-High (diagonal) (B, 3, H/2, W/2)
            - 'all': All subbands concatenated (B, 12, H/2, W/2)
        """
        dwt_coeffs = self.dwt(x)  # (B, 12, H/2, W/2)

        # Split into 4 subbands (each has 3 channels for RGB)
        ll = dwt_coeffs[:, 0:3, :, :]  # Low-Low
        lh = dwt_coeffs[:, 3:6, :, :]  # Low-High
        hl = dwt_coeffs[:, 6:9, :, :]  # High-Low
        hh = dwt_coeffs[:, 9:12, :, :]  # High-High

        return {
            "ll": ll,
            "lh": lh,
            "hl": hl,
            "hh": hh,
            "all": dwt_coeffs,
        }


def compute_dwt_subbands(image: torch.Tensor) -> dict[str, torch.Tensor]:
    """Utility function to compute DWT subbands from a single image.

    Args:
        image: Input image tensor (C, H, W) or (B, C, H, W)

    Returns:
        Dictionary with subband tensors
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)

    extractor = DWTFeatureExtractor()
    return extractor(image)
