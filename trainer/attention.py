"""Attention modules for screen detector training.

Provides lightweight attention mechanisms to enhance feature discrimination:
- ChannelAttention: Squeeze-and-Excitation style channel attention
- SpatialAttention: Spatial attention via channel statistics
- CBAM: Convolutional Block Attention Module (channel + spatial)
- CoordAttention: Coordinate Attention (captures long-range dependencies)

These modules are inserted after the FFT/DWT branches to help the model
focus on discriminative frequency regions for screenshot vs screen_photo.

Reference:
- CBAM: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
- CoordAttention: Hou et al., "Coordinate Attention for Efficient Mobile
  Network Design", CVPR 2021
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention Module (Squeeze-and-Excitation style).

    Learns per-channel weights by aggregating spatial information
    via both average and max pooling.

    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio (default: 16)
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid_channels = max(channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute channel attention weights.

        Args:
            x: Input feature map (B, C, H, W)

        Returns:
            Channel attention weights (B, C, 1, 1)
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module.

    Learns spatial attention weights by aggregating channel information
    via average and max pooling along the channel dimension.

    Args:
        kernel_size: Convolution kernel size (default: 7)
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial attention weights.

        Args:
            x: Input feature map (B, C, H, W)

        Returns:
            Spatial attention weights (B, 1, H, W)
        """
        # Aggregate along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Applies channel attention followed by spatial attention.
    This helps the model focus on both important channels (frequency bands)
    and important spatial regions.

    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio (default: 16)
        kernel_size: Spatial attention kernel size (default: 7)
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CBAM attention.

        Args:
            x: Input feature map (B, C, H, W)

        Returns:
            Attention-weighted feature map (B, C, H, W)
        """
        # Channel attention
        x = x * self.channel_att(x)
        # Spatial attention
        return x * self.spatial_att(x)


class CoordAttention(nn.Module):
    """Coordinate Attention.

    Captures long-range dependencies with precise positional information
    by factorizing channel attention into two 1D feature encoding processes.

    This is particularly useful for frequency domain features where
    positional patterns (e.g., periodic structures) are important.

    Reference: Hou et al., CVPR 2021

    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio (default: 16)
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid_channels = max(channels // reduction, 8)

        # Shared reduction
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Shared MLP
        self.fc1 = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        # Separate F_h and F_w
        self.fc_h = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.fc_w = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Coordinate Attention.

        Args:
            x: Input feature map (B, C, H, W)

        Returns:
            Attention-weighted feature map (B, C, H, W)
        """
        _, _channels, height, width = x.size()

        # Encode along H and W directions
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)

        # Concatenate and transform
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.act(self.bn1(self.fc1(y)))

        # Split and transform
        x_h, x_w = torch.split(y, [height, width], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (B, C, 1, W)

        # Generate attention maps
        att_h = self.sigmoid(self.fc_h(x_h))  # (B, C, H, 1)
        att_w = self.sigmoid(self.fc_w(x_w))  # (B, C, 1, W)

        return x * att_h * att_w


def create_attention(
    channels: int,
    attention_type: str = "cbam",
    reduction: int = 16,
) -> nn.Module:
    """Create attention module based on type.

    Args:
        channels: Number of input channels
        attention_type: Type of attention ('cbam', 'coordinate', 'channel')
        reduction: Channel reduction ratio

    Returns:
        Attention module
    """
    if attention_type == "cbam":
        return CBAM(channels, reduction)
    if attention_type == "coordinate":
        return CoordAttention(channels, reduction)
    if attention_type == "channel":
        return ChannelAttention(channels, reduction)
    raise ValueError(f"Unknown attention type: {attention_type}")
