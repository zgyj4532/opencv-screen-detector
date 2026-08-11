"""Frequency Branch 模型定义

轻量 CNN 用于处理 FFT 频谱图和 DWT 小波特征，提取频域特征。

支持两种模式:
- FrequencyBranch: 仅 FFT 输入 (1 channel)
- FrequencyBranchWithDWT: FFT + DWT 输入 (1 + 4 = 5 channels)

DWT 包含:
- LL: 低频近似系数 (主要结构)
- LH/HL/HH: 高频细节系数 (边缘和纹理)

可选注意力模块:
- CBAM: Convolutional Block Attention Module
- Coordinate Attention: 捕获长程依赖
"""

import torch
import torch.nn as nn

from .attention import create_attention

FREQUENCY_INPUT_SIZE = 224
POOLED_SIZE = 4
POOL_KERNEL_SIZE = FREQUENCY_INPUT_SIZE // POOLED_SIZE


class ResBlock(nn.Module):
    """残差块"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class FrequencyBranch(nn.Module):
    """Frequency Branch 用于处理 FFT 频谱图

    结构: Conv -> ResBlock -> Conv -> ResBlock -> [Attention] -> AdaptivePool -> FC
    输入: (B, 1, H, W) FFT 频谱图

    Args:
        out_features: 输出特征维度
        use_attention: 是否使用注意力模块
        attention_type: 注意力类型 ('cbam', 'coordinate')
        attention_reduction: 通道注意力降维比例
    """

    def __init__(
        self,
        out_features: int = 256,
        use_attention: bool = False,
        attention_type: str = "cbam",
        attention_reduction: int = 16,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
        )

        # Optional attention module
        self.attention = None
        if use_attention:
            self.attention = create_attention(64, attention_type, attention_reduction)

        self.pool = nn.AvgPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.attention is not None:
            x = self.attention(x)
        x = self.pool(x)
        return self.classifier(x)


class FrequencyBranchWithDWT(nn.Module):
    """Frequency Branch with DWT 用于处理 FFT + DWT 特征

    结构:
    - FFT 分支: Conv -> ResBlock -> Conv -> ResBlock -> [Attention]
    - DWT 分支: Conv -> ResBlock -> Conv -> ResBlock -> [Attention]
    - Fusion: Concat -> AdaptivePool -> FC

    输入:
    - fft_input: (B, 1, H, W) FFT 频谱图
    - dwt_input: (B, 4, H/2, W/2) DWT 小波特征 [LL, LH, HL, HH]

    Args:
        out_features: 输出特征维度
        use_attention: 是否使用注意力模块
        attention_type: 注意力类型 ('cbam', 'coordinate')
        attention_reduction: 通道注意力降维比例
    """

    def __init__(
        self,
        out_features: int = 256,
        use_attention: bool = False,
        attention_type: str = "cbam",
        attention_reduction: int = 16,
    ):
        super().__init__()

        # FFT branch (1 channel input)
        self.fft_features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
        )

        # DWT branch (4 channels input: LL, LH, HL, HH)
        self.dwt_features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResBlock(64),
        )

        # Optional attention modules
        self.fft_attention = None
        self.dwt_attention = None
        if use_attention:
            self.fft_attention = create_attention(64, attention_type, attention_reduction)
            self.dwt_attention = create_attention(64, attention_type, attention_reduction)

        # Fusion: 64 (FFT) + 64 (DWT) = 128 channels
        self.fusion = nn.Sequential(
            nn.AvgPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, fft_input: torch.Tensor, dwt_input: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            fft_input: FFT 频谱图 (B, 1, H, W)
            dwt_input: DWT 特征 (B, 4, H/2, W/2)

        Returns:
            Fused frequency features (B, out_features)
        """
        # FFT features
        fft_feat = self.fft_features(fft_input)  # (B, 64, H, W)
        if self.fft_attention is not None:
            fft_feat = self.fft_attention(fft_feat)

        # DWT features
        dwt_feat = self.dwt_features(dwt_input)  # (B, 64, H/2, W/2)
        if self.dwt_attention is not None:
            dwt_feat = self.dwt_attention(dwt_feat)

        # Resize DWT to match FFT spatial dimensions
        dwt_feat = nn.functional.interpolate(dwt_feat, size=fft_feat.shape[2:], mode="bilinear", align_corners=False)

        # Concatenate and fuse
        fused = torch.cat([fft_feat, dwt_feat], dim=1)  # (B, 128, H, W)

        return self.fusion(fused)
