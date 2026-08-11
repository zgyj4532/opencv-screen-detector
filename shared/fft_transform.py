"""FFT 频谱变换模块

将图像转换为 FFT 频谱图和 DWT 小波特征，用于频域特征分析。
训练和推理两端共享此模块。

支持:
- FFT: log(abs(fft)) 频谱图
- DWT: 离散小波变换 (Haar) 低频/高频分量
"""

import cv2
import numpy as np

# ImageNet 灰度归一化常量
FFT_NORM_MEAN = 0.449
FFT_NORM_STD = 0.226


def compute_fft_spectrum(
    image: np.ndarray,
    size: int = 224,
    color_space: str = "bgr",
) -> np.ndarray:
    """将图像转换为 FFT 频谱图

    Args:
        image: 输入图像 (H, W) 灰度 或 (H, W, 3) 彩色
        size: 输出尺寸
        color_space: 彩色图像的色彩空间，"bgr" 或 "rgb"

    Returns:
        FFT 频谱图，形状 (1, 1, H, W)
    """
    # 灰度化
    if len(image.shape) == 3:
        code = cv2.COLOR_BGR2GRAY if color_space == "bgr" else cv2.COLOR_RGB2GRAY
        gray = cv2.cvtColor(image, code)
    else:
        gray = image

    # resize 到目标尺寸
    gray = cv2.resize(gray, (size, size))

    # FFT
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # 频谱图 (magnitude)
    magnitude = np.log(np.abs(fshift) + 1)

    # 归一化到 [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # pyright: ignore[reportCallIssue, reportArgumentType]

    # 归一化到 [0, 1] 然后 ImageNet 灰度归一化
    magnitude = magnitude.astype(np.float32) / 255.0
    magnitude = (magnitude - FFT_NORM_MEAN) / FFT_NORM_STD

    # 转换为 (1, 1, H, W) 形状
    return magnitude.reshape(1, 1, size, size)


def compute_dwt_features(
    image: np.ndarray,
    size: int = 224,
    color_space: str = "bgr",
) -> np.ndarray:
    """将图像转换为 DWT 小波特征

    使用 Haar 小波进行单层分解，提取:
    - LL: 低频近似系数 (包含主要结构信息)
    - LH/HL/HH: 高频细节系数 (包含边缘和纹理)

    每个子带独立做 z-score 归一化:
        band = (band - mean) / (std + 1e-8)

    最后 resize 到目标尺寸，确保与 FFT/RGB 尺寸一致。

    Args:
        image: 输入图像 (H, W) 灰度 或 (H, W, 3) 彩色
        size: 输出尺寸 (默认 224)
        color_space: 彩色图像的色彩空间，"bgr" 或 "rgb"

    Returns:
        DWT 特征图，形状 (1, 4, size, size) 包含 [LL, LH, HL, HH]
    """
    # 灰度化
    if len(image.shape) == 3:
        code = cv2.COLOR_BGR2GRAY if color_space == "bgr" else cv2.COLOR_RGB2GRAY
        gray = cv2.cvtColor(image, code)
    else:
        gray = image

    # resize 到目标尺寸
    gray = cv2.resize(gray, (size, size)).astype(np.float32)

    # Haar 小波分解 (手动实现，避免依赖 pywt)
    # 使用简单的 2x2 平均/差分

    # 低频 (LL): 2x2 块的平均值
    ll = (gray[0::2, 0::2] + gray[0::2, 1::2] + gray[1::2, 0::2] + gray[1::2, 1::2]) / 4.0

    # 水平高频 (LH): 行差分
    lh = (gray[0::2, 0::2] + gray[0::2, 1::2] - gray[1::2, 0::2] - gray[1::2, 1::2]) / 4.0

    # 垂直高频 (HL): 列差分
    hl = (gray[0::2, 0::2] - gray[0::2, 1::2] + gray[1::2, 0::2] - gray[1::2, 1::2]) / 4.0

    # 对角高频 (HH): 对角差分
    hh = (gray[0::2, 0::2] - gray[0::2, 1::2] - gray[1::2, 0::2] + gray[1::2, 1::2]) / 4.0

    # 每个子带独立 z-score 归一化
    # 避免 LL 主导梯度
    def zscore_normalize(band: np.ndarray) -> np.ndarray:
        mean = band.mean()
        std = band.std()
        return (band - mean) / (std + 1e-8)

    ll_norm = zscore_normalize(ll)
    lh_norm = zscore_normalize(np.abs(lh))
    hl_norm = zscore_normalize(np.abs(hl))
    hh_norm = zscore_normalize(np.abs(hh))

    # Stack to (4, H/2, W/2)
    dwt_features = np.stack([ll_norm, lh_norm, hl_norm, hh_norm], axis=0)

    # Resize 到目标尺寸，确保与 FFT/RGB 尺寸一致
    # (4, H/2, W/2) -> (4, size, size)
    dwt_resized = np.zeros((4, size, size), dtype=np.float32)
    for i in range(4):
        dwt_resized[i] = cv2.resize(dwt_features[i], (size, size))

    # 转换为 (1, 4, size, size) 形状
    return dwt_resized.reshape(1, 4, size, size)


def compute_dwt_features_from_bytes(image_bytes: bytes, size: int = 224) -> np.ndarray:
    """从字节数据计算 DWT 特征

    Args:
        image_bytes: 图片字节数据
        size: 输出尺寸

    Returns:
        DWT 特征图，形状 (1, 4, H/2, W/2)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image from bytes")
    return compute_dwt_features(image, size)


def compute_fft_spectrum_from_bytes(image_bytes: bytes, size: int = 224) -> np.ndarray:
    """从字节数据计算 FFT 频谱图

    Args:
        image_bytes: 图片字节数据
        size: 输出尺寸

    Returns:
        FFT 频谱图，形状 (1, 1, H, W)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image from bytes")
    return compute_fft_spectrum(image, size)
