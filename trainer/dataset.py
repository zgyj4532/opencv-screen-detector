"""Dataset loader for screen detector V3 training.

Supports:
- Two-input dataset (RGB + FFT)
- Recursive subdirectory scanning (for natural_photo/ subfolders)
- Data map configuration for single-stage three-class training
- WeightedRandomSampler for oversampling minority classes
"""

# pyright: reportPrivateImportUsage=none
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
    random_split,
)

from shared.fft_transform import compute_dwt_features as _compute_dwt_shared
from shared.fft_transform import compute_fft_spectrum as _compute_fft_shared

from . import config

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_fft_spectrum(image_np: np.ndarray, size: int = 224) -> np.ndarray:
    """将图像转换为 FFT 频谱图

    Args:
        image_np: 输入图像 (RGB numpy array)
        size: 输出尺寸

    Returns:
        FFT 频谱图，形状 (1, H, W)
    """
    # shared 版本返回 (1, 1, H, W)，squeeze 掉 batch 维度
    return _compute_fft_shared(image_np, size, color_space="rgb").squeeze(0)


def compute_dwt_features(image_np: np.ndarray, size: int = 224) -> np.ndarray:
    """将图像转换为 DWT 小波特征

    Args:
        image_np: 输入图像 (RGB numpy array)
        size: 输出尺寸

    Returns:
        DWT 特征图，形状 (4, H/2, W/2) 包含 [LL, LH, HL, HH]
    """
    # shared 版本返回 (1, 4, H/2, W/2)，squeeze 掉 batch 维度
    return _compute_dwt_shared(image_np, size, color_space="rgb").squeeze(0)


class TwoInputDataset(Dataset):
    """返回 (rgb_image, fft_spectrum, label) 的数据集

    支持:
    - data_map 配置，从多个目录加载并映射到统一标签
    - rglob("*") 递归扫描子目录
    - hard_negative 目录加载（误判图片增强）
    """

    def __init__(
        self,
        data_map: dict[str, list[str]],
        data_dir: Path,
        transform=None,
        image_size: int = config.IMAGE_SIZE,
        load_hard_negatives: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.image_size = image_size

        self.samples: list[tuple[str, int]] = []  # (image_path, label_idx)
        self.hard_negative_indices: list[int] = []  # hard_negative 样本的索引

        self._load_samples(data_map)

        if load_hard_negatives:
            self._load_hard_negatives()

    def _load_samples(self, data_map: dict[str, list[str]]) -> None:
        """Load all image paths and labels from data map."""
        for class_idx, (_class_name, source_dirs) in enumerate(data_map.items()):
            for source_dir in source_dirs:
                dir_path = self.data_dir / source_dir
                if not dir_path.exists():
                    continue

                # 递归扫描，支持子目录
                for img_path in dir_path.rglob("*"):
                    if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    self.samples.append((str(img_path), class_idx))

    def _load_hard_negatives(self) -> None:
        """Load hard negative samples from misclassified images.

        Hard negative 目录结构:
        hard_negative/
        ├── screen_photo_to_screenshot/  # 真实是 screen_photo，被误判为 screenshot
        ├── screenshot_to_screen_photo/  # 真实是 screenshot，被误判为 screen_photo
        ├── natural_to_screenshot/       # 真实是 natural，被误判为 screenshot
        └── ...

        对于 screen_photo_to_screenshot 目录中的图片：
        - 真实标签是 screen_photo (class_idx=2)
        - 但模型容易误判为 screenshot (class_idx=1)
        - 这些是最重要的 hard negative
        """
        hard_negative_dir = config.HARD_NEGATIVE_DIR
        if not hard_negative_dir.exists():
            return

        # 映射误判目录到真实标签
        misclass_to_true_label = {
            "screen_photo_to_screenshot": 2,  # 真实是 screen_photo
            "screen_photo_to_natural": 2,  # 真实是 screen_photo
            "screenshot_to_screen_photo": 1,  # 真实是 screenshot
            "screenshot_to_natural": 1,  # 真实是 screenshot
            "natural_to_screenshot": 0,  # 真实是 natural
            "natural_to_screen_photo": 0,  # 真实是 natural
        }

        for misclass_dir, true_label in misclass_to_true_label.items():
            dir_path = hard_negative_dir / misclass_dir
            if not dir_path.exists():
                continue

            for img_path in dir_path.rglob("*"):
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                idx = len(self.samples)
                self.samples.append((str(img_path), true_label))
                self.hard_negative_indices.append(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Skip images larger than 10MB to avoid MemoryError (check BEFORE loading)
        file_size_mb = Path(img_path).stat().st_size / (1024 * 1024)
        if file_size_mb > 10:
            # Return a blank image for oversized files
            blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            image = blank
        else:
            # Load image using PIL (RGB)
            pil_image = Image.open(img_path).convert("RGBA").convert("RGB")
            image = np.array(pil_image)

        # RGB 分支
        if self.transform:
            augmented = self.transform(image=image)
            rgb_tensor = augmented["image"]
        else:
            # Default: resize and normalize
            image_resized = cv2.resize(image, (self.image_size, self.image_size))
            rgb_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        # FFT 分支
        fft_spectrum = compute_fft_spectrum(image, self.image_size)
        fft_tensor = torch.from_numpy(fft_spectrum).float()

        # DWT 分支
        dwt_features = compute_dwt_features(image, self.image_size)
        dwt_tensor = torch.from_numpy(dwt_features).float()

        return rgb_tensor, fft_tensor, dwt_tensor, label


class TransformSubset(Dataset):
    """Subset of a dataset with applied transform."""

    def __init__(
        self,
        subset: Subset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
        transform: A.Compose | None = None,
    ) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        rgb_tensor, fft_tensor, dwt_tensor, label = self.subset[idx]

        # Convert tensor back to numpy for albumentations
        if isinstance(rgb_tensor, torch.Tensor):
            image_np = rgb_tensor.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = rgb_tensor

        if self.transform:
            augmented = self.transform(image=image_np)
            rgb_tensor = augmented["image"]

        return rgb_tensor, fft_tensor, dwt_tensor, label


class SingleInputDataset(Dataset):
    """返回 (rgb_image, label) 的数据集，用于 PAH-ViT 模型。

    PAH-ViT 使用 Learnable Fourier Token Mixer 替代手工 FFT/DWT，
    因此只需要 RGB 输入，无需预计算频域特征。

    支持:
    - data_map 配置，从多个目录加载并映射到统一标签
    - rglob("*") 递归扫描子目录
    - hard_negative 目录加载（误判图片增强）
    """

    def __init__(
        self,
        data_map: dict[str, list[str]],
        data_dir: Path,
        transform=None,
        image_size: int = config.IMAGE_SIZE,
        load_hard_negatives: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.image_size = image_size

        self.samples: list[tuple[str, int]] = []
        self.hard_negative_indices: list[int] = []

        self._load_samples(data_map)

        if load_hard_negatives:
            self._load_hard_negatives()

    def _load_samples(self, data_map: dict[str, list[str]]) -> None:
        """Load all image paths and labels from data map."""
        for class_idx, (_class_name, source_dirs) in enumerate(data_map.items()):
            for source_dir in source_dirs:
                dir_path = self.data_dir / source_dir
                if not dir_path.exists():
                    continue

                for img_path in dir_path.rglob("*"):
                    if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    self.samples.append((str(img_path), class_idx))

    def _load_hard_negatives(self) -> None:
        """Load hard negative samples from misclassified images."""
        hard_negative_dir = config.HARD_NEGATIVE_DIR
        if not hard_negative_dir.exists():
            return

        misclass_to_true_label = {
            "screen_photo_to_screenshot": 2,
            "screen_photo_to_natural": 2,
            "screenshot_to_screen_photo": 1,
            "screenshot_to_natural": 1,
            "natural_to_screenshot": 0,
            "natural_to_screen_photo": 0,
        }

        for misclass_dir, true_label in misclass_to_true_label.items():
            dir_path = hard_negative_dir / misclass_dir
            if not dir_path.exists():
                continue

            for img_path in dir_path.rglob("*"):
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                idx = len(self.samples)
                self.samples.append((str(img_path), true_label))
                self.hard_negative_indices.append(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Skip images larger than 10MB to avoid MemoryError
        file_size_mb = Path(img_path).stat().st_size / (1024 * 1024)
        if file_size_mb > 10:
            blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            image = blank
        else:
            pil_image = Image.open(img_path).convert("RGBA").convert("RGB")
            image = np.array(pil_image)

        # RGB 分支
        if self.transform:
            augmented = self.transform(image=image)
            rgb_tensor = augmented["image"]
        else:
            image_resized = cv2.resize(image, (self.image_size, self.image_size))
            rgb_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        return rgb_tensor, label


class SingleInputTransformSubset(Dataset):
    """Subset of a SingleInputDataset with applied transform."""

    def __init__(
        self,
        subset: Subset[tuple[torch.Tensor, int]],
        transform: A.Compose | None = None,
    ) -> None:
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rgb_tensor, label = self.subset[idx]

        if isinstance(rgb_tensor, torch.Tensor):
            image_np = rgb_tensor.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = rgb_tensor

        if self.transform:
            augmented = self.transform(image=image_np)
            rgb_tensor = augmented["image"]

        return rgb_tensor, label


def create_single_input_data_loaders(
    data_map: dict[str, list[str]],
    data_dir: Path,
    transform_train: A.Compose | None = None,
    transform_val: A.Compose | None = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    train_ratio: float = config.TRAIN_VAL_SPLIT,
    use_weighted_sampler: bool = False,
    hard_negative_weight: float = config.HARD_NEGATIVE_WEIGHT,
):
    """Create train and validation data loaders for single-input dataset (PAH-ViT).

    Args:
        data_map: Data mapping {class_name: [source_dirs]}
        data_dir: Data directory
        transform_train: Training transforms
        transform_val: Validation transforms
        batch_size: Batch size
        num_workers: Number of workers
        train_ratio: Train/val split ratio
        use_weighted_sampler: Whether to use WeightedRandomSampler for oversampling
        hard_negative_weight: Weight multiplier for hard negative samples

    Returns:
        Tuple of (train_loader, val_loader, full_dataset)
    """

    # Create full dataset (with hard negatives)
    full_dataset = SingleInputDataset(
        data_map=data_map,
        data_dir=data_dir,
        transform=None,
        load_hard_negatives=True,
    )

    print(f"数据集大小: {len(full_dataset)} 张图片")
    print(f"  其中 hard_negative: {len(full_dataset.hard_negative_indices)} 张")

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    # Apply transforms by wrapping datasets
    train_dataset = SingleInputTransformSubset(train_dataset, transform_train)
    val_dataset = SingleInputTransformSubset(val_dataset, transform_val)

    # Create sampler for oversampling if requested
    train_sampler = None
    shuffle = True

    if use_weighted_sampler:
        if hasattr(train_dataset, "subset"):
            train_indices = train_dataset.subset.indices
        else:
            train_indices = list(range(len(train_dataset)))
        train_labels = [full_dataset.samples[i][1] for i in train_indices]

        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

        sample_weights = []
        hard_negative_set = set(full_dataset.hard_negative_indices)

        for i, label in enumerate(train_labels):
            weight = class_weights[label]
            if train_indices[i] in hard_negative_set:
                weight *= hard_negative_weight
            sample_weights.append(weight)

        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)

        train_sampler = WeightedRandomSampler(
            weights=cast("Sequence[int]", sample_weights_tensor),
            num_samples=len(sample_weights_tensor),
            replacement=True,
        )
        shuffle = False

        hard_neg_in_train = sum(1 for i in train_indices if i in hard_negative_set)
        print(f"训练集大小: {train_size} 张")
        print(f"  其中 hard_negative: {hard_neg_in_train} 张")
        print(f"  Hard negative 权重: {hard_negative_weight}x")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, full_dataset


def create_data_loaders(
    data_map: dict[str, list[str]],
    data_dir: Path,
    transform_train: A.Compose | None = None,
    transform_val: A.Compose | None = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    train_ratio: float = config.TRAIN_VAL_SPLIT,
    use_weighted_sampler: bool = False,
    hard_negative_weight: float = config.HARD_NEGATIVE_WEIGHT,
):
    """Create train and validation data loaders for two-input dataset.

    Args:
        data_map: Data mapping {class_name: [source_dirs]}
        data_dir: Data directory
        transform_train: Training transforms
        transform_val: Validation transforms
        batch_size: Batch size
        num_workers: Number of workers
        train_ratio: Train/val split ratio
        use_weighted_sampler: Whether to use WeightedRandomSampler for oversampling
        hard_negative_weight: Weight multiplier for hard negative samples

    Returns:
        Tuple of (train_loader, val_loader, full_dataset)
    """

    # Create full dataset (with hard negatives)
    full_dataset = TwoInputDataset(
        data_map=data_map,
        data_dir=data_dir,
        transform=None,
        load_hard_negatives=True,
    )

    print(f"数据集大小: {len(full_dataset)} 张图片")
    print(f"  其中 hard_negative: {len(full_dataset.hard_negative_indices)} 张")

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    # Apply transforms by wrapping datasets
    train_dataset = TransformSubset(train_dataset, transform_train)
    val_dataset = TransformSubset(val_dataset, transform_val)

    # Create sampler for oversampling if requested
    train_sampler = None
    shuffle = True

    if use_weighted_sampler:
        # Get labels for training subset
        if hasattr(train_dataset, "subset"):
            train_indices = train_dataset.subset.indices
        else:
            train_indices = list(range(len(train_dataset)))
        train_labels = [full_dataset.samples[i][1] for i in train_indices]

        # Calculate class counts and weights
        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

        # Assign weight to each sample
        # Hard negative samples get extra weight
        sample_weights = []
        hard_negative_set = set(full_dataset.hard_negative_indices)

        for i, label in enumerate(train_labels):
            weight = class_weights[label]
            # 如果是 hard_negative 样本，增加权重
            if train_indices[i] in hard_negative_set:
                weight *= hard_negative_weight
            sample_weights.append(weight)

        sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)

        train_sampler = WeightedRandomSampler(
            weights=cast("Sequence[int]", sample_weights_tensor),
            num_samples=len(sample_weights_tensor),
            replacement=True,
        )
        shuffle = False  # Sampler handles shuffling

        # 统计
        hard_neg_in_train = sum(1 for i in train_indices if i in hard_negative_set)
        print(f"训练集大小: {train_size} 张")
        print(f"  其中 hard_negative: {hard_neg_in_train} 张")
        print(f"  Hard negative 权重: {hard_negative_weight}x")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, full_dataset
