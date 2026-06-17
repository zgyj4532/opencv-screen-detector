"""Dataset for screen detector with multi-input mode support.

Supports three input modes for different model architectures:
- "rgb": RGB only (for DeiTScreenDetector)
- "fft": RGB + FFT (for FFTDeiT)
- "dwt_fft": RGB + FFT (for DWTFFTDeiT, DWT computed from RGB in model)

Three-class classification: natural, screenshot, screen_photo.
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import get_train_transforms, get_val_transforms

# Label mapping
LABEL_MAP = {
    "natural_photo": 0,  # natural
    "screenshot": 1,  # screenshot
    "screen_photo": 2,  # screen_photo
}

LABEL_NAMES = ["natural", "screenshot", "screen_photo"]

# Valid image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# FFT normalization constants (ImageNet grayscale)
FFT_NORM_MEAN = 0.449
FFT_NORM_STD = 0.226


def compute_fft_spectrum(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Compute FFT spectrum from RGB image.

    Args:
        image: RGB image (H, W, 3) with values in [0, 255]
        size: Output size

    Returns:
        FFT spectrum (1, H, W) normalized with ImageNet stats
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (size, size))

    # FFT
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # Magnitude spectrum
    magnitude = np.log(np.abs(fshift) + 1)

    # Normalize to [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # pyright: ignore[reportCallIssue, reportArgumentType]

    # Normalize to [0, 1] then ImageNet grayscale normalization
    magnitude = magnitude.astype(np.float32) / 255.0
    magnitude = (magnitude - FFT_NORM_MEAN) / FFT_NORM_STD

    return magnitude.reshape(1, size, size)


class ScreenDetectorDataset(Dataset):
    """Screen detector dataset with multi-input mode support.

    Args:
        data_dir: Root directory containing class folders
        transform: Albumentations transform
        split: Dataset split ('train' or 'val')
        input_mode: Input mode ("rgb", "fft", or "dwt_fft")
        image_size: Image size for FFT computation
    """

    def __init__(
        self,
        data_dir: str | Path,
        transform: A.Compose | None = None,
        split: str = "train",
        input_mode: str = "rgb",
        image_size: int = 224,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.input_mode = input_mode
        self.image_size = image_size

        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Load all samples from data directory.

        Uses rglob to recursively scan subdirectories (e.g. natural_photo/animal/).
        """
        for class_name, label in LABEL_MAP.items():
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                logger.warning(f"{class_dir} does not exist, skipping")
                continue

            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, label))

        logger.info(f"[{self.split}] Loaded {len(self.samples)} samples (mode={self.input_mode})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        """Get sample by index.

        Args:
            idx: Sample index

        Returns:
            Tuple depends on input_mode:
            - "rgb": (image_tensor, label)
            - "fft": (image_tensor, fft_tensor, label)
            - "dwt_fft": (image_tensor, fft_tensor, label)
        """
        img_path, label = self.samples[idx]

        # Load image as RGB
        image = Image.open(img_path).convert("RGB")
        # Resize large images to save memory
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        image = np.array(image)

        # Apply transforms (returns tensor)
        if self.transform:
            transformed = self.transform(image=image)
            image_tensor = transformed["image"]
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.input_mode == "rgb":
            return image_tensor, label

        # Compute FFT spectrum for fft/dwt_fft modes
        # Use original image (before normalization) for FFT
        fft_spectrum = compute_fft_spectrum(image, self.image_size)
        fft_tensor = torch.from_numpy(fft_spectrum).float()

        return image_tensor, fft_tensor, label


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    image_size: int = 224,
    seed: int = 42,
    input_mode: str = "rgb",
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        data_dir: Root data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_split: Validation split ratio
        image_size: Image size for model input
        seed: Random seed for split
        input_mode: Input mode ("rgb", "fft", or "dwt_fft")

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create train dataset with augmentation
    train_dataset = ScreenDetectorDataset(
        data_dir=data_dir,
        transform=get_train_transforms(image_size),
        split="train",
        input_mode=input_mode,
        image_size=image_size,
    )

    # Create val dataset without augmentation
    val_dataset = ScreenDetectorDataset(
        data_dir=data_dir,
        transform=get_val_transforms(image_size),
        split="val",
        input_mode=input_mode,
        image_size=image_size,
    )

    # Split dataset
    total_size = len(train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_subset)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_subset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader
