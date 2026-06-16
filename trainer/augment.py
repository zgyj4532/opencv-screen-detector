"""Data augmentation for screen detector V3 training."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from . import config


def get_train_transforms():
    """Get training data augmentation transforms."""
    return A.Compose(
        [
            # Resize
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            # Random crop and resize
            A.RandomResizedCrop(
                size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            ),
            # Horizontal flip
            A.HorizontalFlip(p=0.5),
            # Vertical flip (less common for screen photos)
            A.VerticalFlip(p=0.1),
            # Rotation
            A.Rotate(limit=15, p=0.5),
            # Color jitter
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            # Gaussian blur
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            # Motion blur (simulates camera shake)
            A.MotionBlur(blur_limit=7, p=0.2),
            # JPEG compression artifacts
            A.ImageCompression(
                quality_range=config.JPEG_QUALITY_RANGE,
                p=0.3,
            ),
            # Noise
            A.GaussNoise(p=0.3),
            # Perspective transform (simulates viewing angle)
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            # Normalize (ImageNet stats)
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            # Convert to tensor
            ToTensorV2(),
        ]
    )


def get_val_transforms():
    """Get validation data transforms (no augmentation)."""
    return A.Compose(
        [
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )
