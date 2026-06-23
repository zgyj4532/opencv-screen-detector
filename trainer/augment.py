"""Data augmentation for screen detector V3 training.

Enhanced augmentation strategy for better screen_photo detection.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from . import config


def get_train_transforms():
    """Get training data augmentation transforms.

    Enhanced augmentation for better generalization, especially for screen_photo:
    - Stronger geometric transforms (perspective, rotation)
    - More aggressive color jitter
    - Simulate camera capture artifacts (blur, noise, moire)
    - Random erasing for robustness
    """
    return A.Compose(
        [
            # Resize
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            # Random crop and resize - more aggressive
            A.RandomResizedCrop(
                size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                scale=(0.6, 1.0),
                ratio=(0.8, 1.2),
            ),
            # Horizontal flip
            A.HorizontalFlip(p=0.5),
            # Vertical flip (screen photos can be rotated)
            A.VerticalFlip(p=0.2),
            # Stronger rotation (screen photos may be tilted)
            A.Rotate(limit=30, p=0.6),
            # Perspective transform (simulates viewing angle for screen photos)
            A.Perspective(scale=(0.05, 0.15), p=0.5),
            # Affine transform
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                shear=(-5, 5),
                p=0.3,
            ),
            # Color jitter - stronger
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.15,
                p=0.6,
            ),
            # Random brightness contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            # CLAHE (enhance local contrast)
            A.CLAHE(clip_limit=2.0, p=0.2),
            # Gaussian blur (simulates out-of-focus)
            A.GaussianBlur(blur_limit=(3, 9), p=0.4),
            # Motion blur (simulates camera shake when capturing screen)
            A.MotionBlur(blur_limit=11, p=0.3),
            # Median blur
            A.MedianBlur(blur_limit=5, p=0.1),
            # JPEG compression artifacts (common in screen photos)
            A.ImageCompression(
                quality_range=(30, 95),
                p=0.4,
            ),
            # Noise - stronger (simulates camera sensor noise)
            A.GaussNoise(
                std_range=(0.1, 0.4),
                mean_range=(0.0, 0.0),
                p=0.4,
            ),
            # ISO noise (simulates high ISO in dark environments)
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=0.2,
            ),
            # Random grid distortion (simulates screen curvature)
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.1,
                p=0.2,
            ),
            # Elastic transform (simulates flexible screen)
            A.ElasticTransform(
                alpha=50,
                sigma=5,
                p=0.1,
            ),
            # Coarse dropout (random erasing for robustness)
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                p=0.3,
            ),
            # Cutout (larger holes for robustness)
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(16, 64),
                hole_width_range=(16, 64),
                fill=0,
                p=0.2,
            ),
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
