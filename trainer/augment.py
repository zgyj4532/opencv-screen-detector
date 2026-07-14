"""Data augmentation for screen detector V3 training.

Enhanced augmentation strategy for better screen_photo detection.
Includes moiré simulation and screen reflection augmentation.
"""

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from . import config


class MoireSimulation(A.ImageOnlyTransform):
    """Simulate moiré patterns commonly seen in screen photos.

    Moiré patterns occur when photographing screens due to interference
    between the screen's pixel grid and the camera's sensor grid.

    Args:
        frequency_range: Range of moiré pattern frequency
        amplitude_range: Range of moiré pattern intensity
        p: Probability of applying the transform
    """

    def __init__(
        self,
        frequency_range: tuple[float, float] = (0.1, 0.5),
        amplitude_range: tuple[float, float] = (0.05, 0.2),
        p: float = 0.3,
    ):
        super().__init__(p=p)
        self.frequency_range = frequency_range
        self.amplitude_range = amplitude_range

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        freq = np.random.uniform(*self.frequency_range)
        amp = np.random.uniform(*self.amplitude_range)

        # Create moiré pattern
        x = np.arange(w, dtype=np.float32)
        y = np.arange(h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        # Random angle for moiré orientation
        angle = np.random.uniform(0, np.pi)
        pattern = np.sin(2 * np.pi * freq * (xx * np.cos(angle) + yy * np.sin(angle)))

        # Apply moiré to image
        pattern = pattern[:, :, np.newaxis]  # (H, W, 1)
        img_float = img.astype(np.float32)
        img_moire = img_float + amp * 255 * pattern
        return np.clip(img_moire, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("frequency_range", "amplitude_range")


class ScreenReflection(A.ImageOnlyTransform):
    """Simulate screen reflection/glare artifacts.

    Screen photos often have reflections from ambient light sources,
    creating bright spots and gradients on the screen.

    Args:
        num_spots_range: Range of number of reflection spots
        intensity_range: Range of reflection intensity
        p: Probability of applying the transform
    """

    def __init__(
        self,
        num_spots_range: tuple[int, int] = (1, 3),
        intensity_range: tuple[float, float] = (0.3, 0.8),
        p: float = 0.3,
    ):
        super().__init__(p=p)
        self.num_spots_range = num_spots_range
        self.intensity_range = intensity_range

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        num_spots = np.random.randint(*self.num_spots_range)

        img_float = img.astype(np.float32)
        mask = np.zeros((h, w), dtype=np.float32)

        for _ in range(num_spots):
            # Random position
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)

            # Random size and intensity
            radius = np.random.randint(min(h, w) // 8, min(h, w) // 3)
            intensity = np.random.uniform(*self.intensity_range)

            # Create Gaussian spot
            x = np.arange(w, dtype=np.float32)
            y = np.arange(h, dtype=np.float32)
            xx, yy = np.meshgrid(x, y)
            spot = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius**2))
            mask += intensity * spot

        # Clip mask
        mask = np.clip(mask, 0, 1)

        # Apply reflection (additive brightening)
        img_reflected = img_float + mask[:, :, np.newaxis] * 100
        return np.clip(img_reflected, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("num_spots_range", "intensity_range")


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
            # Moiré simulation (screen photo artifact)
            MoireSimulation(
                frequency_range=(0.1, 0.5),
                amplitude_range=(0.05, 0.2),
                p=0.3,
            ),
            # Screen reflection/glare simulation
            ScreenReflection(
                num_spots_range=(1, 3),
                intensity_range=(0.3, 0.8),
                p=0.3,
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
