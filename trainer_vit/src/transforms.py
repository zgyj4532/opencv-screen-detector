"""ViT data transforms for screen detector.

Standard ImageNet normalization for ViT-Small.
Includes Mixup and CutMix augmentation support.
"""

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2 as transforms_v2

# ImageNet normalization (ViT pretrained on ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Get training transforms with augmentation.

    Args:
        image_size: Target image size (default 224 for ViT)

    Returns:
        Albumentations compose transform
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Get validation transforms (no augmentation).

    Args:
        image_size: Target image size (default 224 for ViT)

    Returns:
        Albumentations compose transform
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def get_inference_transforms(image_size: int = 224) -> A.Compose:
    """Get inference transforms (same as validation).

    Args:
        image_size: Target image size (default 224 for ViT)

    Returns:
        Albumentations compose transform
    """
    return get_val_transforms(image_size)


def get_mixup_cutmix(
    num_classes: int = 3,
) -> transforms_v2.MixUp | transforms_v2.CutMix:
    """Get Mixup or CutMix transform (randomly selected).

    Args:
        num_classes: Number of classes

    Returns:
        MixUp or CutMix transform
    """
    # Randomly choose between Mixup and CutMix
    if np.random.random() < 0.5:
        return transforms_v2.MixUp(alpha=0.8, num_classes=num_classes)
    return transforms_v2.CutMix(alpha=1.0, num_classes=num_classes)


class MixUpCutMixWrapper:
    """Wrapper to apply Mixup or CutMix to batched data.

    Args:
        num_classes: Number of classes
        prob: Probability of applying augmentation
    """

    def __init__(self, num_classes: int = 3, prob: float = 0.5) -> None:
        self.num_classes = num_classes
        self.prob = prob
        self.mixup = transforms_v2.MixUp(alpha=0.8, num_classes=num_classes)
        self.cutmix = transforms_v2.CutMix(alpha=1.0, num_classes=num_classes)

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Mixup or CutMix to batch.

        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)

        Returns:
            Mixed images and labels (one-hot encoded)
        """
        if np.random.random() < self.prob:
            if np.random.random() < 0.5:
                return self.mixup(images, labels)
            return self.cutmix(images, labels)
        # Convert labels to one-hot without augmentation
        one_hot = torch.zeros(labels.size(0), self.num_classes, device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        return images, one_hot
