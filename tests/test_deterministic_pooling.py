"""Regression tests for deterministic pooling used by CUDA release training."""

import torch
from torch.nn import functional

from trainer.fft_branch import POOL_KERNEL_SIZE
from trainer.model import DeterministicGlobalAvgPool2d


def test_fixed_frequency_pool_matches_adaptive_pool_for_release_shape() -> None:
    inputs = torch.arange(2 * 3 * 224 * 224, dtype=torch.float32).reshape(2, 3, 224, 224)

    fixed = functional.avg_pool2d(inputs, kernel_size=POOL_KERNEL_SIZE, stride=POOL_KERNEL_SIZE)
    adaptive = functional.adaptive_avg_pool2d(inputs, output_size=4)

    torch.testing.assert_close(fixed, adaptive)


def test_deterministic_global_pool_matches_adaptive_pool() -> None:
    inputs = torch.arange(2 * 3 * 7 * 7, dtype=torch.float32).reshape(2, 3, 7, 7)

    reduced = DeterministicGlobalAvgPool2d()(inputs)
    adaptive = functional.adaptive_avg_pool2d(inputs, output_size=1).flatten(1)

    torch.testing.assert_close(reduced, adaptive)
