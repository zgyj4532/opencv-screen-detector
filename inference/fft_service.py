"""FFT and DWT spectrum service with LRU cache."""

import functools
import hashlib
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .config import settings
from .fft_transform import compute_dwt_features, compute_fft_spectrum

_DEFAULT_CACHE_SIZE = 128


class FFTService:
    """Computes and caches FFT spectra and DWT features for image files.

    Uses LRU eviction to bound memory usage.

    Returns:
        Tuple of (fft_spectrum, dwt_features) for each image
    """

    def __init__(self, max_size: int = _DEFAULT_CACHE_SIZE) -> None:
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._max_size = max_size

    def get_fft_input(self, image_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Get FFT spectrum and DWT features for an image file, with caching.

        Returns:
            Tuple of (fft_spectrum, dwt_features)
        """
        cache_key = self._hash_file(image_path)

        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        fft_spectrum = compute_fft_spectrum(image, settings.image_size)
        dwt_features = compute_dwt_features(image, settings.image_size)
        result = (fft_spectrum, dwt_features)
        self._put(cache_key, result)
        return result

    def get_fft_input_from_array(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get FFT spectrum and DWT features for a numpy array (no caching).

        Returns:
            Tuple of (fft_spectrum, dwt_features)
        """
        fft_spectrum = compute_fft_spectrum(image, settings.image_size)
        dwt_features = compute_dwt_features(image, settings.image_size)
        return fft_spectrum, dwt_features

    def _put(self, key: str, value: tuple[np.ndarray, np.ndarray]) -> None:
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = value

    @staticmethod
    @functools.lru_cache(maxsize=16)
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def clear(self) -> None:
        self._cache.clear()
