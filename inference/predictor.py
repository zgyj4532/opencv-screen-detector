"""ONNX predictor for screen detector V3 inference.

Single-stage 3-class CNN+FFT+DWT model with TTA, OOD detection, and confidence tiering.
"""

import functools
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .config import settings
from .fft_service import FFTService
from .model_loader import ModelLoader
from .preprocess import normalize_rgb


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _run_stage(
    session: ort.InferenceSession,
    rgb_input: np.ndarray,
    fft_input: np.ndarray,
    dwt_input: np.ndarray,
    class_names: list[str],
) -> dict:
    """Run inference on a single stage and return structured result."""
    rgb_name = session.get_inputs()[0].name
    fft_name = session.get_inputs()[1].name
    dwt_name = session.get_inputs()[2].name
    output_name = session.get_outputs()[0].name

    outputs: Any = session.run(
        [output_name],
        {rgb_name: rgb_input, fft_name: fft_input, dwt_name: dwt_input},
    )

    logits = outputs[0][0]
    probabilities = _softmax(logits)

    class_idx = np.argmax(probabilities)
    probs_dict = {name: float(prob) for name, prob in zip(class_names, probabilities, strict=False)}

    return {
        "class": class_names[class_idx],
        "confidence": float(probabilities[class_idx]),
        "probabilities": probs_dict,
    }


def _get_confidence_tier(confidence: float) -> dict:
    """Get confidence tier and recommended action."""
    if confidence >= settings.confidence_high:
        return {"confidence_tier": "high", "action": "accept"}
    if confidence >= settings.confidence_medium:
        return {"confidence_tier": "medium", "action": "review"}
    return {"confidence_tier": "low", "action": "ignore"}


def _check_ood(probabilities: dict[str, float]) -> bool:
    """Return True if max probability is below the OOD threshold."""
    return max(probabilities.values()) < settings.ood_threshold


class PredictTask:
    def __init__(self, models: ModelLoader, fft: FFTService, image_path: Path) -> None:
        self.models = models
        self.fft = fft
        self.image_path = image_path

    @functools.cached_property
    def original_image(self) -> np.ndarray:
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")
        return image

    @functools.cached_property
    def rgb_input(self) -> np.ndarray:
        return normalize_rgb(self.original_image)

    @functools.cached_property
    def fft_input(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (fft_spectrum, dwt_features)"""
        return self.fft.get_fft_input(self.image_path)

    def run_single_stage(self) -> dict:
        """Single-stage 3-class inference with TTA."""
        if not self.models.model_available:
            raise RuntimeError("Single 3-class model not loaded")

        class_names = settings.class_names
        flipped = cv2.flip(self.original_image, 1)
        flipped_rgb = normalize_rgb(flipped)
        flipped_fft, flipped_dwt = self.fft.get_fft_input_from_array(flipped)

        fft_input, dwt_input = self.fft_input

        with self.models.get_session() as session:
            result = _run_stage(session, self.rgb_input, fft_input, dwt_input, class_names)
            result_flip = _run_stage(session, flipped_rgb, flipped_fft, flipped_dwt, class_names)

        # Average probabilities
        avg_probs: dict[str, float] = {}
        for key in class_names:
            p1 = result["probabilities"][key]
            p2 = result_flip["probabilities"][key]
            avg_probs[key] = float(np.mean([p1, p2]))

        class_idx = np.argmax(list(avg_probs.values()))
        class_name = class_names[class_idx]

        return {
            "class": class_name,
            "confidence": float(avg_probs[class_name]),
            "probabilities": avg_probs,
        }

    def run(self) -> dict:
        """Run single-stage 3-class inference with OOD detection."""
        if not self.models.model_available:
            raise RuntimeError("3-class model not loaded")
        return self._run_single_stage_with_ood()

    def _run_single_stage_with_ood(self) -> dict:
        """Single-stage inference with OOD detection.

        Post-processing logic:
        - If every class probability is below the OOD threshold, return unknown
        - If screen_photo probability reaches its configured threshold, classify as screen_photo
        - Otherwise, use argmax of all classes
        """
        result = self.run_single_stage()
        probs = result["probabilities"]

        if _check_ood(probs):
            max_prob = max(probs.values())
            return {
                "class": "unknown",
                "confidence": float(max_prob),
                "probabilities": probs,
                "confidence_tier": "ood",
                "action": "ignore",
            }

        # Optimized post-processing with a configurable threshold.
        sp_prob = probs.get("screen_photo", 0.0)

        if sp_prob >= settings.screen_photo_threshold:
            # High confidence screen_photo
            result = {
                "class": "screen_photo",
                "confidence": sp_prob,
                "probabilities": probs,
            }
        else:
            # Use argmax for natural/screenshot
            class_idx = np.argmax(list(probs.values()))
            class_name = list(probs.keys())[class_idx]
            result = {
                "class": class_name,
                "confidence": probs[class_name],
                "probabilities": probs,
            }

        return {
            **result,
            **_get_confidence_tier(result["confidence"]),
        }


class ScreenDetectorPredictor:
    """ONNX-based screen detector predictor.

    Single-stage 3-class CNN+FFT model with OOD detection.
    """

    def __init__(
        self,
        model_path: Path | None = None,
    ) -> None:
        m = model_path or settings.model_path
        self._models = ModelLoader(model_path=m)
        self._fft = FFTService()

    # -- Properties --

    @property
    def model_available(self) -> bool:
        """Check if 3-class model is available."""
        return self._models.model_available

    # -- Prediction --

    def predict(self, image_path: Path) -> dict:
        """Single-stage prediction with OOD detection.

        Returns:
            dict with keys: class, confidence, probabilities,
            confidence_tier, action
        """
        return PredictTask(self._models, self._fft, image_path).run()

    def predict_batch(self, image_paths: list[Path]) -> list[dict]:
        """Predict on multiple images."""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result["filename"] = Path(image_path).name
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "filename": Path(image_path).name,
                        "error": str(e),
                    }
                )
        return results
