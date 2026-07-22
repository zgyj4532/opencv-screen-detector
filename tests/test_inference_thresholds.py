"""Tests for configurable inference post-processing thresholds."""

from types import SimpleNamespace

from inference.config import configure, settings
from inference.predictor import PredictTask


def test_screen_photo_threshold_changes_postprocessing(monkeypatch) -> None:
    probabilities = {
        "natural": 0.15,
        "screenshot": 0.45,
        "screen_photo": 0.40,
    }
    monkeypatch.setattr(
        PredictTask,
        "run_single_stage",
        lambda _self: {
            "class": "screenshot",
            "confidence": 0.45,
            "probabilities": probabilities,
        },
    )
    task = object.__new__(PredictTask)
    task.models = SimpleNamespace(model_available=True)
    original_threshold = settings.screen_photo_threshold

    try:
        configure(screen_photo_threshold=0.35)
        assert task.run()["class"] == "screen_photo"

        configure(screen_photo_threshold=0.60)
        assert task.run()["class"] == "screenshot"
    finally:
        configure(screen_photo_threshold=original_threshold)
