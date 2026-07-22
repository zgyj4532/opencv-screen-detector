"""Local-data regression tests for confirmed screenshot misclassifications."""

from pathlib import Path

import pytest

from inference.predictor import ScreenDetectorPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_SCREENSHOTS = [
    "4a6e9f5ce9dc3506cd4dd4484f0eb79b3b9a90551909f5363687a4438baae8f9.png",
    "5cdc3e8dca1eab9bbd5362b995e92c7c171621dbc827ca2af01bb7eaa4123a62.png",
]


@pytest.fixture(scope="module")
def deployed_predictor() -> ScreenDetectorPredictor:
    return ScreenDetectorPredictor()


@pytest.mark.parametrize("filename", TARGET_SCREENSHOTS)
def test_confirmed_screenshot_regressions(
    deployed_predictor: ScreenDetectorPredictor,
    filename: str,
) -> None:
    image_path = PROJECT_ROOT / "data" / "input" / "screenshot" / filename
    if not image_path.exists():
        pytest.skip("Private local screenshot fixture is not available")

    result = deployed_predictor.predict(image_path)

    assert result["class"] == "screenshot", (
        f"{filename}: expected screenshot, got {result['class']} with probabilities={result['probabilities']}"
    )
