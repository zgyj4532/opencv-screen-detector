"""Local-data regression tests for confirmed screenshot misclassifications."""

import pytest

from inference.predictor import ScreenDetectorPredictor
from trainer.evaluation_sets import EvaluationSample, resolve_manifest_samples

CANARY_SAMPLES = resolve_manifest_samples("canary", require_files=False)


@pytest.fixture(scope="module")
def deployed_predictor() -> ScreenDetectorPredictor:
    return ScreenDetectorPredictor()


@pytest.mark.parametrize("sample", CANARY_SAMPLES, ids=lambda sample: sample.sample_id)
def test_confirmed_screenshot_regressions(
    deployed_predictor: ScreenDetectorPredictor,
    sample: EvaluationSample,
) -> None:
    if not sample.path.exists():
        pytest.skip("Private local screenshot fixture is not available")

    result = deployed_predictor.predict(sample.path)

    assert result["class"] == sample.expected_label, (
        f"{sample.sample_id}: expected {sample.expected_label}, got {result['class']} "
        f"with probabilities={result['probabilities']}"
    )
