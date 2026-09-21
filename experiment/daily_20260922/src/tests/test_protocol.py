"""Protect the experiment's statistical and evidence boundaries."""

import numpy as np
import pytest

from experiment.daily_20260922.src.compare import metric_samples, paired_comparison
from experiment.daily_20260922.src.evaluate import metrics
from experiment.daily_20260922.src.protocol import SEEDS, VARIANTS, candidate_name, read_json, write_new
from experiment.daily_20260922.src.select import rank_variants


def validation_reports():
    values = {"baseline": (0.91, 0.91), "low_lr": (0.98, 0.80), "lwf": (0.99, 0.99)}
    return {
        candidate_name(variant, seed): {
            "role": "val",
            "canary": {"correct": 2, "total": 2},
            "metrics": {"metric": values[variant][index]},
        }
        for variant in VARIANTS
        for index, seed in enumerate(SEEDS)
    }


def test_selection_uses_seed_mean_and_requires_every_canary():
    reports = validation_reports()
    reports[candidate_name("lwf", 2024)]["canary"]["correct"] = 1
    ranking = rank_variants(reports)
    eligible = [row for row in ranking if row["eligible"]]
    assert eligible[0]["variant"] == "baseline"
    assert eligible[1]["variant"] == "low_lr"
    assert not ranking[0]["eligible"]


def test_test_results_cannot_enter_selection():
    reports = validation_reports()
    reports[candidate_name("baseline", 42)]["role"] = "test"
    with pytest.raises(ValueError, match="validation reports only"):
        rank_variants(reports)


def test_empty_canary_is_not_a_pass():
    reports = validation_reports()
    reports[candidate_name("lwf", 42)]["canary"] = {"correct": 0, "total": 0}
    row = next(row for row in rank_variants(reports) if row["variant"] == "lwf")
    assert not row["eligible"]


def test_unknown_remains_in_closed_set_denominator():
    result = metrics([0, 1, 2, 2], [0, 1, 2, -1])
    assert result["accuracy"] == 0.75
    assert result["unknown"] == 1
    assert result["sp_recall"] == 0.5
    assert result["sp_f1"] == pytest.approx(2 / 3)


def test_existing_evidence_cannot_be_overwritten(tmp_path):
    target = tmp_path / "selection.json"
    write_new(target, {"winner": "baseline"})
    with pytest.raises(FileExistsError):
        write_new(target, {"winner": "lwf"})
    assert read_json(target) == {"winner": "baseline"}


def test_bootstrap_metric_matches_sklearn_with_unknowns():
    labels = np.asarray([[0, 1, 2, 2], [0, 0, 0, 0]])
    predictions = np.asarray([[0, 1, 2, -1], [-1, 0, 0, -1]])
    expected = [metrics(y, pred)["metric"] for y, pred in zip(labels, predictions, strict=True)]
    np.testing.assert_allclose(metric_samples(labels, predictions), expected)


def test_pairing_rejects_changed_image_identity():
    candidate = {"records": [{"path": "a.jpg", "label": 2}]}
    baseline = {"records": [{"path": "b.jpg", "label": 2}]}
    with pytest.raises(ValueError, match="identical labeled images"):
        paired_comparison(candidate, baseline)
