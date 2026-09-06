"""Tests for governed evaluation sets, isolation, and separated metrics."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from trainer.evaluation_sets import (
    FAIL,
    GROUP_KEYS,
    NOT_READY,
    READY,
    REVIEW_REQUIRED,
    audit_split_isolation,
    evaluate_known_and_ood,
    evaluation_set_readiness,
    perceptual_hash,
    promotion_readiness,
)


def _write_pattern(path: Path, *, jpeg_quality: int | None = None) -> None:
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    cv2.rectangle(image, (12, 10), (112, 76), (240, 240, 240), -1)
    cv2.line(image, (18, 28), (105, 28), (20, 20, 20), 4)
    cv2.circle(image, (65, 54), 15, (80, 80, 80), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if jpeg_quality is not None else []
    assert cv2.imwrite(str(path), image, params)


def _metadata(*paths: Path, shared_source: bool = False) -> dict[str, dict[str, str]]:
    result = {}
    for index, path in enumerate(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[digest] = {
            key: ("shared-source" if shared_source and key == "source_content_id" else f"{key}-{index}")
            for key in GROUP_KEYS
        }
    return result


def test_repository_evaluation_set_readiness_is_truthful() -> None:
    readiness = evaluation_set_readiness()

    assert readiness["canary"]["status"] == READY
    assert readiness["canary"]["entries"] == 2
    assert readiness["frozen_challenge"]["status"] == NOT_READY
    assert readiness["ood"]["status"] == NOT_READY
    assert readiness["rolling_error_pool"]["status"] == READY


def test_known_false_rejection_is_not_reported_as_true_ood() -> None:
    known_probs = np.array([[0.90, 0.05, 0.05], [0.40, 0.35, 0.25], [0.05, 0.90, 0.05]])
    known_labels = np.array([0, 1, 1])

    report = evaluate_known_and_ood(known_probs, known_labels, ood_threshold=0.45)

    assert report["closed_set_classification"]["samples"] == 3
    assert report["known_class_false_rejection"]["false_rejection_count"] == 1
    assert report["true_ood_detection"]["status"] == NOT_READY


def test_true_ood_metrics_are_reported_only_with_labeled_ood() -> None:
    known_probs = np.array([[0.90, 0.05, 0.05], [0.05, 0.90, 0.05]])
    known_labels = np.array([0, 1])
    ood_probs = np.array([[0.34, 0.33, 0.33], [0.40, 0.30, 0.30]])

    report = evaluate_known_and_ood(known_probs, known_labels, ood_probs, ood_threshold=0.45)

    assert report["true_ood_detection"]["status"] == READY
    assert report["true_ood_detection"]["true_positive_rate"] == pytest.approx(1.0)
    assert report["true_ood_detection"]["known_false_positive_rate"] == pytest.approx(0.0)
    assert [point["coverage"] for point in report["selective_risk_coverage"]["points"]] == sorted(
        [point["coverage"] for point in report["selective_risk_coverage"]["points"]], reverse=True
    )


def test_group_metadata_overlap_fails_split_isolation(tmp_path: Path) -> None:
    train = tmp_path / "train.png"
    test = tmp_path / "test.png"
    _write_pattern(train)
    image = cv2.imread(str(train))
    image[0, 0] = (1, 2, 3)
    assert cv2.imwrite(str(test), image)
    metadata = _metadata(train, test, shared_source=True)
    split = {"train": [(str(train), 1)], "val": [], "test": [(str(test), 2)]}

    report = audit_split_isolation(split, metadata)

    assert report["status"] == FAIL
    assert report["group_metadata"]["cross_role_overlaps"]["source_content_id"]


def test_phash_flags_cross_split_near_duplicates_for_review(tmp_path: Path) -> None:
    train = tmp_path / "train.png"
    test = tmp_path / "test.jpg"
    _write_pattern(train)
    _write_pattern(test, jpeg_quality=65)
    metadata = _metadata(train, test)
    split = {"train": [(str(train), 1)], "val": [], "test": [(str(test), 1)]}

    distance = (perceptual_hash(train) ^ perceptual_hash(test)).bit_count()
    report = audit_split_isolation(split, metadata, phash_threshold=8)

    assert distance <= 8
    assert report["status"] == REVIEW_REQUIRED
    assert report["perceptual_similarity"]["cross_role_candidate_pairs"] == 1


def test_promotion_requires_evaluated_challenge_and_true_ood() -> None:
    ready_sets = {
        "frozen_challenge": {"status": READY},
        "ood": {"status": READY},
    }

    unevaluated = promotion_readiness(ready_sets, READY, canary_pass=True)
    accepted = promotion_readiness(
        ready_sets,
        READY,
        canary_pass=True,
        frozen_challenge_pass=True,
        true_ood_pass=True,
    )

    assert unevaluated["status"] == NOT_READY
    assert accepted["status"] == READY
