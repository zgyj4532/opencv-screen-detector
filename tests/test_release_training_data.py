"""Tests for the reproducible release-training data pipeline."""

from pathlib import Path

import pytest

from experiment.cnn_fft_dwt_ablation.harness import (
    build_split,
    collect_samples,
    make_sampler,
)


def _touch_images(root: Path, folder: str, count: int) -> list[Path]:
    directory = root / folder
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = directory / f"{folder.replace('/', '_')}_{index}.png"
        path.write_bytes(f"fixture-{folder}-{index}".encode())
        paths.append(path)
    return paths


def test_collect_samples_uses_true_hard_negative_labels(tmp_path: Path) -> None:
    _touch_images(tmp_path, "natural_photo", 1)
    _touch_images(tmp_path, "screenshot", 1)
    _touch_images(tmp_path, "screen_photo", 1)
    _touch_images(tmp_path, "hard_negative/natural_to_screenshot", 1)
    _touch_images(tmp_path, "hard_negative/screenshot_to_screen_photo", 1)
    _touch_images(tmp_path, "hard_negative/screen_photo_to_screenshot", 1)

    main, hard = collect_samples(tmp_path)

    assert sorted(label for _, label in main) == [0, 1, 2]
    assert sorted(label for _, label in hard) == [0, 1, 2]
    assert len({str(Path(path).resolve()) for path, _ in main + hard}) == 6


def test_build_split_refreshes_for_data_changes_and_keeps_focus_in_train(
    tmp_path: Path,
) -> None:
    _touch_images(tmp_path, "natural_photo", 10)
    screenshots = _touch_images(tmp_path, "screenshot", 10)
    _touch_images(tmp_path, "screen_photo", 10)
    split_path = tmp_path / "split.json"
    focus = {str(screenshots[-1].resolve())}

    first = build_split(seed=42, data_dir=tmp_path, split_path=split_path, focus_paths=focus)
    first_paths = {str(Path(path).resolve()) for path, _ in first["train"]}
    assert focus <= first_paths
    assert len(first["train"] + first["val"] + first["test"]) == 30

    _touch_images(tmp_path, "screenshot", 11)
    second = build_split(seed=42, data_dir=tmp_path, split_path=split_path, focus_paths=focus)

    assert len(second["train"] + second["val"] + second["test"]) == 31
    assert second["meta"]["dataset_fingerprint"] != first["meta"]["dataset_fingerprint"]


def test_build_split_rejects_focus_outside_collected_dataset(tmp_path: Path) -> None:
    _touch_images(tmp_path, "natural_photo", 3)
    _touch_images(tmp_path, "screenshot", 3)
    _touch_images(tmp_path, "screen_photo", 3)
    uncollected = tmp_path / "unmapped" / "focus.png"
    uncollected.parent.mkdir(parents=True)
    uncollected.write_bytes(b"fixture")

    with pytest.raises(RuntimeError, match="not part of the collected dataset"):
        build_split(
            seed=42,
            data_dir=tmp_path,
            split_path=tmp_path / "split.json",
            focus_paths={str(uncollected.resolve())},
        )


def test_make_sampler_upweights_focus_within_the_same_class(tmp_path: Path) -> None:
    screenshots = _touch_images(tmp_path, "screenshot", 2)
    naturals = _touch_images(tmp_path, "natural_photo", 2)
    samples = [
        (str(screenshots[0]), 1),
        (str(screenshots[1]), 1),
        (str(naturals[0]), 0),
        (str(naturals[1]), 0),
    ]

    sampler = make_sampler(
        samples,
        beta=None,
        focus_paths={str(screenshots[0].resolve())},
        focus_weight=4.0,
    )

    assert sampler.weights[0].item() == pytest.approx(4 * sampler.weights[1].item())
