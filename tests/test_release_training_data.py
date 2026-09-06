"""Tests for the reproducible release-training data pipeline."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiment.cnn_fft_dwt_ablation.harness import (
    FocalLossLS,
    _capture_runtime_rng,
    _restore_runtime_rng,
    build_split,
    checkpoint_selection_key,
    collect_samples,
    distillation_kl,
    load_init_checkpoint,
    make_sampler,
    mix_modal_batch,
    mixed_focal_loss,
    remix_label_lambda,
    train_tf,
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


def test_collect_samples_deduplicates_identical_content(tmp_path: Path) -> None:
    screenshot = _touch_images(tmp_path, "screenshot", 1)[0]
    duplicate = _touch_images(tmp_path, "hard_negative/ide_editor", 1)[0]
    duplicate.write_bytes(screenshot.read_bytes())

    main, hard = collect_samples(tmp_path, label_overrides={})

    assert main == []
    assert hard == [(str(screenshot), 1)]


def test_collect_samples_requires_explicit_override_for_conflicting_content(tmp_path: Path) -> None:
    natural = _touch_images(tmp_path, "natural_photo", 1)[0]
    screenshot = _touch_images(tmp_path, "screenshot", 1)[0]
    screenshot.write_bytes(natural.read_bytes())

    with pytest.raises(RuntimeError, match="Conflicting labels for content"):
        collect_samples(tmp_path, label_overrides={})

    digest = hashlib.sha256(natural.read_bytes()).hexdigest()
    main, hard = collect_samples(tmp_path, label_overrides={digest: 1})

    assert main == [(str(screenshot), 1)]
    assert hard == []


def test_build_split_refreshes_for_data_changes_and_keeps_canary_in_train(
    tmp_path: Path,
) -> None:
    _touch_images(tmp_path, "natural_photo", 10)
    screenshots = _touch_images(tmp_path, "screenshot", 10)
    _touch_images(tmp_path, "screen_photo", 10)
    split_path = tmp_path / "split.json"
    canaries = {str(screenshots[-1].resolve())}

    first = build_split(seed=42, data_dir=tmp_path, split_path=split_path, canary_paths=canaries)
    first_paths = {str(Path(path).resolve()) for path, _ in first["train"]}
    assert canaries <= first_paths
    assert len(first["train"] + first["val"] + first["test"]) == 30

    _touch_images(tmp_path, "screenshot", 11)
    second = build_split(seed=42, data_dir=tmp_path, split_path=split_path, canary_paths=canaries)

    assert len(second["train"] + second["val"] + second["test"]) == 31
    assert second["meta"]["dataset_fingerprint"] != first["meta"]["dataset_fingerprint"]
    assert second["val"] == first["val"]
    assert second["test"] == first["test"]
    assert any(Path(path).name == "screenshot_10.png" for path, _ in second["train"])

    stored = json.loads(split_path.read_text(encoding="utf-8"))
    assert stored["meta"]["schema_version"] == 4
    assert all(not Path(sample[0]).is_absolute() for role in ("train", "val", "test") for sample in stored[role])


def test_build_split_keeps_content_with_any_hard_negative_copy_in_train(tmp_path: Path) -> None:
    _touch_images(tmp_path, "natural_photo", 10)
    screenshots = _touch_images(tmp_path, "screenshot", 10)
    _touch_images(tmp_path, "screen_photo", 10)
    hard_copy = _touch_images(tmp_path, "hard_negative/ide_editor", 1)[0]
    hard_copy.write_bytes(screenshots[-1].read_bytes())

    split = build_split(seed=42, data_dir=tmp_path, split_path=tmp_path / "split.json", canary_paths=set())
    digest = hashlib.sha256(hard_copy.read_bytes()).hexdigest()

    assert any(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, _label in split["train"])
    assert all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
        for role in ("val", "test")
        for path, _label in split[role]
    )


def test_canary_fingerprint_is_portable_across_dataset_roots(tmp_path: Path) -> None:
    fingerprints = []
    for root_name in ("first", "second"):
        root = tmp_path / root_name
        _touch_images(root, "natural_photo", 10)
        screenshots = _touch_images(root, "screenshot", 10)
        _touch_images(root, "screen_photo", 10)
        split = build_split(
            seed=42,
            data_dir=root,
            split_path=root / "split.json",
            canary_paths={str(screenshots[-1].resolve())},
        )
        fingerprints.append(split["meta"]["canary_fingerprint"])

    assert fingerprints[0] == fingerprints[1]


def test_build_split_rejects_canary_outside_collected_dataset(tmp_path: Path) -> None:
    _touch_images(tmp_path, "natural_photo", 3)
    _touch_images(tmp_path, "screenshot", 3)
    _touch_images(tmp_path, "screen_photo", 3)
    uncollected = tmp_path / "unmapped" / "canary.png"
    uncollected.parent.mkdir(parents=True)
    uncollected.write_bytes(b"fixture")

    with pytest.raises(RuntimeError, match="not part of the collected dataset"):
        build_split(
            seed=42,
            data_dir=tmp_path,
            split_path=tmp_path / "split.json",
            canary_paths={str(uncollected.resolve())},
        )


def test_make_sampler_upweights_canary_within_the_same_class(tmp_path: Path) -> None:
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
        canary_paths={str(screenshots[0].resolve())},
        canary_weight=4.0,
    )

    assert sampler.weights[0].item() == pytest.approx(4 * sampler.weights[1].item())


def test_checkpoint_selection_does_not_use_canary_outcome() -> None:
    metrics = {"best_metric": 0.91}

    assert checkpoint_selection_key({**metrics, "canary_pass": True}) == checkpoint_selection_key(
        {**metrics, "canary_pass": False}
    )


def test_make_sampler_repeats_the_same_sequence_for_the_same_seed(tmp_path: Path) -> None:
    screenshots = _touch_images(tmp_path, "screenshot", 3)
    naturals = _touch_images(tmp_path, "natural_photo", 3)
    samples = [(str(path), 1) for path in screenshots] + [(str(path), 0) for path in naturals]

    first = list(make_sampler(samples, beta=None, seed=42))
    second = list(make_sampler(samples, beta=None, seed=42))

    assert first == second


def test_train_transform_repeats_the_same_sequence_for_the_same_seed() -> None:
    image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    first = train_tf(seed=42)
    second = train_tf(seed=42)

    first_outputs = [first(image=image)["image"].numpy() for _ in range(3)]
    second_outputs = [second(image=image)["image"].numpy() for _ in range(3)]

    for first_output, second_output in zip(first_outputs, second_outputs, strict=True):
        np.testing.assert_array_equal(first_output, second_output)


def test_training_rng_state_can_resume_exactly(tmp_path: Path) -> None:
    images = _touch_images(tmp_path, "screenshot", 6)
    samples = [(str(path), 1) for path in images]
    transform = train_tf(seed=42)
    sampler = make_sampler(samples, beta=None, seed=42)
    image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    state = _capture_runtime_rng(transform, sampler)

    expected_indices = list(sampler)
    expected_images = [transform(image=image)["image"].numpy() for _ in range(3)]

    _restore_runtime_rng(state, transform, sampler)
    actual_indices = list(sampler)
    actual_images = [transform(image=image)["image"].numpy() for _ in range(3)]

    assert actual_indices == expected_indices
    for actual, expected in zip(actual_images, expected_images, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_remix_label_lambda_assigns_minority_when_ratio_and_tau_match() -> None:
    # screenshot=1221, screen_photo=345 → ratio 3.54 >= kappa=3
    counts = torch.tensor([646.0, 1221.0, 345.0])
    labels_a = torch.tensor([1, 1, 2, 0])
    labels_b = torch.tensor([2, 2, 1, 1])
    lam = torch.tensor([0.4, 0.7, 0.6, 0.4])

    lam_y = remix_label_lambda(labels_a, labels_b, lam, counts, kappa=3.0, tau=0.5)

    torch.testing.assert_close(lam_y, torch.tensor([0.0, 0.7, 1.0, 0.4]))


def test_mix_modal_batch_interpolates_all_three_inputs() -> None:
    rgb = torch.tensor([1.0, 3.0]).view(2, 1, 1, 1)
    fft = torch.tensor([10.0, 30.0]).view(2, 1, 1, 1)
    dwt = torch.tensor([100.0, 300.0]).view(2, 1, 1, 1)
    index = torch.tensor([1, 0])
    lam = torch.tensor([0.25, 0.25])

    mixed_rgb, mixed_fft, mixed_dwt = mix_modal_batch(rgb, fft, dwt, index, lam)

    torch.testing.assert_close(mixed_rgb.reshape(2), torch.tensor([2.5, 1.5]))
    torch.testing.assert_close(mixed_fft.reshape(2), torch.tensor([25.0, 15.0]))
    torch.testing.assert_close(mixed_dwt.reshape(2), torch.tensor([250.0, 150.0]))


def test_mixed_focal_loss_matches_hard_labels_at_lambda_endpoints() -> None:
    criterion = FocalLossLS(gamma=2.0, alpha=[1.0, 1.0, 1.5], smoothing=0.05)
    logits = torch.tensor([[2.0, 0.1, -1.0], [0.2, 1.5, 0.3]])
    labels_a = torch.tensor([0, 1])
    labels_b = torch.tensor([2, 2])

    loss_a = criterion(logits, labels_a)
    loss_b = criterion(logits, labels_b)
    mixed_all_a = mixed_focal_loss(criterion, logits, labels_a, labels_b, torch.ones(2))
    mixed_all_b = mixed_focal_loss(criterion, logits, labels_a, labels_b, torch.zeros(2))

    torch.testing.assert_close(mixed_all_a, loss_a)
    torch.testing.assert_close(mixed_all_b, loss_b)
    per_sample = criterion(logits, labels_a, reduction="none")
    assert per_sample.shape == (2,)


def test_make_sampler_upweights_boost_paths_within_the_same_class(tmp_path: Path) -> None:
    screenshots = _touch_images(tmp_path, "screenshot", 2)
    samples = [(str(screenshots[0]), 1), (str(screenshots[1]), 1)]

    sampler = make_sampler(
        samples,
        beta=None,
        boost_paths={str(screenshots[0].resolve())},
        boost_weight=3.0,
    )

    assert sampler.weights[0].item() == pytest.approx(3 * sampler.weights[1].item())


def test_distillation_kl_is_zero_for_identical_logits() -> None:
    logits = torch.tensor([[1.0, 2.0, 0.5], [0.1, -0.2, 0.3]])
    loss = distillation_kl(logits, logits.clone(), temperature=2.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_load_init_checkpoint_reads_model_state_dict(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 3)
    with torch.no_grad():
        model.weight.fill_(0.25)
        model.bias.fill_(-0.5)
    path = tmp_path / "init.pth"
    torch.save({"model_state_dict": model.state_dict()}, path)

    loaded = torch.nn.Linear(4, 3)
    load_init_checkpoint(loaded, path)

    torch.testing.assert_close(loaded.weight, model.weight)
    torch.testing.assert_close(loaded.bias, model.bias)
