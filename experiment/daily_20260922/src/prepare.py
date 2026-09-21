"""Extract requested screen photos, record provenance, and freeze the experiment."""

import hashlib
import shutil
import zipfile
from collections import Counter

import cv2
import numpy as np

from experiment.daily_20260922.src.protocol import (
    DATA,
    EPOCHS,
    EXPERIMENT,
    ROOT,
    SEEDS,
    THRESHOLDS,
    VARIANTS,
    digest,
    prepare_directories,
    read_json,
    write_new,
)
from trainer.ingest_screen_photo import IMAGE_EXTS, ingest_screen_photo_zips, is_screen_photo_image_member


def main():
    prepare_directories()
    manifest_path = DATA / "ingest.json"
    if manifest_path.exists():
        raise RuntimeError("Ingestion is already recorded; use the existing manifest")
    archive_directory = EXPERIMENT / "archive"
    archive_directory.mkdir(parents=True, exist_ok=True)
    for source, target in (
        (ROOT / "trainer/data_audit.json", archive_directory / "preexisting_data_audit.json"),
        (ROOT / "experiment/cnn_fft_dwt_ablation/split.json", archive_directory / "preexisting_split.json"),
    ):
        if not target.exists():
            shutil.copy2(source, target)
    destination = ROOT / "data/input/screen_photo"
    archives = [ROOT / f"data/daily-package-2026-09-{day}.zip" for day in range(15, 22)]
    before = {
        digest(path): path
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    archive_rows, records = [], []
    # Validate every requested source before extracting anything.
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            target_count = 0
            for info in archive.infolist():
                if info.is_dir() or not is_screen_photo_image_member(info.filename):
                    continue
                payload = archive.read(info)
                rgb = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if rgb is None:
                    raise RuntimeError(f"Undecodable image: {path.name}:{info.filename}")
                target_count += 1
                records.append(
                    {
                        "archive": path.relative_to(ROOT).as_posix(),
                        "member": info.filename,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                        "width": rgb.shape[1],
                        "height": rgb.shape[0],
                    }
                )
            archive_rows.append(
                {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path), "screen_photos": target_count}
            )
    result = ingest_screen_photo_zips(archives, destination)
    after = {
        digest(path): path
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    for row in records:
        row["target"] = after[row["sha256"]].relative_to(ROOT).as_posix()
        row["new_content"] = row["sha256"] not in before
    manifest = {
        "archives": archive_rows,
        "target_members": len(records),
        "unique_target_content": len({row["sha256"] for row in records}),
        "new_unique_content": result["extracted_count"],
        "duplicate_members": len(result["skipped_duplicate"]),
        "skipped_other_members": result["skipped_other_count"],
        "records": records,
        "original_archives_preserved": all(digest(ROOT / row["path"]) == row["sha256"] for row in archive_rows),
    }
    write_new(manifest_path, manifest)
    from experiment.cnn_fft_dwt_ablation import harness
    from trainer.data_audit import build_report

    # Preserve the pre-existing tracked audit; publish this run's snapshot separately.
    report = build_report(42)
    write_new(DATA / "audit.json", report)
    split = read_json(harness.SPLIT_PATH)
    previous = read_json(EXPERIMENT / "archive/preexisting_split.json")
    for role in ("val", "test"):
        if split[role] != previous[role]:
            raise RuntimeError(f"Frozen {role} identities changed")
    active_split = harness.build_split(seed=42, canary_paths=harness.load_canary_paths())
    new_rows = {row["sha256"]: row for row in records if row["new_content"]}
    train_by_digest = {harness.content_sha256(path): (path, label) for path, label in active_split["train"]}
    for identity, row in new_rows.items():
        if identity not in train_by_digest or train_by_digest[identity][1] != 2:
            raise RuntimeError(f"New source is not a screen_photo training sample: {row['target']}")
    write_new(DATA / "split.json", split)
    write_new(
        DATA / "protocol.json",
        {
            "variants": VARIANTS,
            "seeds": SEEDS,
            "epochs_finetune": EPOCHS,
            "batch_size": 8,
            "cuda_memory_fraction": 0.70,
            "cache_images": 64,
            "unfreeze": 3,
            "canary_weight": 2.0,
            "new_content_weight": 8.0,
            "init_checkpoint_sha256": digest(ROOT / "trainer/checkpoints/three_class_best.pth"),
            "production_onnx_sha256": digest(ROOT / "inference/models/three_class.onnx"),
            "split": active_split["meta"],
            "thresholds": THRESHOLDS,
            "checkpoint_selection": "validation metric only, then Canary gate",
            "variant_selection": "mean actual Predictor validation metric across both seeds; both Canary gates must pass",
            "primary_seed": 42,
            "test_policy": "freeze selection before evaluating winner and same-data baseline (both seeds), plus production once",
            "promotion": "candidate only; independent challenge/OOD and collection-group evidence required",
        },
    )
    print(
        {
            "new": len(new_rows),
            "archives": len(archives),
            "split": {role: len(active_split[role]) for role in ("train", "val", "test")},
            "counts_by_day": dict(Counter(row["archive"] for row in records)),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
