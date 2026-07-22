"""Reproducible release training using the ablation-winning pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch

from experiment.cnn_fft_dwt_ablation.harness import (
    EXP_DIR,
    ExpConfig,
    append_leaderboard,
    build_split,
    load_focus_paths,
    precompute_cache,
    preload_ram,
    train_one,
)

from . import config


def _backup_existing_checkpoints(timestamp: str) -> Path | None:
    existing = [
        path
        for path in (
            config.CHECKPOINT_DIR / "three_class_best.pth",
            config.CHECKPOINT_DIR / "three_class_final.pth",
        )
        if path.exists()
    ]
    if not existing:
        return None

    backup_dir = config.CHECKPOINT_DIR / "backup" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", default="candidate_20260722_unf3_focus2_6x12")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focus-weight", type=float, default=2.0)
    parser.add_argument("--epochs-head", type=int, default=6)
    parser.add_argument("--epochs-finetune", type=int, default=12)
    parser.add_argument("--unfreeze", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    focus_paths = load_focus_paths()
    if args.focus_weight > 1.0 and not focus_paths:
        raise RuntimeError("Focus weighting was requested but trainer/hard_examples.txt has no available images")

    split = build_split(seed=args.seed, focus_paths=focus_paths)
    all_samples = [tuple(sample) for sample in split["train"] + split["val"] + split["test"]]
    print(
        f"Release split: train={len(split['train'])} val={len(split['val'])} "
        f"test={len(split['test'])} focus={len(focus_paths)}",
        flush=True,
    )
    precompute_cache(all_samples)
    preload_ram(all_samples)

    cfg = ExpConfig(
        id=args.id,
        desc="release: unfreeze3 focus2 6+12 on current split",
        backbone="efficientnet_b0",
        gamma=2.0,
        alpha=[1.0, 1.0, 1.5],
        smoothing=0.05,
        use_attention=False,
        use_arcface=False,
        ema=True,
        ema_decay=0.999,
        unfreeze=args.unfreeze,
        lr=1e-3,
        weight_decay=1e-4,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        sampler_beta=None,
        heavy_aug=False,
        use_dwt=True,
        seed=args.seed,
        focus_weight=args.focus_weight,
    )

    started_at = datetime.now().astimezone()
    started = time.monotonic()
    result = train_one(cfg, split, device)
    elapsed = time.monotonic() - started

    source_checkpoint = EXP_DIR / cfg.id / "best.pth"
    if not source_checkpoint.exists():
        raise RuntimeError(f"Release checkpoint was not created: {source_checkpoint}")

    timestamp = started_at.strftime("%Y%m%d-%H%M%S")
    backup_dir = _backup_existing_checkpoints(timestamp)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ("three_class_best.pth", "three_class_final.pth"):
        shutil.copy2(source_checkpoint, config.CHECKPOINT_DIR / filename)

    summary = {
        "id": cfg.id,
        "started_at": started_at.isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "device": device,
        "config": cfg.__dict__,
        "split": {
            **split.get("meta", {}),
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
        },
        "focus_examples": sorted(
            Path(path).resolve().relative_to(config.DATA_DIR.resolve()).as_posix() for path in focus_paths
        ),
        "selection": result["selection"],
        "validation": result["val"],
        "test": result["test"],
        "test_threshold": result["test_threshold"],
        "source_checkpoint": str(source_checkpoint),
        "canonical_checkpoint": str(config.CHECKPOINT_DIR / "three_class_best.pth"),
        "backup_dir": str(backup_dir) if backup_dir else None,
    }
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = config.LOG_DIR / "release_training_result.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    append_leaderboard(cfg, result, elapsed)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Release training summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
