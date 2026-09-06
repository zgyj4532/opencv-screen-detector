"""Reproducible release training using the ablation-winning pipeline."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from experiment.cnn_fft_dwt_ablation.harness import (
    EXP_DIR,
    ExpConfig,
    append_leaderboard,
    build_split,
    load_canary_paths,
    precompute_cache,
    preload_ram,
    train_one,
)
from trainer.evaluation_sets import evaluation_set_readiness

from . import config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", default="candidate_20260722_unf3_focus2_6x12")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--canary-weight",
        "--focus-weight",
        dest="canary_weight",
        type=float,
        default=2.0,
        help="Canary sampler weight; --focus-weight is a deprecated alias",
    )
    parser.add_argument("--epochs-head", type=int, default=6)
    parser.add_argument("--epochs-finetune", type=int, default=12)
    parser.add_argument("--unfreeze", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--remix-alpha", type=float, default=0.2)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--boost-path", action="append", default=None)
    parser.add_argument("--boost-weight", type=float, default=1.0)
    parser.add_argument("--distill-alpha", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    canary_paths = load_canary_paths()
    if args.canary_weight > 1.0 and not canary_paths:
        raise RuntimeError("Canary weighting was requested but trainer/evaluation_sets/canary.json has no images")

    split = build_split(seed=args.split_seed, canary_paths=canary_paths)
    all_samples = [tuple(sample) for sample in split["train"] + split["val"] + split["test"]]
    print(
        f"Release split: train={len(split['train'])} val={len(split['val'])} "
        f"test={len(split['test'])} canary={len(canary_paths)}",
        flush=True,
    )
    precompute_cache(all_samples)
    preload_ram(all_samples)

    cfg = ExpConfig(
        id=args.id,
        desc="release: unfreeze3 canary2 remix0.2 6+12 on current split",
        backbone="efficientnet_b0",
        gamma=2.0,
        alpha=[1.0, 1.0, 1.5],
        smoothing=0.05,
        use_attention=False,
        use_arcface=False,
        ema=True,
        ema_decay=0.999,
        unfreeze=args.unfreeze,
        lr=args.lr,
        weight_decay=1e-4,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        sampler_beta=None,
        heavy_aug=False,
        use_dwt=True,
        seed=args.seed,
        # Historical checkpoint schema stores this under focus_weight.
        focus_weight=args.canary_weight,
        # Remix on Mixup λ=Beta(0.2,0.2): ImageNet Mixup default (Zhang et al. ICLR 2018)
        # plus minority-label bias (Chou et al. ECCV 2020 W) for the small extra-data case.
        remix_alpha=args.remix_alpha,
        remix_kappa=3.0,
        remix_tau=0.5,
        init_checkpoint=str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        boost_paths=[str(Path(path).resolve()) for path in (args.boost_path or [])],
        boost_weight=args.boost_weight,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
    )

    started_at = datetime.now().astimezone()
    started = time.monotonic()
    result = train_one(cfg, split, device)
    elapsed = time.monotonic() - started

    source_checkpoint = EXP_DIR / cfg.id / "best.pth"
    if not source_checkpoint.exists():
        raise RuntimeError(f"Release checkpoint was not created: {source_checkpoint}")

    canary_pass = bool(result["selection"].get("canary_pass"))
    set_readiness = evaluation_set_readiness()

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
        "canary_examples": sorted(
            Path(path).resolve().relative_to(config.DATA_DIR.resolve()).as_posix() for path in canary_paths
        ),
        "canary_gate": {
            "status": "PASS" if canary_pass else "FAIL",
            "correct": result["selection"].get("canary_correct", 0),
            "total": result["selection"].get("canary_total", len(canary_paths)),
            "role": "known-regression blocker only; not promotion statistics",
        },
        "evaluation_set_readiness": set_readiness,
        "promotion_status": "candidate_only_pending Predictor, Frozen challenge, true-OOD, and isolation gates",
        "selection": result["selection"],
        "validation": result["val"],
        "test": result["test"],
        "test_threshold": result["test_threshold"],
        "source_checkpoint": str(source_checkpoint),
        "canonical_checkpoint": None,
        "canonical_checkpoint_unchanged": str(config.CHECKPOINT_DIR / "three_class_best.pth"),
        "next_step": "Export the candidate from source_checkpoint, run deploy_eval, then promote only after every gate passes",
    }
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = config.LOG_DIR / "release_training_result.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    append_leaderboard(cfg, result, elapsed)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Release training summary: {summary_path}", flush=True)
    if not canary_pass:
        raise RuntimeError(
            "Selected validation checkpoint failed the Canary regression gate; canonical files unchanged"
        )


if __name__ == "__main__":
    main()
