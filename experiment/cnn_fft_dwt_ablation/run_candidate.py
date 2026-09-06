"""Run one release-training candidate without promoting it to production paths.

This is intentionally narrower than ``trainer.release_train``: it uses the same
split/cache/train_one harness and writes a JSON summary, but it does not copy
the result to ``trainer/checkpoints``. Validation-only runs keep test sealed
and do not append the historical test leaderboard.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiment.cnn_fft_dwt_ablation.harness import (
    EXP_DIR,
    ExpConfig,
    append_leaderboard,
    build_split,
    evaluate_checkpoint_on_test,
    load_canary_paths,
    precompute_cache,
    preload_ram,
    train_one,
)

LOG_DIR = ROOT / "trainer" / "logs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--desc", default="release candidate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--unfreeze", type=int, default=3)
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--smoothing", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, nargs=3, default=[1.0, 1.0, 1.5])
    parser.add_argument("--sampler-beta", type=float, default=None)
    parser.add_argument("--heavy-aug", action="store_true")
    parser.add_argument("--no-dwt", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true", help="continue exactly from this candidate's last epoch")
    parser.add_argument("--max-total-epochs", type=int, default=None, help="pause safely after this many total epochs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validation-only", action="store_true", help="keep the frozen test set sealed")
    mode.add_argument("--evaluate-only", action="store_true", help="evaluate this existing candidate on frozen test")
    args = parser.parse_args()
    if args.evaluate_only and (args.resume or args.max_total_epochs is not None):
        parser.error("--evaluate-only cannot be combined with --resume or --max-total-epochs")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    canary_paths = load_canary_paths()
    if args.canary_weight > 1.0 and not canary_paths:
        raise RuntimeError("Canary weighting was requested but trainer/evaluation_sets/canary.json has no images")

    split = build_split(seed=args.split_seed, canary_paths=canary_paths)
    summary_path = LOG_DIR / f"{args.id}.summary.json"
    evaluation_summary = None
    checkpoint_path = EXP_DIR / args.id / "best.pth"
    if args.evaluate_only:
        if not checkpoint_path.exists():
            raise RuntimeError(f"Candidate checkpoint does not exist: {checkpoint_path}")
        if not summary_path.exists():
            raise RuntimeError(f"Candidate summary does not exist: {summary_path}")
        evaluation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if evaluation_summary.get("training_status") != "complete":
            raise RuntimeError("Candidate training is incomplete; refusing to open the frozen test set")
        if evaluation_summary.get("test_status") == "evaluated_once_after_validation_selection":
            raise RuntimeError("Candidate was already evaluated on the frozen test set")

    if args.evaluate_only:
        active_samples = [tuple(sample) for sample in split["test"]]
    elif args.validation_only:
        active_samples = [tuple(sample) for sample in split["train"] + split["val"]]
    else:
        active_samples = [tuple(sample) for sample in split["train"] + split["val"] + split["test"]]
    print(
        f"Candidate split: train={len(split['train'])} val={len(split['val'])} "
        f"test={len(split['test'])} canary={len(canary_paths)}",
        flush=True,
    )
    precompute_cache(active_samples)
    preload_ram(active_samples)

    if args.evaluate_only:
        summary = evaluation_summary
        test_result = evaluate_checkpoint_on_test(checkpoint_path, split, device)
        summary["test"] = test_result["metrics"]
        summary["test_threshold_diagnostic"] = test_result["threshold_diagnostic"]
        summary["test_status"] = "evaluated_once_after_validation_selection"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"Candidate test summary: {summary_path}", flush=True)
        return

    cfg = ExpConfig(
        id=args.id,
        desc=args.desc,
        backbone=args.backbone,
        gamma=args.gamma,
        alpha=list(args.alpha),
        smoothing=args.smoothing,
        ema=not args.no_ema,
        unfreeze=args.unfreeze,
        epochs_head=args.epochs_head,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        sampler_beta=args.sampler_beta,
        heavy_aug=args.heavy_aug,
        use_dwt=not args.no_dwt,
        seed=args.seed,
        # Historical checkpoint schema stores this under focus_weight.
        focus_weight=args.canary_weight,
    )

    started_at = datetime.now().astimezone()
    started = time.monotonic()
    result = train_one(
        cfg,
        split,
        device,
        evaluate_test=not args.validation_only,
        resume=args.resume,
        max_total_epochs=args.max_total_epochs,
    )
    elapsed = time.monotonic() - started
    if result["status"] == "complete" and not args.validation_only:
        append_leaderboard(cfg, result, elapsed)

    summary = {
        "id": cfg.id,
        "started_at": started_at.isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "device": device,
        "config": asdict(cfg),
        "split": {
            **split.get("meta", {}),
            "train": len(split["train"]),
            "val": len(split["val"]),
            "test": len(split["test"]),
        },
        "canary_examples": sorted(
            Path(path).resolve().relative_to((ROOT / "data" / "input").resolve()).as_posix() for path in canary_paths
        ),
        "canary_gate": {
            "status": "PASS" if result["selection"].get("canary_pass") else "FAIL",
            "correct": result["selection"].get("canary_correct", 0),
            "total": result["selection"].get("canary_total", len(canary_paths)),
            "role": "known-regression blocker only; not promotion statistics",
        },
        "selection": result["selection"],
        "training_status": result["status"],
        "progress": result.get("progress"),
        "validation": result["val"],
        "test": result["test"],
        "test_threshold_diagnostic": result["test_threshold"],
        "test_status": (
            "sealed_paused"
            if result["status"] == "paused"
            else "sealed_validation_only"
            if args.validation_only
            else "evaluated_during_training_run"
        ),
        "source_checkpoint": str(EXP_DIR / cfg.id / "best.pth"),
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Candidate summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
