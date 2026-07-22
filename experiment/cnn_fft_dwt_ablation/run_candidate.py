"""Run one release-training candidate without promoting it to production paths.

This is intentionally narrower than ``trainer.release_train``: it uses the same
split/cache/train_one harness, appends the leaderboard, and writes a JSON
summary, but it does not copy the result to ``trainer/checkpoints``.
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
    load_focus_paths,
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
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--unfreeze", type=int, default=1)
    parser.add_argument("--focus-weight", type=float, default=4.0)
    parser.add_argument("--epochs-head", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--smoothing", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, nargs=3, default=[1.0, 1.0, 1.5])
    parser.add_argument("--sampler-beta", type=float, default=None)
    parser.add_argument("--heavy-aug", action="store_true")
    parser.add_argument("--no-dwt", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

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
        f"Candidate split: train={len(split['train'])} val={len(split['val'])} "
        f"test={len(split['test'])} focus={len(focus_paths)}",
        flush=True,
    )
    precompute_cache(all_samples)
    preload_ram(all_samples)

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
        focus_weight=args.focus_weight,
    )

    started_at = datetime.now().astimezone()
    started = time.monotonic()
    result = train_one(cfg, split, device)
    elapsed = time.monotonic() - started
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
        "focus_examples": sorted(
            Path(path).resolve().relative_to((ROOT / "data" / "input").resolve()).as_posix()
            for path in focus_paths
        ),
        "selection": result["selection"],
        "validation": result["val"],
        "test": result["test"],
        "test_threshold": result["test_threshold"],
        "source_checkpoint": str(EXP_DIR / cfg.id / "best.pth"),
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LOG_DIR / f"{cfg.id}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Candidate summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
