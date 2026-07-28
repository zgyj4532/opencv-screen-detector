"""Evaluate the top 3 leaderboard checkpoints on 5 random 200-image samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiment.cnn_fft_dwt_ablation.harness import (
    ABLATION_DIR,
    EXP_DIR,
    LEADERBOARD,
    CachedDataset,
    collect_samples,
    eval_tf,
    precompute_cache,
    preload_ram,
)
from trainer.model import load_model

OUTPUT_PATH = ABLATION_DIR / "random200_top3_eval.json"
CLASS_NAMES = ["natural", "screenshot", "screen_photo"]


@torch.no_grad()
def evaluate_checkpoint(checkpoint: Path, samples: list[tuple[str, int]], device: str, batch_size: int) -> dict:
    model = load_model(str(checkpoint), device=device, use_dwt=None).to(device)
    model.eval()
    loader = DataLoader(CachedDataset(samples, eval_tf()), batch_size=batch_size, shuffle=False, num_workers=0)
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for rgb, fft, dwt, labels in loader:
        rgb, fft, dwt = rgb.to(device), fft.to(device), dwt.to(device)
        out = model(rgb, fft, dwt)
        if isinstance(out, tuple):
            out = out[0]
        probs_all.append(torch.softmax(out, 1).cpu().numpy())
        labels_all.append(labels.numpy())
    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    preds = probs.argmax(1)
    p, r, f1, support = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2], zero_division=0)
    acc = float((preds == labels).mean())
    macro_f1 = float(f1.mean())
    sp_f1 = float(f1[2])
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "sp_f1": sp_f1,
        "sp_precision": float(p[2]),
        "sp_recall": float(r[2]),
        "metric": float(0.5 * sp_f1 + 0.3 * acc + 0.2 * macro_f1),
        "support": {name: int(count) for name, count in zip(CLASS_NAMES, support, strict=True)},
    }


def load_top3(metric: str) -> list[dict]:
    rows = [json.loads(line) for line in LEADERBOARD.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if (EXP_DIR / row["id"] / "best.pth").exists()]
    rows.sort(key=lambda row: row.get(metric, 0), reverse=True)
    return rows[:3]


def summarize(values: list[float]) -> dict:
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="test_metric")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    main_samples, hard_samples = collect_samples()
    samples = [tuple(sample) for sample in main_samples + hard_samples]
    if len(samples) < args.sample_size:
        raise RuntimeError(f"Need at least {args.sample_size} samples, got {len(samples)}")
    precompute_cache(samples)
    preload_ram(samples)

    top3 = load_top3(args.metric)
    rng = np.random.default_rng(args.seed)
    sample_indices = [
        rng.choice(len(samples), size=args.sample_size, replace=False).tolist() for _ in range(args.repeats)
    ]
    repeats: list[dict] = []
    per_candidate: dict[str, list[dict]] = {row["id"]: [] for row in top3}
    for repeat_index, indices in enumerate(sample_indices, start=1):
        subset = [samples[index] for index in indices]
        row_out = {"repeat": repeat_index, "sample_size": len(subset), "candidates": []}
        for row in top3:
            checkpoint = EXP_DIR / row["id"] / "best.pth"
            metrics = evaluate_checkpoint(checkpoint, subset, device, args.batch_size)
            candidate = {
                "id": row["id"],
                "checkpoint": str(checkpoint),
                **metrics,
            }
            per_candidate[row["id"]].append(metrics)
            row_out["candidates"].append(candidate)
            print(
                f"repeat={repeat_index} id={row['id']} acc={metrics['accuracy']:.4f} "
                f"spF1={metrics['sp_f1']:.4f} macroF1={metrics['macro_f1']:.4f} "
                f"metric={metrics['metric']:.4f}",
                flush=True,
            )
            if device == "cuda":
                torch.cuda.empty_cache()
        repeats.append(row_out)

    summary = []
    for row in top3:
        metrics = per_candidate[row["id"]]
        summary.append(
            {
                "id": row["id"],
                "checkpoint": str(EXP_DIR / row["id"] / "best.pth"),
                "leaderboard_metric": row.get(args.metric),
                "accuracy": summarize([m["accuracy"] for m in metrics]),
                "sp_f1": summarize([m["sp_f1"] for m in metrics]),
                "macro_f1": summarize([m["macro_f1"] for m in metrics]),
                "metric": summarize([m["metric"] for m in metrics]),
            }
        )
    summary.sort(key=lambda row: row["metric"]["mean"], reverse=True)

    output = {
        "metric_used_for_top3": args.metric,
        "sample_source": "data/input main classes + hard_negative",
        "sample_size": args.sample_size,
        "repeats": args.repeats,
        "seed": args.seed,
        "device": device,
        "batch_size": args.batch_size,
        "top3": [row["id"] for row in top3],
        "summary": summary,
        "repeat_results": repeats,
        "best": summary[0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["best"], ensure_ascii=False, indent=2), flush=True)
    print(f"random200 result saved -> {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
