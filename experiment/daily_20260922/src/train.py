"""Serial, bounded-memory warm starts using the shared release training harness."""

import argparse
import time
from dataclasses import asdict
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiment.cnn_fft_dwt_ablation import harness
from experiment.daily_20260922.src.protocol import (
    DATA,
    EPOCHS,
    REPORTS,
    ROOT,
    SEEDS,
    THRESHOLDS,
    VARIANTS,
    candidate_name,
    digest,
    prepare_directories,
    read_json,
    write_new,
)


def bounded_loader(dataset, **kwargs: Any):
    kwargs["batch_size"] = min(kwargs.get("batch_size", 8), 8)
    return DataLoader(dataset, **kwargs)


@lru_cache(maxsize=64)
def read_entry(path):
    cache_path = harness._cache_key(path)  # noqa: SLF001 - preserve the shared cache identity
    rgb_path = cache_path.with_suffix(".rgb.npy")
    if rgb_path.exists():
        rgb = np.load(rgb_path, allow_pickle=False)
    else:
        rgb = cv2.resize(harness._load_rgb(path), (256, 256))  # noqa: SLF001 - exact harness decoding
        np.save(rgb_path, rgb, allow_pickle=False)
    with np.load(cache_path, allow_pickle=False) as cache:
        return {"rgb": rgb, "fft": cache["fft"], "dwt": cache["dwt"]}


class BoundedCache:
    def __getitem__(self, path):
        return read_entry(path)


def checked_split():
    protocol = read_json(DATA / "protocol.json")
    if (
        protocol["variants"] != VARIANTS
        or tuple(protocol["seeds"]) != SEEDS
        or protocol["epochs_finetune"] != EPOCHS
        or protocol["thresholds"] != THRESHOLDS
    ):
        raise RuntimeError("Code settings changed from the frozen experiment protocol")
    split = harness.build_split(seed=42, canary_paths=harness.load_canary_paths())
    if split["meta"] != protocol["split"]:
        raise RuntimeError("Dataset, Canary, or evaluation identity changed since protocol freeze")
    return split


def main():
    prepare_directories()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.70)
    name = candidate_name(args.variant, args.seed)
    summary_path = REPORTS / f"{name}.train.json"
    if summary_path.exists():
        raise RuntimeError("Completed run already exists")
    run_directory = harness.EXP_DIR / name
    if (run_directory / "last.pth").exists() and not args.resume:
        raise RuntimeError("Candidate exists; use --resume with unchanged inputs")
    protocol = read_json(DATA / "protocol.json")
    checkpoint = ROOT / "trainer/checkpoints/three_class_best.pth"
    if digest(checkpoint) != protocol["init_checkpoint_sha256"]:
        raise RuntimeError("Initialization checkpoint changed")
    manifest = read_json(DATA / "ingest.json")
    rows = {row["sha256"]: row for row in manifest["records"] if row["new_content"]}
    for identity, row in rows.items():
        if digest(ROOT / row["target"]) != identity:
            raise RuntimeError(f"Input changed: {row['target']}")
    split = checked_split()
    boost_paths = sorted({str((ROOT / row["target"]).resolve()) for row in rows.values()})
    if not set(boost_paths).issubset({path for path, _ in split["train"]}):
        raise RuntimeError("New images must belong only to train")
    cfg = harness.ExpConfig(
        id=name,
        desc=f"Sept 15–21 +{len(rows)} photos; {args.variant}; validation only",
        seed=args.seed,
        unfreeze=3,
        focus_weight=2.0,
        epochs_head=0,
        epochs_finetune=EPOCHS,
        batch_size=8,
        init_checkpoint=str(checkpoint),
        boost_paths=boost_paths,
        boost_weight=8.0,
        **VARIANTS[args.variant],
    )
    context = {
        "config": asdict(cfg),
        "protocol_sha256": digest(DATA / "protocol.json"),
        "ingest_sha256": digest(DATA / "ingest.json"),
    }
    context_path = run_directory / "run_context.json"
    if args.resume:
        if read_json(context_path) != context:
            raise RuntimeError("Resume configuration or evidence changed")
    else:
        write_new(context_path, context)
    harness.DataLoader = bounded_loader
    harness.precompute_cache([tuple(row) for row in split["train"] + split["val"]])
    harness.RAM = BoundedCache()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    result = harness.train_one(cfg, split, "cuda", evaluate_test=False, resume=args.resume)
    write_new(
        summary_path,
        {
            "id": name,
            "variant": args.variant,
            "seed": args.seed,
            "config": asdict(cfg),
            "split": split["meta"],
            "protocol_sha256": context["protocol_sha256"],
            "best_checkpoint_sha256": digest(run_directory / "best.pth"),
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "test_status": "sealed_validation_only",
            "result": result,
        },
    )
    print({"id": name, "selection": result["selection"]}, flush=True)


if __name__ == "__main__":
    main()
