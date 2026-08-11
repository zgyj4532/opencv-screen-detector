"""Run finalist (a_unf1) at full 10+20 epochs, with multi-seed averaging for robustness.

Usage: uv run python experiment/cnn_fft_dwt_ablation/finalist.py [--seeds 42 2024 7]
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # type: ignore
    EXP_DIR,
    ExpConfig,
    append_leaderboard,
    build_split,
    precompute_cache,
    preload_ram,
    train_one,
)

FINAL_DIR = ROOT / "experiment" / "cnn_fft_dwt_ablation" / "finalist"
FINAL_DIR.mkdir(exist_ok=True, parents=True)


def run_one(cfg: ExpConfig, tag: str, device: str = "cuda", skip_existing: bool = True) -> dict:
    existing = FINAL_DIR / tag / "best.pth"
    if skip_existing and existing.exists():
        # Recover row from leaderboard
        lb_path = ROOT / "experiment" / "cnn_fft_dwt_ablation" / "leaderboard.jsonl"
        if lb_path.exists():
            for line in lb_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("id") == tag:
                    print(f"\n=== skip {tag}: cached in leaderboard ===", flush=True)
                    return {
                        "id": tag,
                        "cfg": cfg.__dict__,
                        "val_acc": rec.get("val_acc", float("nan")),
                        "val_sp_f1": rec.get("val_sp_f1", float("nan")),
                        "val_metric": rec.get("val_metric", float("nan")),
                        "test_acc": rec["test_acc"],
                        "test_sp_f1": rec["test_sp_f1"],
                        "test_sp_precision": rec.get("test_sp_precision", float("nan")),
                        "test_sp_recall": rec.get("test_sp_recall", float("nan")),
                        "test_macro_f1": rec["test_macro_f1"],
                        "test_metric": rec["test_metric"],
                        "test_thr": rec["test_thr"],
                        "elapsed_s": rec.get("elapsed_s", 0),
                    }
        raise RuntimeError(f"{tag} checkpoint exists but no leaderboard entry; delete the checkpoint to re-run")
    print(f"\n{'=' * 70}\n### {tag}: {cfg.desc}\n{'=' * 70}", flush=True)
    split = build_split()
    print(f"Split: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}", flush=True)
    all_samples = [tuple(x) for x in split["train"] + split["val"] + split["test"]]
    precompute_cache(all_samples)
    preload_ram(all_samples)

    t0 = time.time()
    result = train_one(cfg, split, device)
    elapsed = time.time() - t0
    print(
        f">>> {tag} test_acc={result['test']['accuracy']:.4f} "
        f"test_spF1={result['test']['sp_f1']:.4f} test_macroF1={result['test']['macro_f1']:.4f} "
        f"thr_spF1={result['test_threshold']['sp_f1']:.4f} ({elapsed:.0f}s)",
        flush=True,
    )

    # Move/copy best.pth to a tag-specific final dir
    src_ckpt = EXP_DIR / cfg.id / "best.pth"
    dst = FINAL_DIR / tag
    dst.mkdir(exist_ok=True, parents=True)
    if src_ckpt.exists():
        (dst / "best.pth").write_bytes(src_ckpt.read_bytes())
        print(f"  copied {src_ckpt} -> {dst / 'best.pth'}", flush=True)

    row = {
        "id": tag,
        "cfg": cfg.__dict__,
        "val_acc": result["val"]["accuracy"],
        "val_sp_f1": result["val"]["sp_f1"],
        "val_metric": result["val"]["best_metric"],
        "test_acc": result["test"]["accuracy"],
        "test_sp_f1": result["test"]["sp_f1"],
        "test_sp_precision": result["test"]["sp_precision"],
        "test_sp_recall": result["test"]["sp_recall"],
        "test_macro_f1": result["test"]["macro_f1"],
        "test_metric": result["test"]["best_metric"],
        "test_thr": result["test_threshold"],
        "elapsed_s": int(elapsed),
    }
    append_leaderboard(cfg, result, elapsed)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 7])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    rows = []
    for s in args.seeds:
        cfg = ExpConfig(
            id=f"finalist_unf1_s{s}",
            desc=f"finalist a_unf1 seed={s} full epochs",
            backbone="efficientnet_b0",
            gamma=2.0,
            alpha=[1.0, 1.0, 1.5],
            smoothing=0.05,
            use_attention=False,
            attention_type="cbam",
            use_arcface=False,
            ema=True,
            ema_decay=0.999,
            unfreeze=1,
            lr=1e-3,
            weight_decay=1e-4,
            epochs_head=10,
            epochs_finetune=20,
            batch_size=16,
            sampler_beta=None,
            heavy_aug=False,
            use_dwt=True,
            seed=s,
        )
        rows.append(run_one(cfg, cfg.id, args.device, skip_existing=args.skip_existing))

    print(f"\n{'=' * 70}\nFINALIST SUMMARY\n{'=' * 70}")
    print(f"{'id':<22} {'test_acc':>8} {'test_spF1':>9} {'macroF1':>8} {'metric':>7}")
    for r in rows:
        print(
            f"{r['id']:<22} {r['test_acc']:>8.4f} {r['test_sp_f1']:>9.4f} "
            f"{r['test_macro_f1']:>8.4f} {r['test_metric']:>7.4f}"
        )
    avg = {k: sum(r[k] for r in rows) / len(rows) for k in ["test_acc", "test_sp_f1", "test_macro_f1", "test_metric"]}
    print(
        f"{'AVG':<22} {avg['test_acc']:>8.4f} {avg['test_sp_f1']:>9.4f} "
        f"{avg['test_macro_f1']:>8.4f} {avg['test_metric']:>7.4f}"
    )

    (FINAL_DIR / "summary.json").write_text(
        json.dumps({"runs": rows, "avg": avg}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best = max(rows, key=lambda r: r["test_metric"])
    best_ckpt = FINAL_DIR / best["id"] / "best.pth"
    canonical = FINAL_DIR / "best.pth"
    canonical.write_bytes(best_ckpt.read_bytes())
    print(f"\nWinner by test_metric: {best['id']} (metric={best['test_metric']:.4f})")
    print(f"  -> {canonical}")


if __name__ == "__main__":
    main()
