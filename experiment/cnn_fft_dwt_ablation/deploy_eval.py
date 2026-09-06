"""End-to-end inference evaluation using the deployed ONNX model.

Runs the production ONNX through the inference.predictor pipeline against the
held-out test split, applying the inference runtime post-processing
(OOD threshold + screen_photo threshold + TTA).

Usage:
    uv run python experiment/cnn_fft_dwt_ablation/deploy_eval.py
    uv run python experiment/cnn_fft_dwt_ablation/deploy_eval.py \
        --model path/to/model.onnx --output path/to/result.json
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from inference.config import configure
from inference.predictor import ScreenDetectorPredictor
from trainer.evaluation_sets import (
    NOT_READY,
    evaluate_known_and_ood,
    evaluation_set_readiness,
    promotion_readiness,
    resolve_manifest_samples,
)

ONNX_PATH = ROOT / "inference" / "models" / "three_class.onnx"
DEFAULT_OUTPUT_PATH = ROOT / "experiment" / "cnn_fft_dwt_ablation" / "deploy_eval.json"
CLASSES = ["natural", "screenshot", "screen_photo"]
C2I = {"natural": 0, "screenshot": 1, "screen_photo": 2, "unknown": -1}


def portable_path(path: Path) -> str:
    """Keep repository-owned result paths portable across workspaces."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def predict_manifest(pred: ScreenDetectorPredictor, set_name: str) -> dict:
    """Run one governed evaluation set through the production Predictor."""
    samples = resolve_manifest_samples(set_name)
    probabilities = []
    final_labels = []
    expected_labels = []
    records = []
    for sample in samples:
        result = pred.predict(sample.path)
        probs = [result["probabilities"][name] for name in CLASSES]
        probabilities.append(probs)
        final_labels.append(result["class"])
        expected_labels.append(sample.expected_label)
        records.append(
            {
                "id": sample.sample_id,
                "path": portable_path(sample.path),
                "expected_label": sample.expected_label,
                "category": sample.category,
                "probabilities": result["probabilities"],
                "final_label": result["class"],
                "confidence_tier": result.get("confidence_tier"),
                "action": result.get("action"),
            }
        )
    return {
        "probabilities": np.asarray(probabilities, dtype=np.float64),
        "final_labels": final_labels,
        "expected_labels": expected_labels,
        "records": records,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ONNX_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--label", default="production")
    args = parser.parse_args(argv)
    model_path = args.model.resolve()

    print("=== ONNX deploy eval ===")
    print(f"model: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX missing at {model_path}")
    print(f"size: {model_path.stat().st_size / 1e6:.1f} MB")

    # Apply inference runtime config (production thresholds)
    configure(
        ood_threshold=0.45,
        confidence_high=0.92,
        confidence_medium=0.75,
        screen_photo_threshold=0.60,
    )

    # Load the deployment ONNX via the real production predictor
    pred = ScreenDetectorPredictor(model_path=model_path)
    print("predictor loaded ok\n")

    # Load the test split from the harness (same dir as this file)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import harness as h

    split = h.build_split()
    test = [tuple(x) for x in split["test"]]
    print(f"test split: {len(test)} samples\n")

    t0 = time.time()
    labels: list[int] = []
    preds: list[int] = []
    tiers: list[str] = []
    actions: list[str] = []
    known_probs: list[list[float]] = []
    elapsed_per = []

    for path, true_label in test:
        ts = time.time()
        r = pred.predict(Path(path))
        elapsed_per.append(time.time() - ts)
        pl = C2I[r["class"]]
        tl = int(true_label)
        labels.append(tl)
        preds.append(pl)
        tiers.append(r.get("confidence_tier", "?"))
        actions.append(r.get("action", "?"))
        known_probs.append([r["probabilities"][name] for name in CLASSES])

    elapsed = time.time() - t0
    labels_a = np.array(labels)
    preds_a = np.array(preds)

    # Production returns unknown directly, so the raw view is the deployment metric.
    raw_macro = f1_score(labels_a, preds_a, average="macro", labels=[0, 1, 2], zero_division=0)
    raw_acc = float((preds_a == labels_a).mean())
    sp_tp = int(((labels_a == 2) & (preds_a == 2)).sum())
    sp_fp = int(((labels_a != 2) & (preds_a == 2)).sum())
    sp_fn = int(((labels_a == 2) & (preds_a != 2)).sum())
    sp_p = sp_tp / max(1, sp_tp + sp_fp)
    sp_r = sp_tp / max(1, sp_tp + sp_fn)
    sp_f1 = 2 * sp_p * sp_r / max(1e-9, sp_p + sp_r)
    raw_metric = 0.5 * sp_f1 + 0.3 * raw_acc + 0.2 * raw_macro

    # Supplemental sensitivity view only: the production API does not apply this fallback.
    pred_corr = np.where(preds_a == -1, 2, preds_a)
    cor_macro = f1_score(labels_a, pred_corr, average="macro", labels=[0, 1, 2], zero_division=0)
    cor_acc = float((pred_corr == labels_a).mean())
    sp_tp_c = int(((labels_a == 2) & (pred_corr == 2)).sum())
    sp_fp_c = int(((labels_a != 2) & (pred_corr == 2)).sum())
    sp_fn_c = int(((labels_a == 2) & (pred_corr != 2)).sum())
    sp_p_c = sp_tp_c / max(1, sp_tp_c + sp_fp_c)
    sp_r_c = sp_tp_c / max(1, sp_tp_c + sp_fn_c)
    sp_f1_c = 2 * sp_p_c * sp_r_c / max(1e-9, sp_p_c + sp_r_c)
    cor_metric = 0.5 * sp_f1_c + 0.3 * cor_acc + 0.2 * cor_macro

    # Confusion
    print("Confusion (rows=true, cols=pred, [-1]=OOD):")
    print(f"  {'':12} {'natural':>8} {'screenshot':>10} {'screen_photo':>12} {'unknown':>8}")
    for t in [0, 1, 2]:
        row = [int(((labels_a == t) & (preds_a == p)).sum()) for p in [0, 1, 2, -1]]
        print(f"  {CLASSES[t]:12} {row[0]:>8} {row[1]:>10} {row[2]:>12} {row[3]:>8}")

    # Tier distribution
    from collections import Counter

    print("\nConfidence tier distribution:")
    for t, c in Counter(tiers).most_common():
        print(f"  {t}: {c}")
    print("\nAction distribution:")
    for a, c in Counter(actions).most_common():
        print(f"  {a}: {c}")

    print()
    print("=" * 70)
    print(f"{'view':<22} {'acc':>7} {'sp_f1':>7} {'sp_P':>6} {'sp_R':>6} {'macro_F1':>9} {'metric':>7}")
    print("-" * 70)
    print(
        f"{'raw (unknown counted)':<22} {raw_acc:>7.4f} {sp_f1:>7.4f} {sp_p:>6.3f} {sp_r:>6.3f} {raw_macro:>9.4f} {raw_metric:>7.4f}"
    )
    print(
        f"{'hypothetical OOD->SP':<22} {cor_acc:>7.4f} {sp_f1_c:>7.4f} {sp_p_c:>6.3f} {sp_r_c:>6.3f} {cor_macro:>9.4f} {cor_metric:>7.4f}"
    )
    print("=" * 70)

    # Latency
    elapsed_per = np.array(elapsed_per)
    print(f"\nLatency per image (n={len(elapsed_per)}):")
    print(f"  mean: {elapsed_per.mean() * 1000:.1f} ms")
    print(f"  p50:  {np.percentile(elapsed_per, 50) * 1000:.1f} ms")
    print(f"  p95:  {np.percentile(elapsed_per, 95) * 1000:.1f} ms")
    print(f"  total: {elapsed:.1f}s")

    # Evaluate governed sets separately. Empty/unreviewed manifests stay NOT_READY.
    set_readiness = evaluation_set_readiness()
    canary_run = predict_manifest(pred, "canary") if set_readiness["canary"]["status"] == "READY" else None
    canary_pass = bool(
        canary_run
        and all(
            actual == expected
            for actual, expected in zip(canary_run["final_labels"], canary_run["expected_labels"], strict=True)
        )
    )

    challenge_report: dict = {
        "status": NOT_READY,
        "reasons": set_readiness["frozen_challenge"]["reasons"],
    }
    if set_readiness["frozen_challenge"]["status"] == "READY":
        challenge_run = predict_manifest(pred, "frozen_challenge")
        challenge_labels = np.asarray([C2I[label] for label in challenge_run["expected_labels"]], dtype=np.int64)
        challenge_report = {
            "status": "READY",
            **evaluate_known_and_ood(challenge_run["probabilities"], challenge_labels),
            "records": challenge_run["records"],
        }

    ood_probs = None
    ood_records = []
    if set_readiness["ood"]["status"] == "READY":
        ood_run = predict_manifest(pred, "ood")
        ood_probs = ood_run["probabilities"]
        ood_records = ood_run["records"]
    separated_metrics = evaluate_known_and_ood(
        np.asarray(known_probs, dtype=np.float64),
        labels_a,
        ood_probs,
        ood_threshold=0.45,
    )

    audit_path = ROOT / "trainer" / "data_audit.json"
    isolation_status = NOT_READY
    if audit_path.exists():
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_payload.get("split", {}).get("dataset_fingerprint") == split.get("meta", {}).get(
            "dataset_fingerprint"
        ):
            isolation_status = audit_payload.get("split_isolation", {}).get("status", NOT_READY)
    promotion = promotion_readiness(set_readiness, isolation_status, canary_pass=canary_pass)

    # Save result. Existing keys remain for historical consumers.
    out = {
        "schema_version": 2,
        "label": args.label,
        "onnx_path": portable_path(model_path),
        "onnx_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "size_mb": round(model_path.stat().st_size / 1e6, 2),
        "n_test": len(test),
        "raw": {
            "accuracy": raw_acc,
            "sp_f1": sp_f1,
            "sp_precision": sp_p,
            "sp_recall": sp_r,
            "macro_f1": raw_macro,
            "metric": raw_metric,
        },
        "hypothetical_ood_as_screen_photo": {
            "accuracy": cor_acc,
            "sp_f1": sp_f1_c,
            "sp_precision": sp_p_c,
            "sp_recall": sp_r_c,
            "macro_f1": cor_macro,
            "metric": cor_metric,
        },
        "latency_ms": {
            "mean": float(elapsed_per.mean() * 1000),
            "p50": float(np.percentile(elapsed_per, 50) * 1000),
            "p95": float(np.percentile(elapsed_per, 95) * 1000),
        },
        "tiers": dict(Counter(tiers)),
        "actions": dict(Counter(actions)),
        "evaluation_system": {
            "legacy_closed_test": {
                "status": "historical_benchmark_only",
                "samples": len(test),
                "warning": "This set is not a true-OOD set or the independent Frozen challenge set",
            },
            "set_readiness": set_readiness,
            "canary": {
                "status": "PASS" if canary_pass else "FAIL",
                "correct": (
                    sum(
                        actual == expected
                        for actual, expected in zip(
                            canary_run["final_labels"], canary_run["expected_labels"], strict=True
                        )
                    )
                    if canary_run
                    else 0
                ),
                "total": len(canary_run["records"]) if canary_run else 0,
                "role": "known-regression blocker only; not generalization evidence",
                "records": canary_run["records"] if canary_run else [],
            },
            **separated_metrics,
            "frozen_challenge": challenge_report,
            "ood_records": ood_records,
            "split_isolation_status": isolation_status,
            "promotion_readiness": promotion,
        },
        "baseline_to_beat": {"acc": 0.8946, "macro_f1": 0.8668, "sp_f1": 0.7630},
    }
    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresult saved -> {out_path}")


if __name__ == "__main__":
    main()
