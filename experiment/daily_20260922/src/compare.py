"""Report paired uncertainty from recorded predictions without reopening models/test."""

import numpy as np

from experiment.daily_20260922.src.evaluate import CLASSES
from experiment.daily_20260922.src.protocol import (
    DATA,
    REPORTS,
    ROOT,
    SEEDS,
    candidate_name,
    digest,
    read_json,
    write_new,
)


def metric_samples(labels, predictions):
    accuracy = np.mean(labels == predictions, axis=1)
    f1 = []
    for label in range(3):
        true_positive = np.sum((labels == label) & (predictions == label), axis=1)
        denominator = np.sum(labels == label, axis=1) + np.sum(predictions == label, axis=1)
        f1.append(
            np.divide(
                2 * true_positive, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0
            )
        )
    return 0.5 * f1[2] + 0.3 * accuracy + 0.2 * np.mean(f1, axis=0)


def paired_comparison(candidate, baseline):
    candidate_rows = sorted(candidate["records"], key=lambda row: row["path"])
    baseline_rows = sorted(baseline["records"], key=lambda row: row["path"])
    if [(row["path"], row["label"]) for row in candidate_rows] != [
        (row["path"], row["label"]) for row in baseline_rows
    ]:
        raise ValueError("Paired reports must contain identical labeled images")
    labels = np.asarray([row["label"] for row in candidate_rows])

    def predictions(rows):
        return np.asarray(
            [CLASSES.index(row["result"]["class"]) if row["result"]["class"] in CLASSES else -1 for row in rows]
        )

    candidate_predictions, baseline_predictions = predictions(candidate_rows), predictions(baseline_rows)
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(20):
        indices = rng.integers(0, len(labels), (500, len(labels)))
        deltas.append(
            metric_samples(labels[indices], candidate_predictions[indices])
            - metric_samples(labels[indices], baseline_predictions[indices])
        )
    return {
        "candidate": candidate["id"],
        "baseline": baseline["id"],
        "samples": len(labels),
        "metric_delta": candidate["metrics"]["metric"] - baseline["metrics"]["metric"],
        "sp_f1_delta": candidate["metrics"]["sp_f1"] - baseline["metrics"]["sp_f1"],
        "fixed": int(np.sum((candidate_predictions == labels) & (baseline_predictions != labels))),
        "regressed": int(np.sum((candidate_predictions != labels) & (baseline_predictions == labels))),
        "paired_metric_bootstrap_95_interval": np.percentile(np.concatenate(deltas), [2.5, 97.5]).tolist(),
        "bootstrap_seed": 42,
        "bootstrap_replicates": 10000,
        "scope": "historical image-level pairs; not independent collection groups or full training uncertainty",
    }


def main():
    selection = read_json(REPORTS / "selection.json")
    tests = {name: read_json(REPORTS / f"{name}.test.json") for name in selection["test_allowed_ids"]}
    comparisons = []
    selected = selection["selected_variant"]
    if selected:
        for seed in SEEDS:
            winner = tests[candidate_name(selected, seed)]
            comparisons.append(paired_comparison(winner, tests["production"]))
            if selected != "baseline":
                comparisons.append(paired_comparison(winner, tests[candidate_name("baseline", seed)]))
    protocol = read_json(DATA / "protocol.json")
    production_hash = digest(ROOT / "inference/models/three_class.onnx")
    if production_hash != protocol["production_onnx_sha256"]:
        raise RuntimeError("Production model changed during experiment")
    summary = {
        "selected_variant": selected,
        "primary_seed": 42,
        "selection_sha256": digest(REPORTS / "selection.json"),
        "validation_ranking": selection["ranking"],
        "test_metrics": {
            name: {
                "metrics": row["metrics"],
                "canary": {key: row["canary"][key] for key in ("correct", "total")},
                "model_sha256": row["model_sha256"],
            }
            for name, row in tests.items()
        },
        "comparisons": comparisons,
        "production_onnx_sha256": production_hash,
        "production_replaced": False,
        "promotion_status": "NOT_READY: independent challenge, true-OOD, and collection-group isolation incomplete",
    }
    write_new(REPORTS / "results.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
