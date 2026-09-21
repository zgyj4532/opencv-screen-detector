"""Export and evaluate fixed candidates through the actual deployment Predictor."""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import precision_recall_fscore_support

from experiment.cnn_fft_dwt_ablation import harness
from experiment.cnn_fft_dwt_ablation.finalize_export import export
from experiment.daily_20260922.src.protocol import (
    DATA,
    MODELS,
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
from experiment.daily_20260922.src.train import BoundedCache, checked_split
from inference.config import configure
from inference.log import logger
from inference.predictor import ScreenDetectorPredictor
from trainer.evaluation_sets import evaluation_set_readiness, resolve_manifest_samples
from trainer.model import load_model

CLASSES = ["natural", "screenshot", "screen_photo"]


def metrics(labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, labels=[0, 1, 2], zero_division=0)
    accuracy = float((np.asarray(labels) == np.asarray(predictions)).mean())
    return {
        "accuracy": accuracy,
        "sp_precision": float(precision[2]),
        "sp_recall": float(recall[2]),
        "sp_f1": float(f1[2]),
        "macro_f1": float(f1.mean()),
        "metric": float(0.5 * f1[2] + 0.3 * accuracy + 0.2 * f1.mean()),
        "unknown": sum(prediction == -1 for prediction in predictions),
    }


def real_input_parity(checkpoint, model_path, samples):
    model = load_model(str(checkpoint), use_dwt=True).eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    cache = BoundedCache()
    selected = [next(row for row in samples if row[1] == label) for label in range(3)]
    max_error = 0.0
    for path, _ in selected:
        entry = cache[path]
        rgb = harness.eval_tf()(image=entry["rgb"])["image"].unsqueeze(0)
        tensors = [rgb, *(torch.from_numpy(entry[key].astype(np.float32)).unsqueeze(0) for key in ("fft", "dwt"))]
        with torch.no_grad():
            expected = model(*tensors).numpy()
        inputs = dict(
            zip([item.name for item in session.get_inputs()], [tensor.numpy() for tensor in tensors], strict=True)
        )
        actual = session.run(None, inputs)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)
        max_error = max(max_error, float(np.abs(actual - expected).max()))
    return {"status": "PASS", "real_samples": len(selected), "max_absolute_logit_error": max_error}


def main():
    prepare_directories()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=[*VARIANTS, "production"])
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--role", choices=["val", "test"], default="val")
    args = parser.parse_args()
    name = "production" if args.variant == "production" else candidate_name(args.variant, args.seed)
    output = REPORTS / f"{name}.{args.role}.json"
    if output.exists():
        raise RuntimeError("Evaluation already recorded; use existing evidence")
    split = checked_split()
    protocol = read_json(DATA / "protocol.json")
    torch.set_num_threads(4)
    if args.role == "test":
        selection = read_json(REPORTS / "selection.json")
        if name not in selection["test_allowed_ids"]:
            raise RuntimeError("This candidate was not selected before opening test")
        if digest(DATA / "protocol.json") != selection["protocol_sha256"]:
            raise RuntimeError("Protocol changed since selection")
    if name == "production":
        model_path = ROOT / "inference/models/three_class.onnx"
        if digest(model_path) != protocol["production_onnx_sha256"]:
            raise RuntimeError("Production model changed since experiment started")
        parity = {"status": "existing_production_artifact"}
        checkpoint_sha256 = None
    else:
        train = read_json(REPORTS / f"{name}.train.json")
        if train["result"]["status"] != "complete" or train["split"] != split["meta"]:
            raise RuntimeError("Training incomplete or split identity changed")
        checkpoint = harness.EXP_DIR / name / "best.pth"
        checkpoint_sha256 = digest(checkpoint)
        if checkpoint_sha256 != train["best_checkpoint_sha256"]:
            raise RuntimeError("Selected checkpoint changed")
        model_path = MODELS / f"{name}.onnx"
        if args.role == "val":
            export(str(checkpoint), str(model_path))
            parity = real_input_parity(checkpoint, model_path, split["val"])
        else:
            validation = read_json(REPORTS / f"{name}.val.json")
            if digest(REPORTS / f"{name}.val.json") != selection["validation_report_sha256"][name]:
                raise RuntimeError("Validation evidence changed since selection")
            if digest(model_path) != validation["model_sha256"]:
                raise RuntimeError("ONNX changed after validation")
            parity = validation["parity"]
    configure(**THRESHOLDS)
    logger.remove()
    predictor = ScreenDetectorPredictor(model_path=model_path)
    for path, _ in split["val"][:3]:
        predictor.predict(Path(path))
    records, elapsed = [], []
    for index, (path, label) in enumerate(split[args.role]):
        started = time.perf_counter()
        result = predictor.predict(Path(path))
        elapsed.append((time.perf_counter() - started) * 1000)
        records.append({"path": Path(path).relative_to(ROOT).as_posix(), "label": label, "result": result})
        if (index + 1) % 100 == 0:
            print(f"{name} {args.role}: {index + 1}/{len(split[args.role])}", flush=True)
    canaries = [
        {"id": sample.sample_id, "expected": sample.expected_label, "result": predictor.predict(sample.path)}
        for sample in resolve_manifest_samples("canary")
    ]
    labels = [row["label"] for row in records]
    predictions = [
        CLASSES.index(row["result"]["class"]) if row["result"]["class"] in CLASSES else -1 for row in records
    ]
    report = {
        "id": name,
        "role": args.role,
        "samples": len(records),
        "split": split["meta"],
        "protocol_sha256": digest(DATA / "protocol.json"),
        "model_sha256": digest(model_path),
        "checkpoint_sha256": checkpoint_sha256,
        "parity": parity,
        "thresholds": THRESHOLDS,
        "metrics": metrics(labels, predictions),
        "latency_ms": {
            "mean": float(np.mean(elapsed)),
            "p50": float(np.median(elapsed)),
            "p95": float(np.percentile(elapsed, 95)),
        },
        "canary": {
            "correct": sum(row["expected"] == row["result"]["class"] for row in canaries),
            "total": len(canaries),
            "records": canaries,
        },
        "readiness": evaluation_set_readiness(),
        "production_replaced": False,
        "records": records,
    }
    write_new(output, report)
    print({key: report[key] for key in ("id", "role", "metrics", "canary")}, flush=True)


if __name__ == "__main__":
    main()
