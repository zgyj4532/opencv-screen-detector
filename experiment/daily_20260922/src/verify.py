"""Read back dataset/model identities and check the recorded experiment evidence."""

import argparse
import importlib.metadata
import sys

from experiment.daily_20260922.src.protocol import (
    DATA,
    EXPERIMENT,
    MODELS,
    REPORTS,
    ROOT,
    SEEDS,
    VARIANTS,
    candidate_name,
    digest,
    read_json,
    write_new,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    manifest, protocol = read_json(DATA / "ingest.json"), read_json(DATA / "protocol.json")
    for row in manifest["archives"]:
        if digest(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"Original archive changed: {row['path']}")
    for row in manifest["records"]:
        if digest(ROOT / row["target"]) != row["sha256"]:
            raise RuntimeError(f"Imported image changed: {row['target']}")
    split = read_json(DATA / "split.json")
    current = read_json(ROOT / "experiment/cnn_fft_dwt_ablation/split.json")
    if split != current:
        raise RuntimeError("Current split differs from the frozen experiment snapshot")
    previous = read_json(EXPERIMENT / "archive/preexisting_split.json")
    if any(previous[role] != split[role] for role in ("val", "test")):
        raise RuntimeError("Evaluation identities changed during ingestion")
    for path, expected in (
        (ROOT / "trainer/checkpoints/three_class_best.pth", protocol["init_checkpoint_sha256"]),
        (ROOT / "inference/models/three_class.onnx", protocol["production_onnx_sha256"]),
    ):
        if digest(path) != expected:
            raise RuntimeError(f"Canonical artifact changed: {path}")
    summary = {
        "archives_verified": len(manifest["archives"]),
        "imported_images_verified": len(manifest["records"]),
        "frozen_evaluation_unchanged": True,
        "canonical_artifacts_unchanged": True,
        "dataset_fingerprint": protocol["split"]["dataset_fingerprint"],
        "evaluation_fingerprint": protocol["split"]["evaluation_fingerprint"],
    }
    if args.inputs_only:
        print(summary, flush=True)
        return
    selection = read_json(REPORTS / "selection.json")
    for variant in VARIANTS:
        for seed in SEEDS:
            name = candidate_name(variant, seed)
            training = read_json(REPORTS / f"{name}.train.json")
            validation_path = REPORTS / f"{name}.val.json"
            validation = read_json(validation_path)
            if training["result"]["status"] != "complete" or training["result"]["test"] is not None:
                raise RuntimeError(f"Training incomplete or opened test: {name}")
            if digest(validation_path) != selection["validation_report_sha256"][name]:
                raise RuntimeError(f"Validation changed after selection: {name}")
            if digest(MODELS / f"{name}.onnx") != validation["model_sha256"]:
                raise RuntimeError(f"ONNX changed: {name}")
            if validation["parity"]["status"] != "PASS":
                raise RuntimeError(f"ONNX parity missing: {name}")
    recorded_tests = {path.name.removesuffix(".test.json") for path in REPORTS.glob("*.test.json")}
    if recorded_tests != set(selection["test_allowed_ids"]):
        raise RuntimeError("Test records do not match the preselected comparison")
    for name in selection["test_allowed_ids"]:
        report = read_json(REPORTS / f"{name}.test.json")
        if report["samples"] != len(split["test"]) or report["split"] != protocol["split"]:
            raise RuntimeError(f"Test denominator or data identity changed: {name}")
    summary.update(
        {
            "complete_training_runs": len(VARIANTS) * len(SEEDS),
            "test_records": sorted(recorded_tests),
            "validation_only_selection_verified": True,
            "python": sys.version,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "numpy", "onnx", "onnxruntime", "timm", "albumentations")
            },
            "source_sha256": {
                path.relative_to(ROOT).as_posix(): digest(path)
                for path in [
                    *sorted((EXPERIMENT / "src").glob("*.py")),
                    ROOT / "experiment/cnn_fft_dwt_ablation/harness.py",
                ]
            },
        }
    )
    write_new(REPORTS / "verification.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
