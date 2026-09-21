"""Fixed protocol for the September 15–21 incremental-data experiment."""

import hashlib
import json
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
DATA = EXPERIMENT / "data"
REPORTS = EXPERIMENT / "reports"
MODELS = EXPERIMENT / "models"
LOGS = EXPERIMENT / "logs"
SEEDS = (42, 2024)
VARIANTS = {
    "baseline": {"lr": 3e-4, "distill_alpha": 0.0},
    "low_lr": {"lr": 1e-4, "distill_alpha": 0.0},
    "lwf": {"lr": 3e-4, "distill_alpha": 1.0},
}
EPOCHS = 8
THRESHOLDS = {"ood_threshold": 0.45, "screen_photo_threshold": 0.60, "confidence_high": 0.92, "confidence_medium": 0.75}


def prepare_directories():
    for directory in (DATA, REPORTS, MODELS, LOGS):
        directory.mkdir(parents=True, exist_ok=True)


def candidate_name(variant, seed):
    return f"daily20260922_{variant}_s{seed}_b8"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    """Evidence is immutable: fail rather than silently replacing an earlier run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
