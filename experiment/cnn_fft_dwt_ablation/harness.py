"""Autonomous experiment harness for screen-detector V3.

Clean, fast, reproducible training pipeline that fixes the audited bugs:
- Content-level deduplication with explicit reviewed decisions for label conflicts
- Content represented under hard_negative remains train-only
- Portable frozen val/test identities reused across experiments; new content enters train
- FFT/DWT features cached to disk (computed from raw RGB image, matches inference)
- Fixed unfreeze (only real MBConv stages + conv_head + bn2)
- Stage-A optimizer includes LayerNorm params
- EMA, label smoothing, adaptive threshold on held-out TEST
- Release checkpoint selection maximizes the VAL metric only. Confirmed canaries
  remain a separate regression gate and do not count as generalization evidence.

The release trainer imports this harness with the ablation-winning defaults;
legacy trainer modules remain available separately for baseline reproduction.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.core.composition import BaseCompose
from albumentations.core.transforms_interface import BasicTransform
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.fft_transform import (
    compute_dwt_features,
    compute_fft_spectrum,
)
from trainer.evaluation_sets import (
    CANARY_MANIFEST,
)
from trainer.evaluation_sets import (
    load_canary_paths as load_manifest_canary_paths,
)
from trainer.model import create_model

DATA_DIR = ROOT / "data" / "input"
ABLATION_DIR = ROOT / "experiment" / "cnn_fft_dwt_ablation"
CACHE_DIR = ABLATION_DIR / "cache"
EXP_DIR = ABLATION_DIR / "exp"
SPLIT_PATH = ABLATION_DIR / "split.json"
LEADERBOARD = ABLATION_DIR / "leaderboard.jsonl"
# Deprecated text-manifest alias retained for historical scripts/checkpoints.
FOCUS_MANIFEST = ROOT / "trainer" / "hard_examples.txt"
LABEL_OVERRIDES_PATH = ROOT / "trainer" / "content_label_overrides.json"
SPLIT_SCHEMA_VERSION = 4
SPLIT_ASSIGNMENT_POLICY = "frozen_val_test_new_content_to_train_hard_content_train_only"
IMAGE_SIZE = 224
CLASS_NAMES = ["natural", "screenshot", "screen_photo"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# hard_negative subdir -> true label
HN_MAP = {
    "screen_photo_to_screenshot": 2,
    "screen_photo_to_natural": 2,
    "screenshot_to_screen_photo": 1,
    "screenshot_to_natural": 1,
    "natural_to_screenshot": 0,
    "natural_to_screen_photo": 0,
    # generic UI-like hard negatives are real screenshots
    "anime_ui": 1,
    "dark_mode": 1,
    "figma_design": 1,
    "ide_editor": 1,
    "presentation": 1,
    "table_chart": 1,
    "text_document": 1,
    "ui_poster": 1,
}


# --------------------------------------------------------------------------
# Data collection & split
# --------------------------------------------------------------------------
_CONTENT_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def content_sha256(path: str | Path) -> str:
    """Return a content identity that is stable across path and mtime changes."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    cache_key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    if cached := _CONTENT_HASH_CACHE.get(cache_key):
        return cached

    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _CONTENT_HASH_CACHE[cache_key] = value
    return value


def load_label_overrides(path: Path = LABEL_OVERRIDES_PATH) -> dict[str, int]:
    """Load human-reviewed labels keyed by lowercase SHA-256 content hash."""
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported content-label override schema: {path}")

    resolved: dict[str, int] = {}
    for digest, entry in payload.get("overrides", {}).items():
        label_value = entry["label"] if isinstance(entry, dict) else entry
        if isinstance(label_value, str):
            if label_value not in CLASS_NAMES:
                raise RuntimeError(f"Unknown label {label_value!r} for content {digest}")
            label = CLASS_NAMES.index(label_value)
        else:
            label = int(label_value)
        normalized = digest.lower()
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise RuntimeError(f"Invalid SHA-256 in {path}: {digest}")
        if label not in range(len(CLASS_NAMES)):
            raise RuntimeError(f"Invalid label {label} for content {digest}")
        resolved[normalized] = label
    return resolved


def _scan_sample_candidates(data_dir: Path) -> list[tuple[str, int, bool, str]]:
    candidates: list[tuple[str, int, bool, str]] = []
    main_map = {"natural_photo": 0, "screenshot": 1, "screen_photo": 2}
    for folder, label in main_map.items():
        candidates.extend(
            (str(path), label, False, content_sha256(path))
            for path in sorted((data_dir / folder).rglob("*"))
            if path.suffix.lower() in IMAGE_EXTS
        )

    hn_root = data_dir / "hard_negative"
    for sub, label in HN_MAP.items():
        directory = hn_root / sub
        if not directory.exists():
            continue
        candidates.extend(
            (str(path), label, True, content_sha256(path))
            for path in sorted(directory.rglob("*"))
            if path.suffix.lower() in IMAGE_EXTS
        )
    return candidates


def collect_samples(
    data_dir: Path = DATA_DIR,
    label_overrides: dict[str, int] | None = None,
    preferred_paths: set[str] | None = None,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Return content-deduplicated samples after resolving reviewed label conflicts."""
    overrides = load_label_overrides() if label_overrides is None else label_overrides
    preferred = {str(Path(path).resolve()) for path in (preferred_paths or set())}
    groups: dict[str, list[tuple[str, int, bool, str]]] = defaultdict(list)
    for candidate in _scan_sample_candidates(data_dir):
        groups[candidate[3]].append(candidate)

    main: list[tuple[str, int]] = []
    hard: list[tuple[str, int]] = []
    root = data_dir.resolve()
    for digest, candidates in sorted(groups.items()):
        labels = {candidate[1] for candidate in candidates}
        if digest in overrides:
            label = overrides[digest]
        elif len(labels) == 1:
            label = next(iter(labels))
        else:
            details = "; ".join(
                f"{Path(path).resolve().relative_to(root).as_posix()}={CLASS_NAMES[candidate_label]}"
                for path, candidate_label, _is_hard, _digest in sorted(candidates)
            )
            raise RuntimeError(
                f"Conflicting labels for content {digest}: {details}. "
                f"Add a reviewed decision to {LABEL_OVERRIDES_PATH}."
            )

        def candidate_key(
            candidate: tuple[str, int, bool, str], resolved_label: int = label
        ) -> tuple[int, int, int, str]:
            path, candidate_label, is_hard, _digest = candidate
            resolved_path = str(Path(path).resolve())
            relative = Path(path).resolve().relative_to(root).as_posix()
            return (
                int(resolved_path not in preferred),
                int(candidate_label != resolved_label),
                int(is_hard),
                relative,
            )

        canonical = min(candidates, key=candidate_key)
        sample = (canonical[0], label)
        # A content identity that was curated into hard_negative remains train-only,
        # even when a byte-identical copy also exists under a main class directory.
        if any(candidate[2] for candidate in candidates):
            hard.append(sample)
        else:
            main.append(sample)

    main.sort(key=lambda sample: Path(sample[0]).resolve().relative_to(root).as_posix())
    hard.sort(key=lambda sample: Path(sample[0]).resolve().relative_to(root).as_posix())
    return main, hard


def load_canary_paths(data_dir: Path = DATA_DIR, manifest_path: Path = CANARY_MANIFEST) -> set[str]:
    """Load the canonical Canary set used for regression checks and optional sampling."""
    if data_dir.resolve() == DATA_DIR.resolve() and manifest_path.resolve() == CANARY_MANIFEST.resolve():
        return load_manifest_canary_paths(manifest_path=manifest_path, repo_root=ROOT)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths: set[str] = set()
    for entry in payload.get("entries", []):
        relative = Path(entry["path"])
        parts = relative.parts
        if len(parts) >= 3 and parts[:2] == ("data", "input"):
            relative = Path(*parts[2:])
        path = (data_dir / relative).resolve()
        if path.exists() and path.suffix.lower() in IMAGE_EXTS:
            paths.add(str(path))
    return paths


def load_focus_paths(data_dir: Path = DATA_DIR, manifest_path: Path = FOCUS_MANIFEST) -> set[str]:
    """Deprecated adapter for historical text manifests; use load_canary_paths."""
    if manifest_path.resolve() == FOCUS_MANIFEST.resolve():
        return load_canary_paths(data_dir=data_dir)
    if not manifest_path.exists():
        return set()

    paths: set[str] = set()
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        path = (data_dir / Path(line)).resolve()
        if path.exists() and path.suffix.lower() in IMAGE_EXTS:
            paths.add(str(path))
    return paths


def _dataset_fingerprint(
    samples: list[tuple[str, int]],
) -> str:
    """Fingerprint unique content and reviewed labels, independent of path or mtime."""
    rows = [f"{content_sha256(raw_path)}\0{label}" for raw_path, label in samples]
    payload = "\n".join(sorted(rows)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _evaluation_fingerprint(split: dict) -> str:
    rows = []
    for role in ("val", "test"):
        rows.extend(f"{role}\0{content_sha256(path)}\0{label}" for path, label in split[role])
    return hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()


def _canary_fingerprint(canary_paths: set[str]) -> str:
    """Fingerprint Canary examples by content so the split stays portable."""
    hashes = sorted({content_sha256(path) for path in canary_paths})
    return hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()


def _serialize_split(split: dict, data_dir: Path) -> dict:
    root = data_dir.resolve()
    return {
        "meta": split["meta"],
        **{
            role: [
                [
                    Path(path).resolve().relative_to(root).as_posix(),
                    label,
                    content_sha256(path),
                ]
                for path, label in split[role]
            ]
            for role in ("train", "val", "test")
        },
    }


def _load_frozen_split(saved: dict, current: dict[str, tuple[str, int, bool]]) -> dict:
    split = {"train": [], "val": [], "test": []}
    assigned: set[str] = set()
    for role in ("train", "val", "test"):
        for _relative, expected_label, digest in saved[role]:
            normalized = digest.lower()
            candidate = current.get(normalized)
            if candidate is None:
                if role in {"val", "test"}:
                    raise RuntimeError(
                        f"Frozen {role} content is missing: {digest}. Restore it or explicitly version a new split."
                    )
                continue
            path, label, is_hard = candidate
            if label != expected_label:
                raise RuntimeError(
                    f"Frozen {role} label changed for {digest}: expected {expected_label}, current {label}. "
                    "Review the label override before versioning a new split."
                )
            if role in {"val", "test"} and is_hard:
                raise RuntimeError(f"Frozen {role} content now exists only as a hard-negative sample: {digest}")
            if normalized in assigned:
                raise RuntimeError(f"Frozen split repeats content across roles: {digest}")
            assigned.add(normalized)
            split[role].append((path, label))

    for digest, (path, label, _is_hard) in sorted(current.items()):
        if digest not in assigned:
            split["train"].append((path, label))
            assigned.add(digest)
    return split


def build_split(
    seed: int = 42,
    data_dir: Path = DATA_DIR,
    split_path: Path = SPLIT_PATH,
    canary_paths: set[str] | None = None,
    focus_paths: set[str] | None = None,
) -> dict:
    """Build a content-clean split whose validation and test identities never reshuffle."""
    if canary_paths is not None and focus_paths is not None:
        raise ValueError("Pass canary_paths or deprecated focus_paths, not both")
    canaries = canary_paths if canary_paths is not None else focus_paths
    if canaries is None:
        canaries = load_canary_paths(data_dir)
    main, hard = collect_samples(data_dir, preferred_paths=canaries)
    all_samples = main + hard
    fingerprint = _dataset_fingerprint(all_samples)
    collected_paths = {str(Path(path).resolve()) for path, _ in all_samples}
    if missing_canaries := canaries - collected_paths:
        missing = "\n".join(f"- {path}" for path in sorted(missing_canaries))
        raise RuntimeError(f"Canary examples are not part of the collected dataset:\n{missing}")
    canary_fingerprint = _canary_fingerprint(canaries)

    sample_kind = {content_sha256(path): False for path, _label in main}
    sample_kind.update({content_sha256(path): True for path, _label in hard})
    current = {
        content_sha256(path): (str(Path(path).resolve()), label, sample_kind[content_sha256(path)])
        for path, label in all_samples
    }

    saved = json.loads(split_path.read_text(encoding="utf-8")) if split_path.exists() else None
    if saved and saved.get("meta", {}).get("schema_version") == SPLIT_SCHEMA_VERSION:
        saved_seed = saved["meta"].get("seed")
        if saved_seed != seed:
            raise RuntimeError(f"Frozen split uses seed {saved_seed}; refusing to replace it with seed {seed}")
        split = _load_frozen_split(saved, current)
        train_paths = {str(Path(path).resolve()) for path, _ in split["train"]}
        if missing_train_canaries := canaries - train_paths:
            missing = "\n".join(f"- {path}" for path in sorted(missing_train_canaries))
            raise RuntimeError(f"Frozen split places Canary content outside train or cannot resolve it:\n{missing}")
        split["meta"] = {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "seed": seed,
            "dataset_fingerprint": fingerprint,
            "evaluation_fingerprint": _evaluation_fingerprint(split),
            "focus_fingerprint": canary_fingerprint,
            "canary_fingerprint": canary_fingerprint,
            "main_count": len(main),
            "hard_negative_count": len(hard),
            "focus_count": len(canaries),
            "canary_count": len(canaries),
            "assignment_policy": SPLIT_ASSIGNMENT_POLICY,
        }
        serialized = _serialize_split(split, data_dir)
        if serialized != saved:
            split_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
        return split

    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for label in range(3):
        items = [s for s in main if s[1] == label]
        idx = rng.permutation(len(items))
        n = len(items)
        n_tr = int(n * 0.70)
        n_va = int(n * 0.15)
        label_train, label_val, label_test = [], [], []
        for j, i in enumerate(idx):
            if j < n_tr:
                label_train.append(items[i])
            elif j < n_tr + n_va:
                label_val.append(items[i])
            else:
                label_test.append(items[i])

        # Preserve the historical permutation whenever focus samples already
        # landed in train. Otherwise swap within the same class to avoid
        # turning a confirmed regression example into evaluation leakage.
        for held_out in (label_val, label_test):
            for held_index, sample in enumerate(held_out):
                if str(Path(sample[0]).resolve()) not in canaries:
                    continue
                replacement_index = next(
                    (
                        index
                        for index in range(len(label_train) - 1, -1, -1)
                        if str(Path(label_train[index][0]).resolve()) not in canaries
                    ),
                    None,
                )
                if replacement_index is None:
                    raise RuntimeError(f"No train sample available to swap with Canary example: {sample[0]}")
                label_train[replacement_index], held_out[held_index] = (
                    held_out[held_index],
                    label_train[replacement_index],
                )

        train.extend(label_train)
        val.extend(label_val)
        test.extend(label_test)
    train.extend(hard)  # hard negatives -> train only

    split = {
        "meta": {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "seed": seed,
            "dataset_fingerprint": fingerprint,
            "evaluation_fingerprint": "",
            "focus_fingerprint": canary_fingerprint,
            "canary_fingerprint": canary_fingerprint,
            "main_count": len(main),
            "hard_negative_count": len(hard),
            "focus_count": len(canaries),
            "canary_count": len(canaries),
            "assignment_policy": SPLIT_ASSIGNMENT_POLICY,
        },
        "train": train,
        "val": val,
        "test": test,
    }
    split["meta"]["evaluation_fingerprint"] = _evaluation_fingerprint(split)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(_serialize_split(split, data_dir), ensure_ascii=False, indent=2), encoding="utf-8")
    return split


# --------------------------------------------------------------------------
# FFT/DWT cache
# --------------------------------------------------------------------------
def _cache_key(path: str) -> Path:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    identity = f"{resolved}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    h = hashlib.sha256(identity).hexdigest()
    return CACHE_DIR / f"{h}.npz"


def _load_rgb(path: str) -> np.ndarray:
    try:
        if Path(path).stat().st_size > 10 * 1024 * 1024:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


def precompute_cache(samples: list[tuple[str, int]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [s for s in samples if not _cache_key(s[0]).exists()]
    print(
        f"[cache] {len(samples) - len(todo)} cached, computing {len(todo)}...",
        flush=True,
    )
    for i, (path, _lab) in enumerate(todo):
        rgb = _load_rgb(path)
        fft = compute_fft_spectrum(rgb, IMAGE_SIZE, color_space="rgb").squeeze(0)  # (1,H,W)
        dwt = compute_dwt_features(rgb, IMAGE_SIZE, color_space="rgb").squeeze(0)  # (4,H,W)
        np.savez_compressed(
            _cache_key(path),
            fft=fft.astype(np.float16),
            dwt=dwt.astype(np.float16),
        )
        if (i + 1) % 200 == 0:
            print(f"[cache] {i + 1}/{len(todo)}", flush=True)
    print("[cache] done", flush=True)


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------
def train_tf(heavy: bool = False, seed: int | None = None) -> A.Compose:
    if heavy:
        from trainer.augment import get_train_transforms

        transform = get_train_transforms()
    else:
        transform = A.Compose(
            [
                A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.6, 1.0), ratio=(0.75, 1.333)),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.Perspective(scale=(0.02, 0.08), p=0.4),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.25),
                A.ImageCompression(quality_range=(40, 95), p=0.3),
                A.GaussNoise(std_range=(0.05, 0.2), p=0.25),
                A.CoarseDropout(
                    num_holes_range=(1, 6),
                    hole_height_range=(8, 28),
                    hole_width_range=(8, 28),
                    fill=0,
                    p=0.25,
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    if seed is not None:
        transform.set_random_seed(seed)
    return transform


def eval_tf() -> A.Compose:
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class CachedDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform: A.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        entry = RAM[path]
        rgb_t = self.transform(image=entry["rgb"])["image"]
        fft_t = torch.from_numpy(entry["fft"].astype(np.float32))
        dwt_t = torch.from_numpy(entry["dwt"].astype(np.float32))
        return rgb_t, fft_t, dwt_t, label


# In-RAM store built once per process: path -> {rgb(256,256,3 uint8), fft(1,224,224 f16), dwt(4,224,224 f16)}
RAM: dict[str, dict] = {}


def preload_ram(samples: list[tuple[str, int]]) -> None:
    todo = [s for s in samples if s[0] not in RAM]
    print(f"[ram] preloading {len(todo)} images into memory...", flush=True)
    for i, (path, _lab) in enumerate(todo):
        rgb = _load_rgb(path)
        rgb256 = cv2.resize(rgb, (256, 256))
        cache = np.load(_cache_key(path))
        RAM[path] = {"rgb": rgb256, "fft": cache["fft"], "dwt": cache["dwt"]}
        if (i + 1) % 500 == 0:
            print(f"[ram] {i + 1}/{len(todo)}", flush=True)
    print(f"[ram] loaded {len(RAM)} images", flush=True)


def make_sampler(
    samples: list[tuple[str, int]],
    beta: float | None,
    canary_paths: set[str] | None = None,
    canary_weight: float = 1.0,
    seed: int | None = None,
    *,
    focus_paths: set[str] | None = None,
    focus_weight: float | None = None,
    boost_paths: set[str] | None = None,
    boost_weight: float = 1.0,
) -> WeightedRandomSampler:
    """Balance classes and optionally upweight Canary regression examples.

    ``focus_*`` keyword arguments are deprecated compatibility aliases.
    ``boost_paths`` upweights newly ingested train-only content (LwF rehearsal).
    """
    if canary_paths is not None and focus_paths is not None:
        raise ValueError("Pass canary_paths or deprecated focus_paths, not both")
    canaries = canary_paths if canary_paths is not None else (focus_paths or set())
    if focus_weight is not None:
        if canary_weight != 1.0:
            raise ValueError("Pass canary_weight or deprecated focus_weight, not both")
        canary_weight = focus_weight
    boosts = {str(Path(path).resolve()) for path in (boost_paths or set())}
    labels = [s[1] for s in samples]
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if beta is None:  # inverse frequency
        cls_w = len(labels) / np.maximum(counts, 1)
    else:  # class-balanced (effective number)
        cls_w = (1 - beta) / (1 - np.power(beta, np.maximum(counts, 1)))
    w = np.array(
        [
            cls_w[label]
            * (canary_weight if str(Path(path).resolve()) in canaries else 1.0)
            * (boost_weight if str(Path(path).resolve()) in boosts else 1.0)
            for path, label in samples
        ],
        dtype=np.float64,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return WeightedRandomSampler(torch.from_numpy(w), num_samples=len(w), replacement=True, generator=generator)


def seed_everything(seed: int) -> None:
    """Seed every RNG used by the release trainer and require deterministic kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _transform_nodes(transform: BasicTransform | BaseCompose) -> list[BasicTransform | BaseCompose]:
    nodes: list[BasicTransform | BaseCompose] = [transform]
    if isinstance(transform, BaseCompose):
        for child in transform.transforms:
            nodes.extend(_transform_nodes(child))
    return nodes


def _capture_runtime_rng(transform: A.Compose, sampler: WeightedRandomSampler) -> dict:
    """Capture every RNG that advances while producing training batches."""
    transform_states = [
        {
            "numpy": copy.deepcopy(node.random_generator.bit_generator.state),
            "python": node.py_random.getstate(),
        }
        for node in _transform_nodes(transform)
    ]
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "sampler": sampler.generator.get_state() if sampler.generator is not None else None,
        "transforms": transform_states,
    }


def _restore_runtime_rng(state: dict, transform: A.Compose, sampler: WeightedRandomSampler) -> None:
    """Restore captured training RNG state before the next epoch starts."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    if sampler.generator is not None and state["sampler"] is not None:
        sampler.generator.set_state(state["sampler"])

    nodes = _transform_nodes(transform)
    if len(nodes) != len(state["transforms"]):
        raise RuntimeError("Training transform structure changed; refusing an inexact resume")
    for node, node_state in zip(nodes, state["transforms"], strict=True):
        node.random_generator.bit_generator.state = node_state["numpy"]
        node.py_random.setstate(node_state["python"])


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
class FocalLossLS(nn.Module):
    def __init__(self, gamma: float, alpha: list[float] | None, smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.register_buffer("alpha", torch.tensor(alpha) if alpha else None)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: str | None = None,
    ) -> torch.Tensor:
        n = logits.size(1)
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        if self.smoothing > 0:
            true = F.one_hot(targets, n).float()
            true = true * (1 - self.smoothing) + self.smoothing / n
            ce = -(true * logp).sum(1)
            pt = (p * F.one_hot(targets, n).float()).sum(1)
        else:
            ce = F.nll_loss(logp, targets, reduction="none")
            pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            loss = self.alpha.to(logits.device)[targets] * loss
        red = "mean" if reduction is None else reduction
        if red == "none":
            return loss
        if red == "sum":
            return loss.sum()
        return loss.mean()


def remix_label_lambda(
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: torch.Tensor,
    class_counts: torch.Tensor,
    kappa: float = 3.0,
    tau: float = 0.5,
) -> torch.Tensor:
    """Return Remix λ_y given Mixup λ and class counts.

    Chou et al., "Remix: Rebalanced Mixup", ECCV 2020 Workshops. Features stay
    mixed with ``lam``; the label mix is biased toward the minority class when
    the count ratio is at least ``kappa`` and ``lam`` is below ``tau``.
    """
    counts = class_counts.to(device=lam.device, dtype=lam.dtype)
    n_a = counts[labels_a]
    n_b = counts[labels_b]
    ratio = n_a / n_b.clamp_min(1.0)
    lam_y = lam.to(dtype=counts.dtype)
    assign_b = (ratio >= kappa) & (lam_y < tau)
    assign_a = (ratio <= (1.0 / kappa)) & ((1.0 - lam_y) < tau)
    lam_y = torch.where(assign_b, torch.zeros_like(lam_y), lam_y)
    return torch.where(assign_a, torch.ones_like(lam_y), lam_y)


def mix_modal_batch(
    rgb: torch.Tensor,
    fft: torch.Tensor,
    dwt: torch.Tensor,
    index: torch.Tensor,
    lam: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linearly mix RGB, FFT, and DWT with the same Mixup λ (Zhang et al., ICLR 2018)."""
    weight = lam.view(-1, *([1] * (rgb.ndim - 1))).to(dtype=rgb.dtype)
    mixed_rgb = weight * rgb + (1.0 - weight) * rgb.index_select(0, index)
    mixed_fft = weight * fft + (1.0 - weight) * fft.index_select(0, index)
    mixed_dwt = weight * dwt + (1.0 - weight) * dwt.index_select(0, index)
    return mixed_rgb, mixed_fft, mixed_dwt


def mixed_focal_loss(
    criterion: FocalLossLS,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam_y: torch.Tensor,
) -> torch.Tensor:
    """Mean of per-sample Focal losses interpolated with Remix/Mixup λ_y."""
    loss_a = criterion(logits, labels_a, reduction="none")
    loss_b = criterion(logits, labels_b, reduction="none")
    weights = lam_y.to(dtype=loss_a.dtype)
    return (weights * loss_a + (1.0 - weights) * loss_b).mean()


def distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Hinton et al. distillation KL, scaled by T^2; used by LwF (Li & Hoiem, ECCV 2016)."""
    temp = teacher_logits.new_tensor(temperature)
    log_p = F.log_softmax(student_logits / temp, dim=1)
    q = F.softmax(teacher_logits / temp, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (temp * temp)


def load_init_checkpoint(model: nn.Module, path: str | Path) -> dict:
    """Load a harness/release checkpoint's ``model_state_dict`` into ``model``."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"Checkpoint is missing model_state_dict: {path}")
    model.load_state_dict(payload["model_state_dict"])
    return payload


# --------------------------------------------------------------------------
# EMA
# --------------------------------------------------------------------------
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for e, m in zip(self.ema.state_dict().values(), model.state_dict().values(), strict=True):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


# --------------------------------------------------------------------------
# Unfreeze (fixed)
# --------------------------------------------------------------------------
def unfreeze_stages(model: nn.Module, n_stages: int) -> None:
    bb = model.backbone
    for p in bb.parameters():
        p.requires_grad_(False)
    if n_stages > 0:
        stage_container = getattr(bb, "blocks", None) or getattr(bb, "stages", None)
        if stage_container is not None:
            stages = list(stage_container.children())
            if not stages and isinstance(stage_container, (nn.ModuleList, nn.Sequential)):
                stages = list(stage_container)
            for stage in stages[-n_stages:]:
                for p in stage.parameters():
                    p.requires_grad_(True)
        else:
            children = list(bb.children())
            for child in children[-n_stages:]:
                for p in child.parameters():
                    p.requires_grad_(True)
    for attr in ("conv_head", "bn2"):
        if hasattr(bb, attr):
            for p in getattr(bb, attr).parameters():
                p.requires_grad_(True)
    if hasattr(bb, "head"):
        for p in bb.head.parameters():
            p.requires_grad_(True)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class ExpConfig:
    id: str
    desc: str = ""
    backbone: str = "efficientnet_b0"
    gamma: float = 2.0
    alpha: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.5])
    smoothing: float = 0.05
    use_attention: bool = False
    attention_type: str = "cbam"
    use_arcface: bool = False
    ema: bool = True
    ema_decay: float = 0.999
    unfreeze: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs_head: int = 6
    epochs_finetune: int = 12
    batch_size: int = 16
    sampler_beta: float | None = None  # None=inverse-freq; else class-balanced
    heavy_aug: bool = False
    use_dwt: bool = True
    seed: int = 42
    # Serialized name retained so historical last.pth checkpoints remain resumable.
    focus_weight: float = 1.0
    # Remix (Chou et al., ECCV 2020 W) on Mixup λ (Zhang et al., ICLR 2018). 0 disables.
    remix_alpha: float = 0.0
    remix_kappa: float = 3.0
    remix_tau: float = 0.5
    init_checkpoint: str | None = None
    boost_paths: list[str] = field(default_factory=list)
    boost_weight: float = 1.0
    distill_alpha: float = 0.0
    distill_temperature: float = 2.0

    @property
    def canary_weight(self) -> float:
        """Canonical runtime name for the deprecated serialized focus_weight field."""
        return self.focus_weight


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    probs_all, labels_all = [], []
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
    acc = float((preds == labels).mean())
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2], zero_division=0)
    metrics = {
        "accuracy": acc,
        "macro_f1": float(f1.mean()),
        "precision_per_class": p.tolist(),
        "recall_per_class": r.tolist(),
        "f1_per_class": f1.tolist(),
        "sp_precision": float(p[2]),
        "sp_recall": float(r[2]),
        "sp_f1": float(f1[2]),
        "best_metric": float(0.5 * f1[2] + 0.3 * acc + 0.2 * f1.mean()),
    }
    return metrics, probs, labels


def sp_threshold_search(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Grid-search screen_photo threshold; if sp prob>=t force class 2, else argmax."""
    best = {"threshold": 0.5, "sp_f1": -1, "accuracy": 0}
    for t in np.arange(0.30, 0.801, 0.025):
        preds = probs.argmax(1)
        force = probs[:, 2] >= t
        preds = np.where(force, 2, preds)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2], zero_division=0)
        acc = float((preds == labels).mean())
        if f1[2] > best["sp_f1"]:
            best = {
                "threshold": round(float(t), 3),
                "sp_f1": float(f1[2]),
                "sp_precision": float(p[2]),
                "sp_recall": float(r[2]),
                "accuracy": acc,
                "macro_f1": float(f1.mean()),
            }
    return best


def checkpoint_selection_key(metrics: dict) -> float:
    """Select checkpoints only by held-out validation quality.

    Canary outcomes are deliberately absent: they block promotion after a
    checkpoint is selected but are not statistical evidence for epoch ranking.
    """
    return float(metrics["best_metric"])


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def train_one(
    cfg: ExpConfig,
    split: dict,
    device: str = "cuda",
    evaluate_test: bool = True,
    resume: bool = False,
    max_total_epochs: int | None = None,
) -> dict:
    seed_everything(cfg.seed)

    total_epochs = cfg.epochs_head + cfg.epochs_finetune
    if max_total_epochs is not None and max_total_epochs not in range(1, total_epochs + 1):
        raise ValueError(f"max_total_epochs must be between 1 and {total_epochs}")

    out_dir = EXP_DIR / cfg.id
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "last.pth"
    resume_state = None
    if resume:
        if not last_path.exists():
            raise RuntimeError(f"Resume checkpoint does not exist: {last_path}")
        resume_state = torch.load(last_path, map_location="cpu", weights_only=False)
        saved_cfg = {key: value for key, value in resume_state["cfg"].items() if key != "desc"}
        current_cfg = {key: value for key, value in asdict(cfg).items() if key != "desc"}
        if saved_cfg != current_cfg:
            raise RuntimeError("Resume configuration differs from the saved training configuration")

    train_s = [tuple(x) for x in split["train"]]
    val_s = [tuple(x) for x in split["val"]]
    test_s = [tuple(x) for x in split["test"]]

    train_ds = CachedDataset(train_s, train_tf(cfg.heavy_aug, seed=cfg.seed))
    val_ds = CachedDataset(val_s, eval_tf())
    test_ds = CachedDataset(test_s, eval_tf()) if evaluate_test else None

    canary_paths = load_canary_paths()
    canary_s = [sample for sample in train_s if str(Path(sample[0]).resolve()) in canary_paths]
    canary_ds = CachedDataset(canary_s, eval_tf()) if canary_s else None
    sampler = make_sampler(
        train_s,
        cfg.sampler_beta,
        canary_paths=canary_paths,
        canary_weight=cfg.canary_weight,
        seed=cfg.seed,
        boost_paths=set(cfg.boost_paths),
        boost_weight=cfg.boost_weight,
    )
    nw = 0  # Windows shared-memory mapping fails with workers>0; FFT/DWT are cached so CPU is light
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=nw,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = (
        DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=nw, pin_memory=True)
        if test_ds is not None
        else None
    )
    canary_loader = (
        DataLoader(canary_ds, batch_size=32, shuffle=False, num_workers=nw, pin_memory=True)
        if canary_ds is not None
        else None
    )
    if canary_s:
        print(
            f"  Canary examples: {len(canary_s)} at {cfg.canary_weight:.1f}x sampler weight",
            flush=True,
        )

    model = create_model(
        model_name=cfg.backbone,
        num_classes=3,
        pretrained=resume_state is None and not cfg.init_checkpoint,
        freeze_backbone=True,
        use_dwt=cfg.use_dwt,
        use_arcface=cfg.use_arcface,
        use_fft_attention=cfg.use_attention,
        attention_type=cfg.attention_type,
    ).to(device)
    if cfg.init_checkpoint and resume_state is None:
        load_init_checkpoint(model, cfg.init_checkpoint)
        print(f"  Loaded init checkpoint {cfg.init_checkpoint}", flush=True)

    criterion = FocalLossLS(cfg.gamma, cfg.alpha, cfg.smoothing).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema else None
    teacher = None
    if cfg.distill_alpha > 0:
        if not cfg.init_checkpoint:
            raise RuntimeError("distill_alpha > 0 requires init_checkpoint")
        teacher = copy.deepcopy(model).eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
    train_counts = torch.tensor(
        np.bincount([label for _path, label in train_s], minlength=3),
        device=device,
        dtype=torch.float32,
    )

    history: list[dict] = []
    best = {
        "best_metric": -1,
        "canary_pass": False,
        "canary_correct": 0,
        "canary_total": len(canary_s),
    }
    best_key = -1.0
    best_state = None
    completed_head = 0
    completed_finetune = 0

    if resume_state is not None:
        model.load_state_dict(resume_state["model_state_dict"])
        if ema is not None:
            if resume_state["ema_state_dict"] is None:
                raise RuntimeError("Resume checkpoint is missing EMA state")
            ema.ema.load_state_dict(resume_state["ema_state_dict"])
        scaler.load_state_dict(resume_state["scaler_state_dict"])
        history = resume_state["history"]
        best = resume_state["selection"]
        if "canary_pass" not in best and "focus_pass" in best:
            best = {
                **best,
                "canary_pass": best["focus_pass"],
                "canary_correct": best["focus_correct"],
                "canary_total": best["focus_total"],
                "canary_probabilities": best.get("focus_probabilities", []),
            }
        saved_best_key = resume_state["best_key"]
        best_key = float(saved_best_key[-1] if isinstance(saved_best_key, (list, tuple)) else saved_best_key)
        best_state = resume_state["best_state_dict"]
        completed_head = resume_state["completed_head"]
        completed_finetune = resume_state["completed_finetune"]
        _restore_runtime_rng(resume_state["runtime_rng"], train_ds.transform, sampler)
        print(
            f"  Resuming after A={completed_head}/{cfg.epochs_head}, B={completed_finetune}/{cfg.epochs_finetune}",
            flush=True,
        )

    def run_epoch(optimizer):
        model.train()
        for rgb, fft, dwt, labels in train_loader:
            rgb, fft, dwt, labels = (
                rgb.to(device),
                fft.to(device),
                dwt.to(device),
                labels.to(device),
            )
            mix_index = None
            lam_y = None
            if cfg.remix_alpha > 0 and not cfg.use_arcface and rgb.size(0) > 1:
                mix_index = torch.randperm(rgb.size(0), device=rgb.device)
                lam = (
                    torch.distributions.Beta(cfg.remix_alpha, cfg.remix_alpha)
                    .sample((rgb.size(0),))
                    .to(
                        device=rgb.device,
                        dtype=rgb.dtype,
                    )
                )
                rgb, fft, dwt = mix_modal_batch(rgb, fft, dwt, mix_index, lam)
                lam_y = remix_label_lambda(
                    labels,
                    labels[mix_index],
                    lam,
                    train_counts,
                    kappa=cfg.remix_kappa,
                    tau=cfg.remix_tau,
                )
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(rgb, fft, dwt, labels) if cfg.use_arcface else model(rgb, fft, dwt)
                logits = out[0] if isinstance(out, tuple) else out
                if lam_y is not None and mix_index is not None:
                    loss = mixed_focal_loss(criterion, logits, labels, labels[mix_index], lam_y)
                else:
                    loss = criterion(logits, labels)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_out = teacher(rgb, fft, dwt)
                        teacher_logits = teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out
                    loss = loss + cfg.distill_alpha * distillation_kl(
                        logits,
                        teacher_logits,
                        temperature=cfg.distill_temperature,
                    )
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema:
                ema.update(model)

    def maybe_save(tag_model, stage: str, epoch: int):
        nonlocal best, best_key, best_state
        m, _, _ = evaluate(tag_model, val_loader, device)
        canary_correct = 0
        canary_probabilities: list[list[float]] = []
        if canary_loader is not None:
            _, canary_probs, canary_labels = evaluate(tag_model, canary_loader, device)
            canary_correct = int((canary_probs.argmax(1) == canary_labels).sum())
            canary_probabilities = canary_probs.tolist()
        canary_total = len(canary_s)
        canary_pass = canary_total > 0 and canary_correct == canary_total
        candidate_key = checkpoint_selection_key(m)
        row = {
            "stage": stage,
            "epoch": epoch,
            **m,
            "canary_correct": canary_correct,
            "canary_total": canary_total,
            "canary_pass": canary_pass,
        }
        history.append(row)
        if candidate_key > best_key:
            best_key = candidate_key
            best = {**row, "canary_probabilities": canary_probabilities}
            best_state = copy.deepcopy(tag_model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "model_name": cfg.backbone,
                    "num_classes": 3,
                    "use_dwt": cfg.use_dwt,
                    "use_arcface": cfg.use_arcface,
                    "cfg": asdict(cfg),
                    "selection": best,
                    "selection_policy": "validation_metric_only_then_canary_gate",
                },
                out_dir / "best.pth",
            )
        (out_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return m

    def save_last(stage: str, optimizer, scheduler) -> None:
        payload = {
            "schema_version": 2,
            "selection_policy": "validation_metric_only_then_canary_gate",
            "cfg": asdict(cfg),
            "stage": stage,
            "completed_head": completed_head,
            "completed_finetune": completed_finetune,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.ema.state_dict() if ema is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "runtime_rng": _capture_runtime_rng(train_ds.transform, sampler),
            "history": history,
            "selection": best,
            "best_key": best_key,
            "best_state_dict": best_state,
        }
        temporary = out_dir / "last.tmp.pth"
        torch.save(payload, temporary)
        temporary.replace(last_path)

    def paused_result() -> dict:
        return {
            "status": "paused",
            "progress": {
                "completed_head": completed_head,
                "completed_finetune": completed_finetune,
                "total_epochs": total_epochs,
            },
            "val": None,
            "test": None,
            "test_threshold": None,
            "selection": best,
            "history": history,
        }

    def should_pause() -> bool:
        completed = completed_head + completed_finetune
        return max_total_epochs is not None and completed >= max_total_epochs and completed < total_epochs

    # Stage A: head + freq_branch + norms
    from torch.optim.lr_scheduler import CosineAnnealingLR

    if cfg.epochs_head > 0:
        params_a = (
            list(model.classifier.parameters())
            + list(model.freq_branch.parameters())
            + list(model.spatial_norm.parameters())
            + list(model.freq_norm.parameters())
        )
        opt_a = torch.optim.AdamW(params_a, lr=cfg.lr, weight_decay=cfg.weight_decay)
        sch_a = CosineAnnealingLR(opt_a, T_max=cfg.epochs_head)
        if resume_state is not None and resume_state["stage"] == "head" and completed_head < cfg.epochs_head:
            opt_a.load_state_dict(resume_state["optimizer_state_dict"])
            sch_a.load_state_dict(resume_state["scheduler_state_dict"])
        for ep in range(completed_head, cfg.epochs_head):
            t0 = time.time()
            run_epoch(opt_a)
            sch_a.step()
            m = maybe_save(ema.ema if ema else model, "head", ep + 1)
            completed_head = ep + 1
            save_last("head", opt_a, sch_a)
            canary_status = f" canary={history[-1]['canary_correct']}/{history[-1]['canary_total']}" if canary_s else ""
            print(
                f"  [A {ep + 1}/{cfg.epochs_head}] acc={m['accuracy']:.4f} spF1={m['sp_f1']:.4f} "
                f"macroF1={m['macro_f1']:.4f} metric={m['best_metric']:.4f}{canary_status} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
            if should_pause():
                print(f"  Paused safely after {completed_head + completed_finetune}/{total_epochs} epochs", flush=True)
                return paused_result()
    elif resume_state is None:
        m = maybe_save(ema.ema if ema else model, "init", 0)
        canary_status = f" canary={history[-1]['canary_correct']}/{history[-1]['canary_total']}" if canary_s else ""
        print(
            f"  [init] acc={m['accuracy']:.4f} spF1={m['sp_f1']:.4f} "
            f"macroF1={m['macro_f1']:.4f} metric={m['best_metric']:.4f}{canary_status}",
            flush=True,
        )

    # Stage B: unfreeze
    unfreeze_stages(model, cfg.unfreeze)
    params_b = [
        {
            "params": [p for p in model.backbone.parameters() if p.requires_grad],
            "lr": cfg.lr * 0.1,
        },
        {"params": model.freq_branch.parameters(), "lr": cfg.lr * 0.1},
        {"params": model.classifier.parameters(), "lr": cfg.lr},
        {"params": model.spatial_norm.parameters(), "lr": cfg.lr},
        {"params": model.freq_norm.parameters(), "lr": cfg.lr},
    ]
    opt_b = torch.optim.AdamW(params_b, weight_decay=cfg.weight_decay)
    sch_b = CosineAnnealingLR(opt_b, T_max=max(cfg.epochs_finetune, 1))
    if resume_state is not None and resume_state["stage"] == "finetune":
        opt_b.load_state_dict(resume_state["optimizer_state_dict"])
        sch_b.load_state_dict(resume_state["scheduler_state_dict"])
    for ep in range(completed_finetune, cfg.epochs_finetune):
        t0 = time.time()
        run_epoch(opt_b)
        sch_b.step()
        m = maybe_save(ema.ema if ema else model, "finetune", ep + 1)
        completed_finetune = ep + 1
        save_last("finetune", opt_b, sch_b)
        canary_status = f" canary={history[-1]['canary_correct']}/{history[-1]['canary_total']}" if canary_s else ""
        print(
            f"  [B {ep + 1}/{cfg.epochs_finetune}] acc={m['accuracy']:.4f} spF1={m['sp_f1']:.4f} "
            f"macroF1={m['macro_f1']:.4f} metric={m['best_metric']:.4f}{canary_status} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        if should_pause():
            print(f"  Paused safely after {completed_head + completed_finetune}/{total_epochs} epochs", flush=True)
            return paused_result()

    # Final eval on TEST with best checkpoint
    eval_model = create_model(
        model_name=cfg.backbone,
        num_classes=3,
        pretrained=False,
        use_dwt=cfg.use_dwt,
        use_arcface=cfg.use_arcface,
        use_fft_attention=cfg.use_attention,
        attention_type=cfg.attention_type,
    ).to(device)
    eval_model.load_state_dict(best_state)
    val_m, _, _ = evaluate(eval_model, val_loader, device)
    test_m = None
    thr = None
    if test_loader is not None:
        test_m, test_probs, test_labels = evaluate(eval_model, test_loader, device)
        thr = sp_threshold_search(test_probs, test_labels)

    # Save best checkpoint with final selection metadata.
    torch.save(
        {
            "model_state_dict": best_state,
            "model_name": cfg.backbone,
            "num_classes": 3,
            "use_dwt": cfg.use_dwt,
            "use_arcface": cfg.use_arcface,
            "cfg": asdict(cfg),
            "selection": best,
        },
        out_dir / "best.pth",
    )

    return {
        "status": "complete",
        "val": val_m,
        "test": test_m,
        "test_threshold": thr,
        "selection": best,
        "history": history,
    }


def evaluate_checkpoint_on_test(checkpoint_path: Path, split: dict, device: str = "cuda") -> dict:
    """Evaluate one validation-selected checkpoint on the frozen test set."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("cfg", {})
    model = create_model(
        model_name=checkpoint.get("model_name", cfg.get("backbone", "efficientnet_b0")),
        num_classes=checkpoint.get("num_classes", 3),
        pretrained=False,
        use_dwt=checkpoint.get("use_dwt", cfg.get("use_dwt", True)),
        use_arcface=checkpoint.get("use_arcface", cfg.get("use_arcface", False)),
        use_fft_attention=cfg.get("use_attention", False),
        attention_type=cfg.get("attention_type", "cbam"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    samples = [tuple(sample) for sample in split["test"]]
    loader = DataLoader(CachedDataset(samples, eval_tf()), batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    metrics, probabilities, labels = evaluate(model, loader, device)
    return {"metrics": metrics, "threshold_diagnostic": sp_threshold_search(probabilities, labels)}


def append_leaderboard(cfg: ExpConfig, result: dict, elapsed: float) -> None:
    row = {
        "id": cfg.id,
        "desc": cfg.desc,
        "elapsed_s": round(elapsed, 1),
        "cfg": asdict(cfg),
        "val_acc": result["val"]["accuracy"],
        "val_sp_f1": result["val"]["sp_f1"],
        "val_macro_f1": result["val"]["macro_f1"],
        "val_metric": result["val"]["best_metric"],
        "test_acc": result["test"]["accuracy"],
        "test_sp_f1": result["test"]["sp_f1"],
        "test_sp_precision": result["test"]["sp_precision"],
        "test_sp_recall": result["test"]["sp_recall"],
        "test_macro_f1": result["test"]["macro_f1"],
        "test_metric": result["test"]["best_metric"],
        "test_thr": result["test_threshold"],
    }
    with LEADERBOARD.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_configs(configs: list[ExpConfig]) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = build_split()
    print(
        f"Split: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}",
        flush=True,
    )
    all_samples = split["train"] + split["val"] + split["test"]
    precompute_cache([tuple(x) for x in all_samples])
    preload_ram([tuple(x) for x in all_samples])
    for cfg in configs:
        print(f"\n{'=' * 70}\n### {cfg.id}: {cfg.desc}\n{'=' * 70}", flush=True)
        t0 = time.time()
        try:
            result = train_one(cfg, split, device)
            elapsed = time.time() - t0
            append_leaderboard(cfg, result, elapsed)
            print(
                f">>> {cfg.id} DONE test_acc={result['test']['accuracy']:.4f} "
                f"test_spF1={result['test']['sp_f1']:.4f} test_macroF1={result['test']['macro_f1']:.4f} "
                f"thr_spF1={result['test_threshold']['sp_f1']:.4f} ({elapsed:.0f}s)",
                flush=True,
            )
        except Exception as e:
            import traceback

            print(f"!!! {cfg.id} FAILED: {e}", flush=True)
            traceback.print_exc()


def screening_queue() -> list[ExpConfig]:
    """Curated experiments, highest-value first (unattended run)."""
    H, F = 6, 12  # noqa: N806  # concise names used throughout the experiment matrix
    return [
        # Candidate clean default (audit-recommended)
        ExpConfig(
            id="ref",
            desc="clean default: g2 a[1,1,1.5] ls0.05 ema unf2",
            epochs_head=H,
            epochs_finetune=F,
        ),
        # Reproduce past strategy on clean split (focal-only, g3, heavy aug, unfreeze-all, no ema)
        ExpConfig(
            id="old_focal",
            desc="past strategy: g3 ls0 heavy-aug unfreeze-all no-ema",
            gamma=3.0,
            smoothing=0.0,
            ema=False,
            unfreeze=7,
            heavy_aug=True,
            epochs_head=H,
            epochs_finetune=F,
        ),
        # Loss / class-weight ablations from ref
        ExpConfig(
            id="a_alpha20",
            desc="alpha[1,1,2.0]",
            alpha=[1.0, 1.0, 2.0],
            epochs_head=H,
            epochs_finetune=F,
        ),
        ExpConfig(
            id="a_alpha25",
            desc="alpha[1,1,2.5]",
            alpha=[1.0, 1.0, 2.5],
            epochs_head=H,
            epochs_finetune=F,
        ),
        ExpConfig(id="a_gamma3", desc="gamma=3", gamma=3.0, epochs_head=H, epochs_finetune=F),
        ExpConfig(
            id="a_ls0",
            desc="no label smoothing",
            smoothing=0.0,
            epochs_head=H,
            epochs_finetune=F,
        ),
        ExpConfig(
            id="a_ls10",
            desc="label smoothing 0.10",
            smoothing=0.10,
            epochs_head=H,
            epochs_finetune=F,
        ),
        # Fine-tune depth
        ExpConfig(
            id="a_unf1",
            desc="unfreeze 1 stage",
            unfreeze=1,
            epochs_head=H,
            epochs_finetune=F,
        ),
        ExpConfig(
            id="a_unf3",
            desc="unfreeze 3 stages",
            unfreeze=3,
            epochs_head=H,
            epochs_finetune=F,
        ),
        # EMA ablation
        ExpConfig(id="a_noema", desc="no EMA", ema=False, epochs_head=H, epochs_finetune=F),
        # Architecture
        ExpConfig(
            id="a_attn",
            desc="+CBAM freq attention",
            use_attention=True,
            epochs_head=H,
            epochs_finetune=F,
        ),
        ExpConfig(
            id="a_fftonly",
            desc="FFT only (no DWT)",
            use_dwt=False,
            epochs_head=H,
            epochs_finetune=F,
        ),
        ExpConfig(
            id="a_b1",
            desc="backbone b1, unfreeze1",
            backbone="efficientnet_b1",
            unfreeze=1,
            epochs_head=H,
            epochs_finetune=F,
        ),
        # Sampler
        ExpConfig(
            id="a_cb99",
            desc="class-balanced sampler b=0.99",
            sampler_beta=0.99,
            epochs_head=H,
            epochs_finetune=F,
        ),
        # Aug
        ExpConfig(
            id="a_heavyaug",
            desc="heavy augmentation",
            heavy_aug=True,
            epochs_head=H,
            epochs_finetune=F,
        ),
    ]


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if arg == "smoke":
        run_configs([ExpConfig(id="smoke", desc="smoke test", epochs_head=1, epochs_finetune=1)])
    elif arg == "screen":
        run_configs(screening_queue())
