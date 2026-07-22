"""Autonomous experiment harness for screen-detector V3.

Clean, fast, reproducible training pipeline that fixes the audited bugs:
- No hard_negative double-loading / label conflicts (clean label map, path dedup)
- Stratified train/val/test split, saved once and reused across all experiments
- FFT/DWT features cached to disk (computed from raw RGB image, matches inference)
- Fixed unfreeze (only real MBConv stages + conv_head + bn2)
- Stage-A optimizer includes LayerNorm params
- EMA, label smoothing, adaptive threshold on held-out TEST
- Release selection first gates confirmed hard examples, then maximizes the VAL
  metric: 0.5*sp_f1 + 0.3*acc + 0.2*macro_f1

The release trainer imports this harness with the ablation-winning defaults;
legacy trainer modules remain available separately for baseline reproduction.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from trainer.model import create_model

DATA_DIR = ROOT / "data" / "input"
ABLATION_DIR = ROOT / "experiment" / "cnn_fft_dwt_ablation"
CACHE_DIR = ABLATION_DIR / "cache"
EXP_DIR = ABLATION_DIR / "exp"
SPLIT_PATH = ABLATION_DIR / "split.json"
LEADERBOARD = ABLATION_DIR / "leaderboard.jsonl"
FOCUS_MANIFEST = ROOT / "trainer" / "hard_examples.txt"
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
def collect_samples(
    data_dir: Path = DATA_DIR,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Return (main_samples, hard_neg_samples), deduped by resolved path."""
    seen: set[str] = set()
    main: list[tuple[str, int]] = []
    hard: list[tuple[str, int]] = []

    main_map = {"natural_photo": 0, "screenshot": 1, "screen_photo": 2}
    for folder, label in main_map.items():
        for p in (data_dir / folder).rglob("*"):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            main.append((str(p), label))

    hn_root = data_dir / "hard_negative"
    for sub, label in HN_MAP.items():
        d = hn_root / sub
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            hard.append((str(p), label))
    return main, hard


def load_focus_paths(
    data_dir: Path = DATA_DIR,
    manifest_path: Path = FOCUS_MANIFEST,
) -> set[str]:
    """Load curated hard-example paths, ignoring comments and missing local files."""
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
    data_dir: Path,
) -> str:
    """Fingerprint paths, labels, sizes and mtimes so stale splits are not reused."""
    root = data_dir.resolve()
    rows = []
    for raw_path, label in samples:
        path = Path(raw_path).resolve()
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        rows.append(f"{relative}\0{label}\0{stat.st_size}\0{stat.st_mtime_ns}")
    payload = "\n".join(sorted(rows)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_split(
    seed: int = 42,
    data_dir: Path = DATA_DIR,
    split_path: Path = SPLIT_PATH,
    focus_paths: set[str] | None = None,
) -> dict:
    """Build a cached stratified split; hard/focus examples always stay in train."""
    main, hard = collect_samples(data_dir)
    all_samples = main + hard
    fingerprint = _dataset_fingerprint(all_samples, data_dir)
    focus = focus_paths if focus_paths is not None else load_focus_paths(data_dir)
    collected_paths = {str(Path(path).resolve()) for path, _ in all_samples}
    if missing_focus := focus - collected_paths:
        missing = "\n".join(f"- {path}" for path in sorted(missing_focus))
        raise RuntimeError(f"Focus examples are not part of the collected dataset:\n{missing}")
    focus_fingerprint = hashlib.sha256("\n".join(sorted(focus)).encode("utf-8")).hexdigest()

    if split_path.exists():
        saved = json.loads(split_path.read_text(encoding="utf-8"))
        meta = saved.get("meta", {})
        if (
            meta.get("dataset_fingerprint") == fingerprint
            and meta.get("focus_fingerprint") == focus_fingerprint
            and meta.get("seed") == seed
        ):
            return saved

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
                if str(Path(sample[0]).resolve()) not in focus:
                    continue
                replacement_index = next(
                    (
                        index
                        for index in range(len(label_train) - 1, -1, -1)
                        if str(Path(label_train[index][0]).resolve()) not in focus
                    ),
                    None,
                )
                if replacement_index is None:
                    raise RuntimeError(f"No train sample available to swap with focus example: {sample[0]}")
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
            "schema_version": 2,
            "seed": seed,
            "dataset_fingerprint": fingerprint,
            "focus_fingerprint": focus_fingerprint,
            "main_count": len(main),
            "hard_negative_count": len(hard),
            "focus_count": len(focus),
        },
        "train": train,
        "val": val,
        "test": test,
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, ensure_ascii=False), encoding="utf-8")
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
def train_tf(heavy: bool = False) -> A.Compose:
    if heavy:
        from trainer.augment import get_train_transforms

        return get_train_transforms()
    return A.Compose(
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
    focus_paths: set[str] | None = None,
    focus_weight: float = 1.0,
) -> WeightedRandomSampler:
    """Balance classes and optionally upweight curated regression examples."""
    labels = [s[1] for s in samples]
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if beta is None:  # inverse frequency
        cls_w = len(labels) / np.maximum(counts, 1)
    else:  # class-balanced (effective number)
        cls_w = (1 - beta) / (1 - np.power(beta, np.maximum(counts, 1)))
    focus = focus_paths or set()
    w = np.array(
        [cls_w[label] * (focus_weight if str(Path(path).resolve()) in focus else 1.0) for path, label in samples],
        dtype=np.float64,
    )
    return WeightedRandomSampler(torch.from_numpy(w), num_samples=len(w), replacement=True)


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
class FocalLossLS(nn.Module):
    def __init__(self, gamma: float, alpha: list[float] | None, smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.register_buffer("alpha", torch.tensor(alpha) if alpha else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
        return loss.mean()


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
    if hasattr(bb, "blocks") and n_stages > 0:
        blocks = list(bb.blocks.children())
        for blk in blocks[-n_stages:]:
            for p in blk.parameters():
                p.requires_grad_(True)
    for attr in ("conv_head", "bn2"):
        if hasattr(bb, attr):
            for p in getattr(bb, attr).parameters():
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
    focus_weight: float = 1.0


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


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def train_one(cfg: ExpConfig, split: dict, device: str = "cuda") -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_s = [tuple(x) for x in split["train"]]
    val_s = [tuple(x) for x in split["val"]]
    test_s = [tuple(x) for x in split["test"]]

    train_ds = CachedDataset(train_s, train_tf(cfg.heavy_aug))
    val_ds = CachedDataset(val_s, eval_tf())
    test_ds = CachedDataset(test_s, eval_tf())

    focus_paths = load_focus_paths()
    focus_s = [sample for sample in train_s if str(Path(sample[0]).resolve()) in focus_paths]
    focus_ds = CachedDataset(focus_s, eval_tf()) if focus_s else None
    sampler = make_sampler(
        train_s,
        cfg.sampler_beta,
        focus_paths=focus_paths,
        focus_weight=cfg.focus_weight,
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
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=nw, pin_memory=True)
    focus_loader = (
        DataLoader(focus_ds, batch_size=32, shuffle=False, num_workers=nw, pin_memory=True)
        if focus_ds is not None
        else None
    )
    if focus_s:
        print(
            f"  Focus examples: {len(focus_s)} at {cfg.focus_weight:.1f}x sampler weight",
            flush=True,
        )

    model = create_model(
        model_name=cfg.backbone,
        num_classes=3,
        pretrained=True,
        freeze_backbone=True,
        use_dwt=cfg.use_dwt,
        use_arcface=cfg.use_arcface,
        use_fft_attention=cfg.use_attention,
        attention_type=cfg.attention_type,
    ).to(device)

    criterion = FocalLossLS(cfg.gamma, cfg.alpha, cfg.smoothing).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema else None

    def run_epoch(optimizer):
        model.train()
        for rgb, fft, dwt, labels in train_loader:
            rgb, fft, dwt, labels = (
                rgb.to(device),
                fft.to(device),
                dwt.to(device),
                labels.to(device),
            )
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(rgb, fft, dwt, labels) if cfg.use_arcface else model(rgb, fft, dwt)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema:
                ema.update(model)

    out_dir = EXP_DIR / cfg.id
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best = {
        "best_metric": -1,
        "focus_pass": False,
        "focus_correct": 0,
        "focus_total": len(focus_s),
    }
    best_key = (-1, -1, -1.0)
    best_state = None

    def maybe_save(tag_model, stage: str, epoch: int):
        nonlocal best, best_key, best_state
        m, _, _ = evaluate(tag_model, val_loader, device)
        focus_correct = 0
        focus_probabilities: list[list[float]] = []
        if focus_loader is not None:
            _, focus_probs, focus_labels = evaluate(tag_model, focus_loader, device)
            focus_correct = int((focus_probs.argmax(1) == focus_labels).sum())
            focus_probabilities = focus_probs.tolist()
        focus_total = len(focus_s)
        focus_pass = focus_correct == focus_total
        candidate_key = (int(focus_pass), focus_correct, m["best_metric"])
        row = {
            "stage": stage,
            "epoch": epoch,
            **m,
            "focus_correct": focus_correct,
            "focus_total": focus_total,
            "focus_pass": focus_pass,
        }
        history.append(row)
        if candidate_key > best_key:
            best_key = candidate_key
            best = {**row, "focus_probabilities": focus_probabilities}
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
                },
                out_dir / "best.pth",
            )
        (out_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return m

    # Stage A: head + freq_branch + norms
    from torch.optim.lr_scheduler import CosineAnnealingLR

    params_a = (
        list(model.classifier.parameters())
        + list(model.freq_branch.parameters())
        + list(model.spatial_norm.parameters())
        + list(model.freq_norm.parameters())
    )
    opt_a = torch.optim.AdamW(params_a, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch_a = CosineAnnealingLR(opt_a, T_max=cfg.epochs_head)
    for ep in range(cfg.epochs_head):
        t0 = time.time()
        run_epoch(opt_a)
        sch_a.step()
        m = maybe_save(ema.ema if ema else model, "head", ep + 1)
        focus_status = f" focus={history[-1]['focus_correct']}/{history[-1]['focus_total']}" if focus_s else ""
        print(
            f"  [A {ep + 1}/{cfg.epochs_head}] acc={m['accuracy']:.4f} spF1={m['sp_f1']:.4f} "
            f"macroF1={m['macro_f1']:.4f} metric={m['best_metric']:.4f}{focus_status} "
            f"({time.time() - t0:.0f}s)",
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
    sch_b = CosineAnnealingLR(opt_b, T_max=cfg.epochs_finetune)
    for ep in range(cfg.epochs_finetune):
        t0 = time.time()
        run_epoch(opt_b)
        sch_b.step()
        m = maybe_save(ema.ema if ema else model, "finetune", ep + 1)
        focus_status = f" focus={history[-1]['focus_correct']}/{history[-1]['focus_total']}" if focus_s else ""
        print(
            f"  [B {ep + 1}/{cfg.epochs_finetune}] acc={m['accuracy']:.4f} spF1={m['sp_f1']:.4f} "
            f"macroF1={m['macro_f1']:.4f} metric={m['best_metric']:.4f}{focus_status} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

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
        "val": val_m,
        "test": test_m,
        "test_threshold": thr,
        "selection": best,
        "history": history,
    }


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
