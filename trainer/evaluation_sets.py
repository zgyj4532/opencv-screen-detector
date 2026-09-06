"""Evaluation-set contracts, isolation audits, and deployment metrics.

This module is the single seam for the release evaluation policy.  It keeps
known regressions (canaries), promotion statistics (frozen challenge), online
errors (rolling pool), and true out-of-distribution samples separate.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "trainer" / "evaluation_sets"
CANARY_MANIFEST = MANIFEST_DIR / "canary.json"
FROZEN_CHALLENGE_MANIFEST = MANIFEST_DIR / "frozen_challenge.json"
ROLLING_ERROR_POOL_MANIFEST = MANIFEST_DIR / "rolling_error_pool.json"
OOD_MANIFEST = MANIFEST_DIR / "ood.json"
GROUP_METADATA_MANIFEST = MANIFEST_DIR / "group_metadata.json"

CLASS_NAMES = ("natural", "screenshot", "screen_photo")
GROUP_KEYS = (
    "source_content_id",
    "capture_session_id",
    "camera_device_id",
    "display_device_id",
    "transformation_lineage_id",
)
EVALUATION_SET_NAMES = ("canary", "frozen_challenge", "rolling_error_pool", "ood")
READY = "READY"
NOT_READY = "NOT_READY"
FAIL = "FAIL"
REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True)
class EvaluationSample:
    """A validated manifest entry resolved to a local file."""

    sample_id: str
    path: Path
    expected_label: str
    category: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Evaluation manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported evaluation manifest schema: {path}")
    return payload


def _manifest_path(set_name: str, manifest_dir: Path) -> Path:
    if set_name not in EVALUATION_SET_NAMES:
        raise ValueError(f"Unknown evaluation set: {set_name}")
    return manifest_dir / f"{set_name}.json"


def load_evaluation_manifest(set_name: str, manifest_dir: Path = MANIFEST_DIR) -> dict[str, Any]:
    """Load and validate one evaluation-set manifest."""
    path = _manifest_path(set_name, manifest_dir)
    payload = _read_json(path)
    if payload.get("set_type") != set_name:
        raise RuntimeError(f"Manifest {path} declares set_type={payload.get('set_type')!r}, expected {set_name!r}")
    if not isinstance(payload.get("entries"), list):
        raise TypeError(f"Manifest entries must be a list: {path}")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for entry in payload["entries"]:
        if not isinstance(entry, dict):
            raise TypeError(f"Manifest entry must be an object: {path}")
        sample_id = str(entry.get("id", "")).strip()
        sample_path = str(entry.get("path", "")).strip()
        if not sample_id or not sample_path:
            raise RuntimeError(f"Manifest entries require non-empty id and path: {path}")
        if Path(sample_path).is_absolute() or ".." in Path(sample_path).parts:
            raise RuntimeError(f"Manifest paths must be repository-relative: {sample_path}")
        if sample_id in seen_ids or sample_path in seen_paths:
            raise RuntimeError(f"Duplicate id or path in {path}: {sample_id} / {sample_path}")
        seen_ids.add(sample_id)
        seen_paths.add(sample_path)

        digest = str(entry.get("sha256", "")).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RuntimeError(f"Manifest entries require a valid SHA-256: {sample_id}")

        expected_label = entry.get("expected_label")
        if set_name == "ood":
            if expected_label != "ood" or not str(entry.get("category", "")).strip():
                raise RuntimeError(f"OOD entries require expected_label='ood' and a category: {sample_id}")
        elif expected_label not in CLASS_NAMES:
            raise RuntimeError(f"Unknown expected_label for {sample_id}: {expected_label!r}")

        if payload.get("readiness", {}).get("requires_complete_group_metadata"):
            group_ids = entry.get("group_ids", {})
            missing_group_keys = [key for key in GROUP_KEYS if not group_ids.get(key)]
            if missing_group_keys:
                raise RuntimeError(f"Challenge entry {sample_id} is missing group IDs: {', '.join(missing_group_keys)}")
    return payload


def resolve_manifest_samples(
    set_name: str,
    manifest_dir: Path = MANIFEST_DIR,
    repo_root: Path = ROOT,
    *,
    require_files: bool = True,
) -> list[EvaluationSample]:
    """Resolve manifest entries without exposing manifest policy to callers."""
    payload = load_evaluation_manifest(set_name, manifest_dir)
    samples: list[EvaluationSample] = []
    for entry in payload["entries"]:
        path = (repo_root / entry["path"]).resolve()
        try:
            path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Evaluation sample escapes repository root: {entry['path']}") from exc
        if require_files and not path.is_file():
            raise RuntimeError(f"Evaluation sample is missing: {entry['path']}")
        samples.append(
            EvaluationSample(
                sample_id=entry["id"],
                path=path,
                expected_label=entry["expected_label"],
                category=entry.get("category"),
            )
        )
    return samples


def load_canary_paths(
    manifest_path: Path = CANARY_MANIFEST,
    repo_root: Path = ROOT,
    *,
    require_files: bool = False,
) -> set[str]:
    """Return available canary paths; canaries may be training samples."""
    samples = resolve_manifest_samples(
        "canary",
        manifest_dir=manifest_path.parent,
        repo_root=repo_root,
        require_files=require_files,
    )
    return {str(sample.path) for sample in samples if sample.path.is_file()}


def evaluation_set_readiness(
    manifest_dir: Path = MANIFEST_DIR,
    repo_root: Path = ROOT,
) -> dict[str, dict[str, Any]]:
    """Evaluate declared governance and locally verifiable set readiness."""
    report: dict[str, dict[str, Any]] = {}
    for set_name in EVALUATION_SET_NAMES:
        payload = load_evaluation_manifest(set_name, manifest_dir)
        entries = payload["entries"]
        policy = payload.get("policy", {})
        readiness = payload.get("readiness", {})
        reasons: list[str] = []
        minimum = int(readiness.get("minimum_entries", 0))
        maximum = readiness.get("maximum_entries")
        if len(entries) < minimum:
            reasons.append(f"requires at least {minimum} entries; found {len(entries)}")
        if maximum is not None and len(entries) > int(maximum):
            reasons.append(f"allows at most {maximum} entries; found {len(entries)}")

        missing_paths = [entry["path"] for entry in entries if not (repo_root / entry["path"]).is_file()]
        if missing_paths:
            reasons.append(f"{len(missing_paths)} manifest files are missing")
        hash_mismatches = []
        for entry in entries:
            sample_path = repo_root / entry["path"]
            expected_hash = str(entry.get("sha256", "")).lower()
            if sample_path.is_file() and expected_hash and file_sha256(sample_path) != expected_hash:
                hash_mismatches.append(entry["path"])
        if hash_mismatches:
            reasons.append(f"{len(hash_mismatches)} files do not match their declared SHA-256")

        if payload.get("status") != READY:
            reasons.append(f"manifest governance status is {payload.get('status', NOT_READY)}")

        if readiness.get("requires_complete_group_metadata"):
            for key in GROUP_KEYS:
                values = [entry["group_ids"][key] for entry in entries]
                if len(values) != len(set(values)):
                    reasons.append(f"challenge entries are not independent by {key}")

        report[set_name] = {
            "status": READY if not reasons else NOT_READY,
            "declared_status": payload.get("status", NOT_READY),
            "entries": len(entries),
            "policy": policy,
            "reasons": reasons,
            "missing_paths": missing_paths,
            "hash_mismatches": hash_mismatches,
        }
    identities: dict[tuple[str, str], list[str]] = defaultdict(list)
    for set_name in EVALUATION_SET_NAMES:
        payload = load_evaluation_manifest(set_name, manifest_dir)
        for entry in payload["entries"]:
            identities[("sha256", entry["sha256"].lower())].append(set_name)
            identities[("path", entry["path"])].append(set_name)
    overlaps = {
        f"{kind}:{value}": sorted(set(owners)) for (kind, value), owners in identities.items() if len(set(owners)) > 1
    }
    for owners in overlaps.values():
        for set_name in owners:
            report[set_name]["status"] = NOT_READY
            report[set_name]["reasons"].append("entry identity overlaps another governed evaluation set")
    for set_name in EVALUATION_SET_NAMES:
        report[set_name]["cross_set_overlaps"] = {
            identity: owners for identity, owners in overlaps.items() if set_name in owners
        }
    return report


def load_group_metadata(path: Path = GROUP_METADATA_MANIFEST) -> dict[str, dict[str, str | None]]:
    """Load capture/content lineage metadata keyed by exact content hash."""
    payload = _read_json(path)
    if payload.get("set_type") != "group_metadata":
        raise RuntimeError(f"Manifest {path} is not group metadata")
    if tuple(payload.get("required_group_keys", [])) != GROUP_KEYS:
        raise RuntimeError(f"Group metadata must declare required keys in canonical order: {GROUP_KEYS}")

    resolved: dict[str, dict[str, str | None]] = {}
    for entry in payload.get("entries", []):
        digest = str(entry.get("sha256", "")).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RuntimeError(f"Invalid group-metadata SHA-256: {digest!r}")
        if digest in resolved:
            raise RuntimeError(f"Duplicate group metadata for SHA-256: {digest}")
        missing_keys = [key for key in GROUP_KEYS if key not in entry]
        if missing_keys:
            raise RuntimeError(f"Group metadata {digest} is missing keys: {', '.join(missing_keys)}")
        resolved[digest] = {
            key: str(entry[key]).strip() if entry[key] is not None and str(entry[key]).strip() else None
            for key in GROUP_KEYS
        }
    return resolved


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def perceptual_hash(path: str | Path, hash_size: int = 8, high_frequency_factor: int = 4) -> int:
    """Compute a deterministic DCT pHash as an integer."""
    raw = np.fromfile(Path(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to decode image for pHash: {path}")
    size = hash_size * high_frequency_factor
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low_frequency = dct[:hash_size, :hash_size]
    median = float(np.median(low_frequency.reshape(-1)[1:]))
    bits = (low_frequency > median).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def audit_split_isolation(
    split: dict[str, Any],
    group_metadata: dict[str, dict[str, str | None]] | None = None,
    *,
    phash_threshold: int = 8,
    max_evidence_pairs: int = 100,
) -> dict[str, Any]:
    """Audit exact, capture-group, and perceptual cross-split isolation.

    pHash matches are review candidates, not automatically proven leakage.
    Missing group metadata makes the promotion gate NOT_READY.
    """
    metadata = group_metadata or {}
    records: list[dict[str, Any]] = []
    for role in ("train", "val", "test"):
        for raw_path, label in split[role]:
            path = Path(raw_path).resolve()
            records.append(
                {
                    "role": role,
                    "path": path,
                    "label": int(label),
                    "sha256": file_sha256(path),
                }
            )

    by_digest: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_digest[record["sha256"]].add(record["role"])
    exact_overlaps = [digest for digest, roles in by_digest.items() if len(roles) > 1]

    group_overlaps: dict[str, list[dict[str, Any]]] = {}
    group_coverage: dict[str, dict[str, int | float]] = {}
    for key in GROUP_KEYS:
        values: dict[str, set[str]] = defaultdict(set)
        covered = 0
        for record in records:
            value = metadata.get(record["sha256"], {}).get(key)
            if value:
                covered += 1
                values[value].add(record["role"])
        overlaps = [
            {"group_id": value, "roles": sorted(roles)} for value, roles in sorted(values.items()) if len(roles) > 1
        ]
        group_overlaps[key] = overlaps
        group_coverage[key] = {
            "covered": covered,
            "total": len(records),
            "ratio": covered / len(records) if records else 0.0,
        }

    def compute_record_phash(record: dict[str, Any]) -> tuple[int | None, str | None]:
        try:
            return perceptual_hash(record["path"]), None
        except (OSError, ValueError, cv2.error) as exc:
            return None, str(exc)

    with ThreadPoolExecutor(max_workers=4) as executor:
        phash_results = list(executor.map(compute_record_phash, records))

    phashes: list[int | None] = []
    phash_errors: list[dict[str, str]] = []
    for record, (value, error) in zip(records, phash_results, strict=True):
        phashes.append(value)
        if error is not None:
            phash_errors.append({"path": _portable_path(record["path"]), "error": error})

    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    near_pairs = 0
    pair_evidence: list[dict[str, Any]] = []
    for left in range(len(records)):
        left_hash = phashes[left]
        if left_hash is None:
            continue
        for right in range(left + 1, len(records)):
            if records[left]["sha256"] == records[right]["sha256"]:
                continue
            right_hash = phashes[right]
            if right_hash is None:
                continue
            distance = (left_hash ^ right_hash).bit_count()
            if distance > phash_threshold:
                continue
            union(left, right)
            if records[left]["role"] == records[right]["role"]:
                continue
            near_pairs += 1
            if len(pair_evidence) < max_evidence_pairs:
                pair_evidence.append(
                    {
                        "left": _portable_path(records[left]["path"]),
                        "left_role": records[left]["role"],
                        "right": _portable_path(records[right]["path"]),
                        "right_role": records[right]["role"],
                        "hamming_distance": distance,
                    }
                )

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        clusters[find(index)].append(index)
    cross_role_clusters = []
    for indexes in clusters.values():
        roles = {records[index]["role"] for index in indexes}
        if len(indexes) < 2 or len(roles) < 2:
            continue
        cross_role_clusters.append(
            {
                "roles": sorted(roles),
                "members": [
                    {
                        "path": _portable_path(records[index]["path"]),
                        "role": records[index]["role"],
                        "sha256": records[index]["sha256"],
                    }
                    for index in indexes
                ],
            }
        )

    metadata_complete = bool(records) and all(
        coverage["covered"] == coverage["total"] for coverage in group_coverage.values()
    )
    proven_group_overlap = any(group_overlaps.values())
    if exact_overlaps or proven_group_overlap:
        status = FAIL
    elif not metadata_complete or phash_errors:
        status = NOT_READY
    elif near_pairs:
        status = REVIEW_REQUIRED
    else:
        status = READY

    return {
        "status": status,
        "samples": len(records),
        "exact_content": {
            "status": READY if not exact_overlaps else FAIL,
            "cross_role_overlap_count": len(exact_overlaps),
            "sha256": exact_overlaps,
        },
        "group_metadata": {
            "status": READY
            if metadata_complete and not proven_group_overlap
            else (FAIL if proven_group_overlap else NOT_READY),
            "metadata_entries": len(metadata),
            "coverage": group_coverage,
            "cross_role_overlaps": group_overlaps,
        },
        "perceptual_similarity": {
            "status": NOT_READY if phash_errors else (REVIEW_REQUIRED if near_pairs else READY),
            "method": "DCT pHash",
            "implementation_version": 1,
            "hash_size_bits": 64,
            "hamming_threshold": phash_threshold,
            "cross_role_candidate_pairs": near_pairs,
            "cross_role_clusters": cross_role_clusters,
            "evidence_pairs": pair_evidence,
            "decode_errors": phash_errors,
            "interpretation": "pHash matches require human or embedding/local-feature review before being called leakage",
        },
    }


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    precision, recall, per_class_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    return {
        "samples": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "precision_per_class": dict(zip(CLASS_NAMES, precision.tolist(), strict=True)),
        "recall_per_class": dict(zip(CLASS_NAMES, recall.tolist(), strict=True)),
        "f1_per_class": dict(zip(CLASS_NAMES, per_class_f1.tolist(), strict=True)),
    }


def evaluate_known_and_ood(
    known_probs: np.ndarray,
    known_labels: np.ndarray,
    ood_probs: np.ndarray | None = None,
    *,
    ood_threshold: float = 0.45,
    coverage_thresholds: tuple[float, ...] = (0.0, 0.45, 0.6, 0.75, 0.92),
) -> dict[str, Any]:
    """Report closed-set quality separately from rejection and true OOD."""
    probs = np.asarray(known_probs, dtype=np.float64)
    labels = np.asarray(known_labels, dtype=np.int64)
    if probs.ndim != 2 or probs.shape[1] != len(CLASS_NAMES) or probs.shape[0] != labels.size or not labels.size:
        raise ValueError("known_probs must be non-empty N x 3 and align with known_labels")

    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    rejected = confidence < ood_threshold
    closed_set = _classification_metrics(labels, predictions)
    false_rejections = {
        "status": READY,
        "threshold": ood_threshold,
        "known_samples": int(labels.size),
        "false_rejection_count": int(rejected.sum()),
        "false_rejection_rate": float(rejected.mean()),
    }

    risk_coverage = []
    for threshold in sorted(set(coverage_thresholds)):
        accepted = confidence >= threshold
        accepted_count = int(accepted.sum())
        errors = int((predictions[accepted] != labels[accepted]).sum()) if accepted_count else 0
        risk_coverage.append(
            {
                "confidence_threshold": threshold,
                "coverage": accepted_count / labels.size,
                "accepted": accepted_count,
                "errors": errors,
                "selective_risk": errors / accepted_count if accepted_count else None,
            }
        )

    if ood_probs is None or np.asarray(ood_probs).size == 0:
        ood_detection = {
            "status": NOT_READY,
            "reason": "No labeled true-OOD samples are available; known-class rejections are not OOD evidence",
            "known_false_positive_rate": float(rejected.mean()),
        }
    else:
        ood = np.asarray(ood_probs, dtype=np.float64)
        if ood.ndim != 2 or ood.shape[1] != len(CLASS_NAMES):
            raise ValueError("ood_probs must be M x 3")
        known_scores = 1.0 - confidence
        ood_scores = 1.0 - ood.max(axis=1)
        binary_labels = np.concatenate([np.zeros(labels.size, dtype=np.int64), np.ones(ood.shape[0], dtype=np.int64)])
        scores = np.concatenate([known_scores, ood_scores])
        predicted_ood = scores > (1.0 - ood_threshold)
        true_positive = int(predicted_ood[labels.size :].sum())
        false_positive = int(predicted_ood[: labels.size].sum())
        ood_detection = {
            "status": READY,
            "threshold": ood_threshold,
            "known_samples": int(labels.size),
            "ood_samples": int(ood.shape[0]),
            "true_positive_rate": true_positive / ood.shape[0],
            "known_false_positive_rate": false_positive / labels.size,
            "auroc": float(roc_auc_score(binary_labels, scores)),
            "average_precision": float(average_precision_score(binary_labels, scores)),
        }

    return {
        "closed_set_classification": {"status": READY, **closed_set},
        "known_class_false_rejection": false_rejections,
        "true_ood_detection": ood_detection,
        "selective_risk_coverage": {"status": READY, "points": risk_coverage},
    }


def promotion_readiness(
    set_readiness: dict[str, dict[str, Any]],
    isolation_status: str,
    *,
    canary_pass: bool | None,
    frozen_challenge_pass: bool | None = None,
    true_ood_pass: bool | None = None,
) -> dict[str, Any]:
    """Combine independent release gates without treating canaries as statistics."""
    gates = {
        "canary_regression": READY if canary_pass else NOT_READY,
        "frozen_challenge_data": set_readiness["frozen_challenge"]["status"],
        "frozen_challenge_acceptance": READY if frozen_challenge_pass else NOT_READY,
        "true_ood_data": set_readiness["ood"]["status"],
        "true_ood_acceptance": READY if true_ood_pass else NOT_READY,
        "group_and_perceptual_isolation": isolation_status,
    }
    return {
        "status": READY if all(status == READY for status in gates.values()) else NOT_READY,
        "gates": gates,
        "rule": "Canary prevents known regressions; only independent frozen challenge and true-OOD sets provide promotion statistics",
    }
