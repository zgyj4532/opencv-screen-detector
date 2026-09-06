"""Audit release-training content identities and materialize the frozen split."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiment.cnn_fft_dwt_ablation.harness import (
    CLASS_NAMES,
    DATA_DIR,
    LABEL_OVERRIDES_PATH,
    ROOT,
    _scan_sample_candidates,
    build_split,
    content_sha256,
    load_canary_paths,
    load_label_overrides,
)
from trainer.evaluation_sets import (
    audit_split_isolation,
    evaluation_set_readiness,
    load_group_metadata,
)

DEFAULT_OUTPUT = ROOT / "trainer" / "data_audit.json"


def _relative(path: str) -> str:
    return Path(path).resolve().relative_to(DATA_DIR.resolve()).as_posix()


def build_report(split_seed: int = 42) -> dict:
    candidates = _scan_sample_candidates(DATA_DIR)
    groups: dict[str, list[tuple[str, int, bool, str]]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate[3]].append(candidate)

    overrides = load_label_overrides()
    conflicts = []
    reviewed_decisions = []
    for digest, group in sorted(groups.items()):
        labels = sorted({candidate[1] for candidate in group})
        if digest in overrides:
            reviewed_decisions.append(
                {
                    "sha256": digest,
                    "observed_labels": [CLASS_NAMES[label] for label in labels],
                    "reviewed_label": CLASS_NAMES[overrides[digest]],
                    "paths": [_relative(path) for path, _label, _is_hard, _digest in sorted(group)],
                }
            )
        if len(labels) <= 1:
            continue
        conflicts.append(
            {
                "sha256": digest,
                "observed_labels": [CLASS_NAMES[label] for label in labels],
                "reviewed_label": CLASS_NAMES[overrides[digest]] if digest in overrides else None,
                "paths": [
                    {"path": _relative(path), "observed_label": CLASS_NAMES[label], "hard_negative": is_hard}
                    for path, label, is_hard, _digest in sorted(group)
                ],
            }
        )

    unresolved = [row["sha256"] for row in conflicts if row["reviewed_label"] is None]
    if unresolved:
        raise RuntimeError(
            f"Unresolved content-label conflicts: {', '.join(unresolved)}. Review {LABEL_OVERRIDES_PATH}."
        )

    split = build_split(seed=split_seed, canary_paths=load_canary_paths())
    role_hashes: dict[str, set[str]] = {}
    role_counts = {}
    for role in ("train", "val", "test"):
        hashes = [content_sha256(path) for path, _label in split[role]]
        role_hashes[role] = set(hashes)
        counts = Counter(label for _path, label in split[role])
        role_counts[role] = {
            "total": len(split[role]),
            "classes": {name: counts[index] for index, name in enumerate(CLASS_NAMES)},
        }

    overlap = {
        "train_val": sorted(role_hashes["train"] & role_hashes["val"]),
        "train_test": sorted(role_hashes["train"] & role_hashes["test"]),
        "val_test": sorted(role_hashes["val"] & role_hashes["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Frozen split contains cross-role content leakage: {overlap}")

    hard_hashes = {digest for digest, group in groups.items() if any(candidate[2] for candidate in group)}
    hard_content_in_eval = {role: sorted(role_hashes[role] & hard_hashes) for role in ("val", "test")}
    if any(hard_content_in_eval.values()):
        raise RuntimeError(f"Hard-negative content escaped into evaluation: {hard_content_in_eval}")

    used_hashes = set(groups)
    isolation = audit_split_isolation(split, load_group_metadata())

    return {
        "schema_version": 2,
        "raw_paths": len(candidates),
        "unique_content": len(groups),
        "duplicate_content_groups": sum(len(group) > 1 for group in groups.values()),
        "cross_source_duplicate_groups": sum(
            any(candidate[2] for candidate in group) and any(not candidate[2] for candidate in group)
            for group in groups.values()
        ),
        "duplicate_paths_removed": len(candidates) - len(groups),
        "raw_conflicting_label_groups": len(conflicts),
        "unresolved_conflicting_label_groups": len(unresolved),
        "reviewed_overrides": len(overrides),
        "unused_overrides": sorted(set(overrides) - used_hashes),
        "split": {**split["meta"], **role_counts},
        "cross_role_content_overlap": {name: len(hashes) for name, hashes in overlap.items()},
        "split_isolation": isolation,
        "evaluation_sets": evaluation_set_readiness(),
        "hard_negative_content_in_eval": {role: len(hashes) for role, hashes in hard_content_in_eval.items()},
        "reviewed_label_decisions": reviewed_decisions,
        "conflicts": conflicts,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = build_report(args.split_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"Data audit: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
