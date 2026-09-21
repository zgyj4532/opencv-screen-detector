"""Freeze validation-only model selection before opening historical test results."""

from datetime import UTC, datetime

from experiment.daily_20260922.src.protocol import (
    DATA,
    REPORTS,
    SEEDS,
    VARIANTS,
    candidate_name,
    digest,
    read_json,
    write_new,
)


def rank_variants(reports):
    ranking = []
    for variant in VARIANTS:
        rows = [reports[candidate_name(variant, seed)] for seed in SEEDS]
        if any(row["role"] != "val" for row in rows):
            raise ValueError("Selection must use validation reports only")
        eligible = all(
            row["canary"]["total"] > 0 and row["canary"]["correct"] == row["canary"]["total"] for row in rows
        )
        ranking.append(
            {
                "variant": variant,
                "eligible": eligible,
                "mean_metric": sum(row["metrics"]["metric"] for row in rows) / len(rows),
                "metrics_by_seed": {str(seed): row["metrics"] for seed, row in zip(SEEDS, rows, strict=True)},
            }
        )
    return sorted(ranking, key=lambda row: (-row["mean_metric"], row["variant"]))


def main():
    if (REPORTS / "selection.json").exists() or list(REPORTS.glob("*.test.json")):
        raise RuntimeError("Selection already frozen or test already opened")
    reports, hashes = {}, {}
    protocol_sha256 = digest(DATA / "protocol.json")
    for variant in VARIANTS:
        for seed in SEEDS:
            name = candidate_name(variant, seed)
            path = REPORTS / f"{name}.val.json"
            row = read_json(path)
            if row["id"] != name or row["protocol_sha256"] != protocol_sha256 or row["parity"]["status"] != "PASS":
                raise RuntimeError(f"Invalid or incompatible validation evidence: {name}")
            reports[name], hashes[name] = row, digest(path)
    ranking = rank_variants(reports)
    eligible = [row for row in ranking if row["eligible"]]
    selected = eligible[0]["variant"] if eligible else None
    allowed = ["production"]
    if selected is not None:
        allowed += sorted({candidate_name(variant, seed) for variant in ("baseline", selected) for seed in SEEDS})
    selection = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol_sha256,
        "ranking": ranking,
        "selected_variant": selected,
        "primary_seed": 42,
        "validation_report_sha256": hashes,
        "test_allowed_ids": allowed,
        "selection_basis": "mean Predictor validation metric across two seeds, then all-Canary pass eligibility",
        "promotion": "candidate only; independent challenge/OOD and group isolation evidence still required",
    }
    write_new(REPORTS / "selection.json", selection)
    print(selection, flush=True)


if __name__ == "__main__":
    main()
