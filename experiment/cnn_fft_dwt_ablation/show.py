"""Print the experiment leaderboard sorted by a metric."""

import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parent / "leaderboard.jsonl"
rows = [json.loads(line) for line in LB.read_text(encoding="utf-8").splitlines() if line.strip()]
key = sys.argv[1] if len(sys.argv) > 1 else "test_metric"
rows.sort(key=lambda r: r.get(key, 0), reverse=True)
hdr = f"{'id':<12} {'test_acc':>8} {'test_spF1':>9} {'spP':>6} {'spR':>6} {'macroF1':>8} {'metric':>7} {'thrF1':>7} {'thr':>5} {'val_acc':>7} {'s':>5}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    t = r.get("test_thr", {})
    print(
        f"{r['id']:<12} {r['test_acc']:>8.4f} {r['test_sp_f1']:>9.4f} "
        f"{r['test_sp_precision']:>6.3f} {r['test_sp_recall']:>6.3f} {r['test_macro_f1']:>8.4f} "
        f"{r['test_metric']:>7.4f} {t.get('sp_f1', 0):>7.4f} {t.get('threshold', 0):>5.2f} "
        f"{r['val_acc']:>7.4f} {r['elapsed_s']:>5.0f}"
    )
print(f"\n(sorted by {key}; {len(rows)} experiments; baseline to beat: acc 0.8946, macroF1 0.8668, sp_f1 0.7630)")
