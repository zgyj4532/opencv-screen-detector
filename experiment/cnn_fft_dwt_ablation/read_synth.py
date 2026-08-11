import json
from pathlib import Path

p = Path(
    r"C:\Users\ll\AppData\Local\Temp\claude\E--github-opencv-screen-detector\8a2c7dd6-f924-4c3b-822c-5e3c797ab266\tasks\wn15ra3vh.output"
)
d = json.loads(p.read_text(encoding="utf-8"))
r = d["result"]
if isinstance(r, str):
    r = json.loads(r)
s = r["synthesis"]
out = []
out.append("=== RECOMMENDED DEFAULT CONFIG ===")
out.append(s["recommended_default_config"])
out.append("\n=== EXPERIMENT MATRIX (by priority) ===")
out.extend(
    f"P{e['priority']} [{e['id']}] {e['name']}\n    hyp: {e['hypothesis']}\n    change: {e['change']}"
    for e in sorted(s["experiment_matrix"], key=lambda x: x["priority"])
)
out.append("\n=== RISKS ===")
out.extend("- " + x for x in s.get("risks", []))
content = "\n".join(out)
Path("experiment/cnn_fft_dwt_ablation/synth.txt").write_text(content, encoding="utf-8")
print("written", len(content))
