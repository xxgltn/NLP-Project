"""One results table: traditional baselines, prompting (lenient), prompt tuning (strict + lenient).
Accuracy in %, mean ± sd over pool draws."""
import json, statistics as st
from collections import defaultdict
from pathlib import Path

rows = defaultdict(lambda: defaultdict(list))  # system -> k -> list of (acc, strict_acc, missing)
for r in json.loads(Path("results/baselines.json").read_text())["few_shot"]:
    rows[r["system"]][r["k"]].append((r["token_acc"], None, 0.0))
for d in sorted(Path("runs_hf").glob("*/")):
    s, l = json.loads((d / "metrics.json").read_text()), json.loads((d / "metrics_lenient.json").read_text())
    c = s["config"]
    rows[f"{c['model']} (prompting)"][c["k"]].append((l["token_acc"], s["token_acc"], l["missing_rate"]))
for d in sorted(Path("runs_pt").glob("*/")):
    if not (d / "metrics_lenient.json").exists():
        continue
    s, l = json.loads((d / "metrics.json").read_text()), json.loads((d / "metrics_lenient.json").read_text())
    c = s["config"]
    rows[f"{c['model']} (prompt tuning)"][c["k"]].append((l["token_acc"], s["token_acc"], l["missing_rate"]))

def fmt(v):
    v = [x * 100 for x in v]
    return f"{st.mean(v):5.1f}" + (f" ± {st.stdev(v):4.1f}" if len(v) > 1 else "       ") + f" (n={len(v)})"

ks = [0, 8, 32, 128]
print("LENIENT token accuracy (%)   [strict accuracy and missing-rate in the second table]")
print(f"{'system':32}" + "".join(f"{'k=' + str(k):>20}" for k in ks))
for name in rows:
    print(f"{name:32}" + "".join(f"{(fmt([x[0] for x in rows[name][k]]) if rows[name].get(k) else '-'):>20}" for k in ks))
print("\nSTRICT accuracy / missing-rate (lenient), LLM systems only")
for name in rows:
    if "(" not in name:
        continue
    cells = []
    for k in ks:
        v = rows[name].get(k)
        cells.append(f"{st.mean(x[1] for x in v)*100:5.1f} / {st.mean(x[2] for x in v)*100:5.1f}%" if v else "-")
    print(f"{name:32}" + "".join(f"{c:>20}" for c in cells))
