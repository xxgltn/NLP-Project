"""Collect baselines + all Claude runs into one markdown results table.
Usage: python summarize.py > results/week1_table.md"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path

rows = defaultdict(list)  # (system, k) -> list of metric dicts
b = json.loads(Path("results/baselines.json").read_text())
for r in b["few_shot"]:
    rows[(r["system"], r["k"])].append(r)
for p in sorted(Path("runs").glob("*/metrics.json")):
    r = json.loads(p.read_text()); c = r["config"]
    if c.get("limit"):  # smoke tests are not results
        continue
    name = f"Claude-{c['model']}" + (f"[{c['tag']}]" if c.get("tag") else "")
    rows[(name, c["k"])].append(r)

def ms(v):
    return f"{st.mean(v):.3f}" + (f" ± {st.stdev(v):.3f}" if len(v) > 1 else "")

print("| System | k | seeds | Token acc | Macro-F1 | Sent. exact | Unseen-word acc | Format errors |")
print("|---|---|---|---|---|---|---|---|")
for (name, k), rs in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0])):
    uns = [r["buckets"]["unseen_in_pool"]["acc"] for r in rs if "unseen_in_pool" in r["buckets"]]
    fe = [r.get("missing_rate", 0) for r in rs]
    print(f"| {name} | {k} | {len(rs)} | {ms([r['token_acc'] for r in rs])} | "
          f"{ms([r['macro_f1'] for r in rs])} | {ms([r['sent_exact'] for r in rs])} | "
          f"{ms(uns) if uns else 'n/a'} | {st.mean(fe):.3%} |")
ref = b["reference"]["CRF_full_train_on_subset"]
print(f"| CRF (full train, 12.5k sents; reference, not few-shot) | 12544 | 1 | {ref['token_acc']:.3f} | {ref['macro_f1']:.3f} | {ref['sent_exact']:.3f} | n/a | 0 |")
