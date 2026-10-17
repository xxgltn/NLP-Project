import json, re
from pathlib import Path
from common import UPOS, pool, evaluate, train_ambiguity

def parse_lenient(text, words):
    """Strict parser + 3 fixed rules: read all lines; 'word / TAG' -> 'word/TAG';
    'word/X/TAG' matched on the part before the first slash, tag = last part."""
    text = re.sub(r"\s+/\s+", "/", text.replace("\n", " "))
    tags = ["_MISSING"] * len(words)
    j = 0
    for item in text.split():
        if j >= len(words):
            break
        if "/" not in item:
            continue
        w, t = item.rsplit("/", 1)
        t = t.strip(".,;:").upper()
        cands = {w, w.split("/")[0]}
        for look in range(j, min(j + 4, len(words))):
            if any(c == words[look] or c.lower() == words[look].lower() for c in cands):
                tags[look] = t if t in UPOS else "_INVALID"
                j = look + 1
                break
    return tags

amb = train_ambiguity()
print(f"{'run':22} {'strict acc':>10} {'lenient acc':>11} {'strict miss':>11} {'lenient miss':>12}")
for d in sorted(Path("runs_hf").glob("*/")):
    m = json.loads((d / "metrics.json").read_text()); c = m["config"]
    p = json.loads((d / "predictions.json").read_text())
    gold = [{"words": x["words"], "tags": x["gold"]} for x in p]
    preds = [parse_lenient(x["raw"], x["words"]) for x in p]
    demos = pool(c["k"], c["seed"]) if c["k"] > 0 else None
    r = evaluate(gold, preds, pool_sents=demos, ambiguous=amb)
    (d / "metrics_lenient.json").write_text(json.dumps({"config": c, **r}, indent=2))
    print(f"{d.name:22} {m['token_acc']:>10.3f} {r['token_acc']:>11.3f} "
          f"{m['missing_rate']:>11.1%} {r['missing_rate']:>12.1%}")
