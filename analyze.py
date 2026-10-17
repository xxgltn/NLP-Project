import json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
from common import UPOS, test_subset, train_ambiguity, save_json
from pt_train import parse_lenient

SYSTEMS = {
    "llama-1b (prompting)": ("runs_hf", "llama-1b_k{k}_s{s}"),
    "llama-1b (prompt tuning)": ("runs_pt", "llama-1b_k{k}_s{s}_lr0.01"),
    "flan-t5-base (prompt tuning)": ("runs_pt", "flan-t5-base_k{k}_s{s}_lr0.3"),
    "flan-t5-large (prompt tuning)": ("runs_pt", "flan-t5-large_k{k}_s{s}_lr0.3"),
}

def load(system, k, seeds=(0, 1, 2)):
    folder, pat = SYSTEMS[system]
    out = {}
    for s in seeds:
        cands = list(Path(folder).glob(pat.format(k=k, s=s) + "*"))
        if not cands:
            continue
        p = json.loads((cands[0] / "predictions.json").read_text())
        out[s] = [(x["words"], x["gold"], parse_lenient(x["raw"], x["words"])) for x in p]
    return out

def per_sentence_scores(runs):
    seeds = sorted(runs)
    n = len(runs[seeds[0]])
    cor = [sum(sum(g == p for g, p in zip(runs[s][i][1], runs[s][i][2])) for s in seeds) / len(seeds)
           for i in range(n)]
    tot = [len(runs[seeds[0]][i][1]) for i in range(n)]
    return cor, tot

def compare(a, b, k, n_boot=10000, seed=0):
    ra, rb = load(a, k), load(b, k)
    if not ra or not rb:
        print(f"k={k}: missing runs for {a if not ra else b}"); return
    ca, tot = per_sentence_scores(ra)
    cb, _ = per_sentence_scores(rb)
    acc_a, acc_b = sum(ca) / sum(tot), sum(cb) / sum(tot)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        pick = [rng.randrange(len(tot)) for _ in tot]
        T = sum(tot[i] for i in pick)
        diffs.append(sum(ca[i] for i in pick) / T - sum(cb[i] for i in pick) / T)
    diffs.sort()
    lo, hi = diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]
    p = 2 * min(sum(d <= 0 for d in diffs), sum(d >= 0 for d in diffs)) / n_boot
    print(f"k={k:3d}  {a} {acc_a*100:.1f}  vs  {b} {acc_b*100:.1f}  "
          f"diff {100*(acc_a-acc_b):+.1f}  95% CI [{lo*100:+.1f}, {hi*100:+.1f}]  p={p:.4f}"
          f"  {'significant' if lo * hi > 0 else 'NOT significant'}  (seeds {sorted(ra)} vs {sorted(rb)})")

def errors(run_dir, n_sample=50, seed=0):
    p = json.loads((Path(run_dir) / "predictions.json").read_text())
    amb = train_ambiguity()
    conf, bad = Counter(), []
    tag_stats = defaultdict(lambda: [0, 0])
    for x in p:
        pred = parse_lenient(x["raw"], x["words"])
        for w, g, pr in zip(x["words"], x["gold"], pred):
            tag_stats[g][1] += 1
            if g == pr:
                tag_stats[g][0] += 1
            else:
                conf[(g, pr)] += 1
                bad.append({"word": w, "gold": g, "pred": pr,
                            "ambiguous_type": w.lower() in amb,
                            "sentence": " ".join(x["words"])})
    print(f"{run_dir}: {len(bad)} wrong tokens\n\ntop confusions (gold -> predicted):")
    for (g, pr), n in conf.most_common(12):
        print(f"  {g:6} -> {pr:10} {n:4}")
    print("\nworst tags (accuracy, support):")
    for t, (c, n) in sorted(tag_stats.items(), key=lambda x: x[1][0] / max(x[1][1], 1))[:8]:
        print(f"  {t:6} {c / n:5.1%}  (n={n})")
    sample = random.Random(seed).sample(bad, min(n_sample, len(bad)))
    out = Path(run_dir) / "error_sample.csv"
    with out.open("w", encoding="utf-8") as f:
        f.write("word,gold,pred,ambiguous_type,my_label,sentence\n")
        for e in sample:
            f.write(f'"{e["word"]}",{e["gold"]},{e["pred"]},{e["ambiguous_type"]},,'
                    f'"{e["sentence"][:200]}"\n')
    print(f"\n{len(sample)} errors written to {out} - label the 'my_label' column by hand: "
          "convention / ambiguity / format / gold_error / other")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "compare":
        compare(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif cmd == "compare_all":
        for k in (8, 32):
            compare("llama-1b (prompt tuning)", "llama-1b (prompting)", k)
        for k in (8, 32, 128):
            compare("llama-1b (prompt tuning)", "flan-t5-large (prompt tuning)", k)
    elif cmd == "errors":
        errors(sys.argv[2])
