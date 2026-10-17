"""Shared utilities: data loading, few-shot pool sampling, evaluation.

Design rules (see thesis Methodology):
  * Systems may only see the k-sentence POOL drawn from the train split.
  * The fixed TEST SUBSET is never inspected while designing prompts.
  * Train-split statistics are used ONLY for post-hoc analysis (ambiguity buckets),
    never as input to any system.
"""
import json, random, re
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
UD_VERSION = "r2.18"  # pinned release of UD_English-EWT
UPOS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
        "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
TEST_SUBSET_SIZE = 300
TEST_SUBSET_SEED = 0
K_VALUES = [8, 32, 128]
POOL_SEEDS = [0, 1, 2]


def load_conllu(split):
    """Return list of dicts {id, words, tags}. Uses syntactic words:
    multiword-token range lines (3-4) and empty nodes (5.1) are skipped."""
    path = DATA_DIR / f"en_ewt-ud-{split}.conllu"
    sents, words, tags, sid = [], [], [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# sent_id"):
            sid = line.split("=", 1)[1].strip()
        elif not line.strip():
            if words:
                sents.append({"id": sid, "words": words, "tags": tags})
            words, tags = [], []
        elif not line.startswith("#"):
            cols = line.split("\t")
            if re.fullmatch(r"\d+", cols[0]):
                words.append(cols[1]); tags.append(cols[3])
    if words:
        sents.append({"id": sid, "words": words, "tags": tags})
    return sents


def test_subset():
    test = load_conllu("test")
    rng = random.Random(TEST_SUBSET_SEED)
    return rng.sample(test, TEST_SUBSET_SIZE)


def pool(k, seed):
    """Nested pools: pool(8,s) is a prefix of pool(32,s) is a prefix of pool(128,s).
    This reduces noise in learning curves (larger k only ADDS data)."""
    train = load_conllu("train")
    idx = list(range(len(train)))
    random.Random(1000 + seed).shuffle(idx)
    return [train[i] for i in idx[:k]]


# ---------------------------------------------------------------- evaluation
def train_ambiguity():
    """Word forms (lowercased) seen with >1 UPOS tag in the full train split.
    ANALYSIS ONLY."""
    c = defaultdict(Counter)
    for s in load_conllu("train"):
        for w, t in zip(s["words"], s["tags"]):
            c[w.lower()][t] += 1
    return {w for w, cnt in c.items() if len(cnt) > 1}


def evaluate(gold_sents, pred_tags, pool_sents=None, ambiguous=None):
    """gold_sents: list of sentence dicts; pred_tags: list of tag lists (same
    length as each sentence; use '_MISSING' for unaligned tokens)."""
    tot = cor = exact = 0
    tp, fp, fn = Counter(), Counter(), Counter()
    confusions = Counter()
    buckets = defaultdict(lambda: [0, 0])  # name -> [correct, total]
    pool_vocab = {w.lower() for s in (pool_sents or []) for w in s["words"]}
    missing = 0
    for g, p in zip(gold_sents, pred_tags):
        assert len(g["tags"]) == len(p), "prediction length must be aligned"
        ok_sent = True
        for w, gt, pt in zip(g["words"], g["tags"], p):
            tot += 1
            hit = gt == pt
            cor += hit
            missing += pt == "_MISSING"
            if hit:
                tp[gt] += 1
            else:
                ok_sent = False
                fn[gt] += 1; fp[pt] += 1
                confusions[(gt, pt)] += 1
            if pool_sents is not None:
                b = "seen_in_pool" if w.lower() in pool_vocab else "unseen_in_pool"
                buckets[b][0] += hit; buckets[b][1] += 1
            if ambiguous is not None:
                b = "ambiguous_type" if w.lower() in ambiguous else "unambiguous_type"
                buckets[b][0] += hit; buckets[b][1] += 1
        exact += ok_sent
    per_tag = {}
    for t in UPOS:
        if tp[t] + fn[t] == 0:
            continue
        P = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else 0.0
        R = tp[t] / (tp[t] + fn[t])
        F = 2 * P * R / (P + R) if P + R else 0.0
        per_tag[t] = {"P": round(P, 4), "R": round(R, 4), "F1": round(F, 4),
                      "support": tp[t] + fn[t]}
    return {
        "token_acc": round(cor / tot, 4),
        "macro_f1": round(sum(v["F1"] for v in per_tag.values()) / len(per_tag), 4),
        "sent_exact": round(exact / len(gold_sents), 4),
        "missing_rate": round(missing / tot, 4),
        "n_tokens": tot, "n_sents": len(gold_sents),
        "buckets": {k: {"acc": round(c / n, 4), "n": n} for k, (c, n) in buckets.items()},
        "per_tag": per_tag,
        "top_confusions": [{"gold": g, "pred": p, "n": n}
                           for (g, p), n in confusions.most_common(15)],
    }


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
