"""Non-LLM baselines on the SAME k-sentence pools the LLM will see.
Usage:  python baselines.py            (few-shot grid + full-data CRF reference)
Output: results/baselines.json and a printed summary table.
"""
import statistics as st
from collections import Counter, defaultdict
import sklearn_crfsuite
from common import (K_VALUES, POOL_SEEDS, load_conllu, pool, test_subset,
                    evaluate, train_ambiguity, save_json)


def mft_train(sents):
    c = defaultdict(Counter); overall = Counter()
    for s in sents:
        for w, t in zip(s["words"], s["tags"]):
            c[w.lower()][t] += 1; overall[t] += 1
    lex = {w: cnt.most_common(1)[0][0] for w, cnt in c.items()}
    return lex, overall.most_common(1)[0][0]


def mft_predict(model, sents):
    lex, default = model
    return [[lex.get(w.lower(), default) for w in s["words"]] for s in sents]


def feats(words, i):
    w = words[i]
    f = {"bias": 1.0, "w.lower": w.lower(), "suf1": w[-1:], "suf2": w[-2:],
         "suf3": w[-3:], "pre2": w[:2], "is_title": w.istitle(),
         "is_upper": w.isupper(), "is_digit": w.isdigit(),
         "has_digit": any(ch.isdigit() for ch in w), "has_hyphen": "-" in w,
         "is_alpha": w.isalpha(), "len": min(len(w), 10)}
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(words):
            f[f"{off}:w.lower"] = words[j].lower()
            f[f"{off}:suf2"] = words[j][-2:]
            f[f"{off}:is_title"] = words[j].istitle()
        else:
            f[f"{off}:pad"] = True
    return f


def crf_train(sents):
    X = [[feats(s["words"], i) for i in range(len(s["words"]))] for s in sents]
    y = [s["tags"] for s in sents]
    m = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=200)
    m.fit(X, y)
    return m


def crf_predict(m, sents):
    X = [[feats(s["words"], i) for i in range(len(s["words"]))] for s in sents]
    return m.predict(X)


def main():
    test = test_subset()
    amb = train_ambiguity()
    out = {"few_shot": [], "reference": {}}
    for k in K_VALUES:
        for seed in POOL_SEEDS:
            p = pool(k, seed)
            for name, tr, pr in [("MFT", mft_train, mft_predict),
                                 ("CRF", crf_train, crf_predict)]:
                m = tr(p)
                r = evaluate(test, pr(m, test), pool_sents=p, ambiguous=amb)
                out["few_shot"].append({"system": name, "k": k, "seed": seed, **r})
    # Full-data reference (NOT few-shot): upper-bound context for the thesis.
    train = load_conllu("train")
    m = crf_train(train)
    out["reference"]["CRF_full_train_on_subset"] = evaluate(test, crf_predict(m, test), ambiguous=amb)
    full_test = load_conllu("test")
    out["reference"]["CRF_full_train_on_full_test"] = evaluate(full_test, crf_predict(m, full_test))
    save_json(out, "results/baselines.json")

    print(f"{'system':6} {'k':>4}  {'acc mean±sd':>14}  {'macroF1':>8}  {'unseen-acc':>10}")
    for name in ("MFT", "CRF"):
        for k in K_VALUES:
            rs = [r for r in out["few_shot"] if r["system"] == name and r["k"] == k]
            a = [r["token_acc"] for r in rs]
            f = st.mean(r["macro_f1"] for r in rs)
            u = st.mean(r["buckets"]["unseen_in_pool"]["acc"] for r in rs)
            print(f"{name:6} {k:>4}  {st.mean(a):.3f}±{st.stdev(a):.3f}     {f:.3f}     {u:.3f}")
    for key, r in out["reference"].items():
        print(f"{key}: acc={r['token_acc']:.4f} macroF1={r['macro_f1']:.4f} (n_tok={r['n_tokens']})")


if __name__ == "__main__":
    main()
