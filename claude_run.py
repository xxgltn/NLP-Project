"""Claude few-shot POS tagging runner.

Examples
  python claude_run.py --k 0 --dry-run                     # show prompt + cost estimate, no API call
  python claude_run.py --k 8 --seed 0 --limit 20 --sync    # cheap smoke test (20 sentences)
  python claude_run.py --k 8 --seed 0 submit               # Batch API (50% cheaper), returns batch id
  python claude_run.py --k 8 --seed 0 collect              # fetch results, parse, evaluate

Every raw API response is stored under runs/<run_name>/ and is NEVER re-requested,
so re-running evaluation costs nothing.  Requires ANTHROPIC_API_KEY in the environment.
"""
import argparse, json, re, sys, time
from pathlib import Path
from common import UPOS, pool, test_subset, evaluate, train_ambiguity, save_json

MODELS = {"haiku": "claude-haiku-4-5-20251001", "sonnet": "claude-sonnet-5"}
# USD per million tokens, list prices as of Sept 2026 -- VERIFY on the pricing page.
PRICES = {"haiku": (1.0, 5.0), "sonnet": (3.0, 15.0)}
BATCH_DISCOUNT = 0.5

TAG_GLOSS = {  # short, self-written glosses (NOT the official UD guideline text)
    "ADJ": "adjective", "ADP": "adposition (preposition/postposition)", "ADV": "adverb",
    "AUX": "auxiliary (incl. copula 'be', modals, tense/aspect 'have'/'do')",
    "CCONJ": "coordinating conjunction", "DET": "determiner", "INTJ": "interjection",
    "NOUN": "common noun", "NUM": "numeral", "PART": "particle (e.g. 's, not, infinitival to)",
    "PRON": "pronoun", "PROPN": "proper noun", "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction", "SYM": "symbol", "VERB": "main verb",
    "X": "other (foreign words, typos, unanalysable)"}

INSTRUCTIONS = (
    "You are an expert linguistic annotator. Tag every token with its Universal "
    "Dependencies (UD v2) universal part-of-speech tag (UPOS), following English UD "
    "(EWT) conventions.\n\nTagset:\n"
    + "\n".join(f"{t}: {g}" for t, g in TAG_GLOSS.items())
    + "\n\nThe sentence is already tokenized; tokens are numbered. Do not merge, split "
    "or skip tokens. Output exactly one line per token in the form `<number> <TAG>` "
    "and nothing else.")


def fmt_input(words):
    return "\n".join(f"{i+1} {w}" for i, w in enumerate(words))


def fmt_output(tags):
    return "\n".join(f"{i+1} {t}" for i, t in enumerate(tags))


def system_prompt(demos):
    s = INSTRUCTIONS
    if demos:
        s += "\n\nAnnotated examples:\n"
        for d in demos:
            s += f"\n<input>\n{fmt_input(d['words'])}\n</input>\n<output>\n{fmt_output(d['tags'])}\n</output>\n"
    return s


def build_params(model_key, sys_prompt, words):
    return {
        "model": MODELS[model_key],
        "max_tokens": 8 * len(words) + 64,
        "temperature": 0.0,
        # System prompt is identical across the run -> cacheable prefix. Caching only
        # activates above a model-specific minimum length (check docs); harmless otherwise.
        "system": [{"type": "text", "text": sys_prompt, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": f"<input>\n{fmt_input(words)}\n</input>"}],
    }


LINE = re.compile(r"^\s*(\d+)\s*[\t :.\-|]*\s*([A-Z]+)\b")


def parse(text, n):
    """Return (tags, n_bad). Unparseable / missing positions -> '_MISSING';
    tags outside UPOS -> '_INVALID'. Both count as errors in evaluation."""
    got = {}
    for line in text.splitlines():
        m = LINE.match(line)
        if m:
            i = int(m.group(1)) - 1
            if 0 <= i < n and i not in got:
                t = m.group(2)
                got[i] = t if t in UPOS else "_INVALID"
    tags = [got.get(i, "_MISSING") for i in range(n)]
    return tags, sum(t.startswith("_") for t in tags)


def run_dir(a):
    return Path("runs") / f"{a.model}_k{a.k}_s{a.seed}_n{a.limit or 'all'}{('_' + a.tag) if a.tag else ''}"


def setup(a):
    test = test_subset()[: a.limit] if a.limit else test_subset()
    demos = pool(a.k, a.seed) if a.k > 0 else []
    return test, demos, system_prompt(demos)


def estimate(a, test, sp):
    ci, co = PRICES[a.model]
    n_in = sum((len(sp) + len(fmt_input(s["words"])) + 40) / 3.5 for s in test)  # rough chars->tokens
    n_out = sum(4 * len(s["words"]) for s in test)
    cost = (n_in * ci + n_out * co) / 1e6
    print(f"~{n_in/1e6:.3f}M input tok, ~{n_out/1e6:.3f}M output tok -> "
          f"~${cost:.2f} sync / ~${cost*BATCH_DISCOUNT:.2f} batch (before caching; rough)")


def finish(a, test, demos, raw):
    """raw: {idx: text}. Parse, evaluate, log cost."""
    d = run_dir(a)
    preds, bad = [], 0
    for i, s in enumerate(test):
        tags, b = parse(raw.get(str(i), ""), len(s["words"]))
        preds.append(tags); bad += b > 0
    r = evaluate(test, preds, pool_sents=demos or None, ambiguous=train_ambiguity())
    r["sents_with_format_errors"] = bad
    usage = [json.loads(l) for l in (d / "usage.jsonl").read_text().splitlines()] if (d / "usage.jsonl").exists() else []
    if usage:
        ci, co = PRICES[a.model]; mult = BATCH_DISCOUNT if a.mode == "batch" else 1.0
        tin = sum(u["input_tokens"] for u in usage); tout = sum(u["output_tokens"] for u in usage)
        cr = sum(u.get("cache_read_input_tokens") or 0 for u in usage)
        cw = sum(u.get("cache_creation_input_tokens") or 0 for u in usage)
        r["usage"] = {"input": tin, "output": tout, "cache_read": cr, "cache_write": cw,
                      "usd": round(mult * (tin * ci + tout * co + cr * 0.1 * ci + cw * 1.25 * ci) / 1e6, 4)}
    save_json({"config": vars(a), **r}, d / "metrics.json")
    save_json([{"id": s["id"], "words": s["words"], "gold": s["tags"], "pred": p}
               for s, p in zip(test, preds)], d / "predictions.json")
    print(json.dumps({k: r[k] for k in ("token_acc", "macro_f1", "sent_exact", "missing_rate",
                                        "sents_with_format_errors", "buckets")}, indent=1))
    if "usage" in r:
        print("cost:", r["usage"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", nargs="?", default="run", choices=["run", "submit", "collect"])
    ap.add_argument("--model", default="haiku", choices=MODELS)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N test-subset sentences only")
    ap.add_argument("--tag", default="", help="suffix for run name (prompt variant)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sync", action="store_true", help="synchronous calls instead of Batch API")
    a = ap.parse_args()
    a.mode = "sync" if a.sync else "batch"
    test, demos, sp = setup(a)
    d = run_dir(a); d.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        print("=== SYSTEM ===\n" + sp[:3000] + ("\n...[truncated]" if len(sp) > 3000 else ""))
        print("=== USER (first test sentence) ===\n" + build_params(a.model, sp, test[0]["words"])["messages"][0]["content"])
        estimate(a, test, sp); return

    import anthropic
    client = anthropic.Anthropic()
    raw_path = d / "raw.jsonl"
    raw = {json.loads(l)["idx"]: json.loads(l)["text"] for l in raw_path.read_text().splitlines()} if raw_path.exists() else {}

    if a.action == "run":  # synchronous; resumes where it stopped
        a.mode = "sync"
        with raw_path.open("a") as fr, (d / "usage.jsonl").open("a") as fu:
            for i, s in enumerate(test):
                if str(i) in raw:
                    continue
                msg = client.messages.create(**build_params(a.model, sp, s["words"]))
                text = "".join(b.text for b in msg.content if b.type == "text")
                raw[str(i)] = text
                fr.write(json.dumps({"idx": str(i), "text": text}) + "\n"); fr.flush()
                fu.write(json.dumps(msg.usage.model_dump()) + "\n"); fu.flush()
        finish(a, test, demos, raw)
    elif a.action == "submit":
        if (d / "batch_id.txt").exists():
            sys.exit(f"Batch already submitted: {(d / 'batch_id.txt').read_text()} -- use collect")
        reqs = [{"custom_id": f"s{i}", "params": build_params(a.model, sp, s["words"])}
                for i, s in enumerate(test)]
        b = client.messages.batches.create(requests=reqs)
        (d / "batch_id.txt").write_text(b.id)
        print("submitted", b.id, "- run `collect` later (usually minutes, max 24h)")
    else:  # collect
        bid = (d / "batch_id.txt").read_text().strip()
        while client.messages.batches.retrieve(bid).processing_status != "ended":
            print("waiting..."); time.sleep(30)
        with raw_path.open("w") as fr, (d / "usage.jsonl").open("w") as fu:
            for res in client.messages.batches.results(bid):
                i = res.custom_id[1:]
                if res.result.type == "succeeded":
                    m = res.result.message
                    text = "".join(b.text for b in m.content if b.type == "text")
                    fu.write(json.dumps(m.usage.model_dump()) + "\n")
                else:
                    text = ""; print("failed:", res.custom_id, res.result.type)
                raw[i] = text
                fr.write(json.dumps({"idx": i, "text": text}) + "\n")
        finish(a, test, demos, raw)


if __name__ == "__main__":
    main()
