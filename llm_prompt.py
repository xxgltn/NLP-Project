"""Zero-shot and few-shot PROMPTING of open LLMs (no training, weights frozen).

Usage (Colab, GPU runtime):
  python llm_prompt.py --model flan-t5-base --k 0
  python llm_prompt.py --model llama-1b --k 8 --seed 0
  python llm_prompt.py --grid            # all Week-1 runs, skips finished ones

Results go to runs_hf/<model>_k<k>_s<seed>/metrics.json (+ predictions.json).
A finished run is never recomputed, so if Colab disconnects just run the cell again.
"""
import argparse, json, time
from pathlib import Path
from common import UPOS, pool, test_subset, evaluate, train_ambiguity, save_json

MODELS = {
    "flan-t5-base": ("google/flan-t5-base", "t5"),
    "flan-t5-large": ("google/flan-t5-large", "t5"),
    "llama-1b": ("meta-llama/Llama-3.2-1B-Instruct", "chat"),
}
INSTRUCTION = (
    "Tag each word with its Universal Dependencies part-of-speech tag. "
    "Possible tags: " + " ".join(UPOS) + ". "
    "Keep the words exactly as given, in the same order, and write each word "
    "followed by / and its tag, separated by spaces.")


def tagged(s):
    return " ".join(f"{w}/{t}" for w, t in zip(s["words"], s["tags"]))


def t5_prompt(demos, words):
    p = INSTRUCTION + "\n\n"
    for d in demos:
        p += f"Sentence: {' '.join(d['words'])}\nTagged: {tagged(d)}\n\n"
    return p + f"Sentence: {' '.join(words)}\nTagged:"


def chat_messages(demos, words):
    msgs = [{"role": "system", "content": INSTRUCTION}]
    for d in demos:  # demonstrations as previous conversation turns
        msgs += [{"role": "user", "content": " ".join(d["words"])},
                 {"role": "assistant", "content": tagged(d)}]
    return msgs + [{"role": "user", "content": " ".join(words)}]


def parse(text, words):
    """Align 'word/TAG word/TAG ...' output to the gold words.
    Walks through the output; a predicted word is matched to the next expected
    word (or one up to 3 positions later, if the model skipped some).
    Unmatched gold words -> '_MISSING', unknown tags -> '_INVALID'."""
    text = text.strip().split("\n")[0]
    tags = ["_MISSING"] * len(words)
    j = 0
    for item in text.split():
        if j >= len(words):
            break
        if "/" not in item:
            continue
        w, t = item.rsplit("/", 1)
        t = t.strip(".,;:").upper()
        for look in range(j, min(j + 4, len(words))):
            if w == words[look] or w.lower() == words[look].lower():
                tags[look] = t if t in UPOS else "_INVALID"
                j = look + 1
                break
    return tags


class Tagger:
    def __init__(self, key):
        import torch
        from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                                  AutoModelForCausalLM)
        name, self.kind = MODELS[key]
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.kind == "t5":  # T5 is unstable in float16 -> keep float32
            self.model = AutoModelForSeq2SeqLM.from_pretrained(name).to("cuda")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float16).to("cuda")
            self.tok.pad_token = self.tok.eos_token
            self.tok.padding_side = "left"
        self.model.eval()

    def tag_batch(self, demos, batch):
        max_new = max(8 * len(s["words"]) for s in batch) + 20
        if self.kind == "t5":
            prompts = [t5_prompt(demos, s["words"]) for s in batch]
        else:
            prompts = [self.tok.apply_chat_template(chat_messages(demos, s["words"]),
                       tokenize=False, add_generation_prompt=True) for s in batch]
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       add_special_tokens=(self.kind == "t5")).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        if self.kind != "t5":
            out = out[:, enc["input_ids"].shape[1]:]  # keep only the new tokens
        return self.tok.batch_decode(out, skip_special_tokens=True)


def run(tagger, key, k, seed, batch_size=8):
    d = Path("runs_hf") / f"{key}_k{k}_s{seed}"
    if (d / "metrics.json").exists():
        print(f"[skip] {d} already done"); return
    test = test_subset()
    demos = pool(k, seed) if k > 0 else []
    t0, raw = time.time(), []
    for i in range(0, len(test), batch_size):
        raw += tagger.tag_batch(demos, test[i:i + batch_size])
    preds = [parse(r, s["words"]) for r, s in zip(raw, test)]
    r = evaluate(test, preds, pool_sents=demos or None, ambiguous=train_ambiguity())
    r["minutes"] = round((time.time() - t0) / 60, 1)
    save_json({"config": {"model": key, "k": k, "seed": seed}, **r}, d / "metrics.json")
    save_json([{"id": s["id"], "words": s["words"], "gold": s["tags"], "pred": p, "raw": x}
               for s, p, x in zip(test, preds, raw)], d / "predictions.json")
    print(f"{key} k={k} seed={seed}: acc={r['token_acc']:.3f} macroF1={r['macro_f1']:.3f} "
          f"missing={r['missing_rate']:.1%} ({r['minutes']} min)")


GRID = [(0, 0), (8, 0), (8, 1), (8, 2)]  # (k, seed): zero-shot + 3 few-shot draws

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="flan-t5-base")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    models = list(MODELS) if a.grid else [a.model]
    for key in models:
        tagger = Tagger(key)
        for k, seed in (GRID if a.grid else [(a.k, a.seed)]):
            run(tagger, key, k, seed, a.batch)
        del tagger
        import torch; torch.cuda.empty_cache()
