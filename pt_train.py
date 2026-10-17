"""Soft PROMPT TUNING: learn a few prompt vectors, keep every model weight frozen.

Usage (Colab, GPU):
  python pt_train.py --model flan-t5-base --k 32 --seed 0     # one run
  python pt_train.py --grid                                   # all 27 runs, skips finished ones

Output: runs_pt/<model>_k<k>_s<seed>/ with metrics.json (strict), metrics_lenient.json,
predictions.json and train_log.json (training loss).  Hyperparameters are FIXED in advance
(see HP below) and must not be tuned on the test set.
"""
import argparse, json, os, random, re, time
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
from pathlib import Path
from common import UPOS, pool, test_subset, evaluate, train_ambiguity, save_json
from llm_prompt import parse as parse_strict

MODELS = {
    "flan-t5-base": ("google/flan-t5-base", "t5"),
    "flan-t5-large": ("google/flan-t5-large", "t5"),
    "llama-1b": ("meta-llama/Llama-3.2-1B-Instruct", "causal"),
}
HP = {"n_virtual": 50, "lr": 0.03, "steps": 400, "batch": 4, "init": "text"}
# learning rate per model family, chosen by training loss in pilot runs
LR = {"flan-t5-base": 0.3, "flan-t5-large": 0.3, "llama-1b": 0.01}
INIT_TEXT = "Tag each word with its Universal Dependencies part-of-speech tag, as word/TAG:"


def src(words):
    return "Sentence: " + " ".join(words) + "\nTagged:"


def tgt(s):
    return " ".join(f"{w}/{t}" for w, t in zip(s["words"], s["tags"]))


def parse_lenient(text, words):  # identical rules to rescore.py (fixed, do not change)
    text = re.sub(r"\s+/\s+", "/", text.replace("\n", " "))
    tags, j = ["_MISSING"] * len(words), 0
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


class PromptTuned:
    def __init__(self, key, seed, hp, name=None, kind=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
        from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
        name0, kind0 = MODELS.get(key, (None, None))
        self.name, self.kind, self.hp, self.torch = name or name0, kind or kind0, hp, torch
        torch.manual_seed(seed); random.seed(seed)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.name)
        if self.kind == "t5":
            base = AutoModelForSeq2SeqLM.from_pretrained(self.name, dtype=torch.float32)
            task = TaskType.SEQ_2_SEQ_LM
        else:  # float32 for stable gradients through the frozen model
            base = AutoModelForCausalLM.from_pretrained(self.name, dtype=torch.float32)
            task = TaskType.CAUSAL_LM
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.config.use_cache = False
        cfg = PromptTuningConfig(
            task_type=task, num_virtual_tokens=hp["n_virtual"],
            prompt_tuning_init=PromptTuningInit.TEXT if hp["init"] == "text" else PromptTuningInit.RANDOM,
            prompt_tuning_init_text=INIT_TEXT if hp["init"] == "text" else None,
            tokenizer_name_or_path=self.name,
            # T5: soft prompt on the encoder input only, as in Lester et al. (2021)
            num_transformer_submodules=1 if self.kind == "t5" else None)
        self.model = get_peft_model(base, cfg).to(self.dev)
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.model.parameters())
        self.params = {"trainable": n_train, "total": n_all}
        print(f"trainable parameters: {n_train:,} of {n_all:,} ({100 * n_train / n_all:.4f}%)")

    # ---------------------------------------------------------------- training
    def _batch(self, sents):
        T = self.torch
        if self.kind == "t5":
            enc = self.tok([src(s["words"]) for s in sents], return_tensors="pt",
                           padding=True, truncation=True, max_length=512)
            lab = self.tok([tgt(s) for s in sents], return_tensors="pt", padding=True,
                           truncation=True, max_length=512).input_ids
            lab[lab == self.tok.pad_token_id] = -100
            return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask, "labels": lab}
        rows = []
        for s in sents:
            p = self.tok(src(s["words"])).input_ids
            t = self.tok(" " + tgt(s), add_special_tokens=False).input_ids + [self.tok.eos_token_id]
            rows.append((p + t, [-100] * len(p) + t))
        L = max(len(r[0]) for r in rows)
        pad = self.tok.pad_token_id
        ids = T.tensor([r[0] + [pad] * (L - len(r[0])) for r in rows])
        att = T.tensor([[1] * len(r[0]) + [0] * (L - len(r[0])) for r in rows])
        lab = T.tensor([r[1] + [-100] * (L - len(r[1])) for r in rows])
        return {"input_ids": ids, "attention_mask": att, "labels": lab}

    def train(self, data):
        T, hp = self.torch, self.hp
        opt = T.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=hp["lr"])
        self.model.train()
        order, log, run = [], [], 0.0
        for step in range(1, hp["steps"] + 1):
            if len(order) < hp["batch"]:
                order += random.sample(range(len(data)), len(data))
            b = [data[order.pop()] for _ in range(min(hp["batch"], len(data)))]
            if self.kind == "t5":
                out = self.model(**{k: v.to(self.dev) for k, v in self._batch(b).items()})
                out.loss.backward()
                loss_val = out.loss.item()
            else:  # one sentence at a time to save GPU memory; token-weighted so the
                   # gradient is the same as for the full batch of 4 (gradient accumulation)
                parts = [self._batch([x]) for x in b]
                ns = [int((p["labels"][:, 1:] != -100).sum()) for p in parts]
                loss_val = 0.0
                for p, n in zip(parts, ns):
                    o = self.model(**{k: v.to(self.dev) for k, v in p.items()})
                    (o.loss * n / sum(ns)).backward()
                    loss_val += o.loss.item() * n / sum(ns)
            T.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            run += loss_val
            if step % 50 == 0:
                log.append({"step": step, "loss": round(run / 50, 4)})
                print(f"  step {step:4d}  train loss {run / 50:.4f}"); run = 0.0
        return log

    # ---------------------------------------------------------------- tagging
    def tag(self, sents, bs=16):
        T, out_text = self.torch, []
        self.model.eval()
        if self.kind != "t5":
            self.tok.padding_side = "left"
        for i in range(0, len(sents), bs):
            b = sents[i:i + bs]
            enc = self.tok([src(s["words"]) for s in b], return_tensors="pt", padding=True).to(self.dev)
            with T.no_grad():
                gen = self.model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                          max_new_tokens=max(8 * len(s["words"]) for s in b) + 20,
                                          do_sample=False, pad_token_id=self.tok.pad_token_id, use_cache=True)
            if self.kind != "t5" and gen.shape[1] > enc.input_ids.shape[1]:
                gen = gen[:, enc.input_ids.shape[1]:]
            out_text += self.tok.batch_decode(gen, skip_special_tokens=True)
        self.tok.padding_side = "right"
        return out_text


def run(key, k, seed, hp=HP, name=None, kind=None, test=None, outdir="runs_pt"):
    tagstr = "" if hp == HP else "_" + "_".join(f"{a}{b}" for a, b in hp.items() if HP[a] != b)
    d = Path(outdir) / f"{key}_k{k}_s{seed}{tagstr}"
    if (d / "metrics_lenient.json").exists():
        print(f"[skip] {d} already done"); return
    test = test or test_subset()
    data = pool(k, seed)
    t0 = time.time()
    m = PromptTuned(key, seed, hp, name, kind)
    log = m.train(data)
    t1 = time.time()
    raw = m.tag(test)
    amb = train_ambiguity()
    cfg = {"model": key, "k": k, "seed": seed, "method": "prompt_tuning", **hp, **m.params}
    res = {}
    for label, parser in (("strict", parse_strict), ("lenient", parse_lenient)):
        preds = [parser(r, s["words"]) for r, s in zip(raw, test)]
        res[label] = evaluate(test, preds, pool_sents=data, ambiguous=amb)
    res["strict"]["train_minutes"] = res["lenient"]["train_minutes"] = round((t1 - t0) / 60, 1)
    save_json({"config": cfg, **res["strict"]}, d / "metrics.json")
    save_json({"config": cfg, **res["lenient"]}, d / "metrics_lenient.json")
    save_json(log, d / "train_log.json")
    save_json([{"id": s["id"], "words": s["words"], "gold": s["tags"], "raw": r}
               for s, r in zip(test, raw)], d / "predictions.json")
    print(f"{key} k={k} seed={seed}: strict acc={res['strict']['token_acc']:.3f}  "
          f"lenient acc={res['lenient']['token_acc']:.3f}  missing={res['lenient']['missing_rate']:.1%}  "
          f"({round((time.time() - t0) / 60, 1)} min)")
    del m
    import gc, torch; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="flan-t5-base")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    if a.grid:  # fastest model first, so results arrive early
        for key in ["flan-t5-base", "llama-1b", "flan-t5-large"]:
            for k in [8, 32, 128]:
                for seed in [0, 1, 2]:
                    run(key, k, seed, hp=dict(HP, lr=LR[key]))
    else:
        run(a.model, a.k, a.seed)
