# Few-Shot POS Tagging with Prompt-Tuned Language Models

Code and results for a bachelor's/master's thesis comparing **in-context prompting**, **soft prompt tuning**
and **traditional taggers** for few-shot Universal Dependencies POS tagging, under an identical
labelled-data budget.

## Research question

Given only k labelled sentences (k = 8, 32, 128), how well can small pretrained language models tag
UD parts of speech? Does training a soft prompt beat putting the same sentences in the prompt, and
which errors does each approach fix?

## Main results (token accuracy %, 300-sentence test subset, mean of 3 pool draws)

| System | k = 8 | k = 32 | k = 128 |
|---|---|---|---|
| Most-frequent tag | 47.2 | 58.8 | 69.3 |
| CRF | 56.6 | 68.8 | 80.0 |
| Flan-T5-base, prompting | 0.0 | — | — |
| Flan-T5-large, prompting | 1.0 | — | — |
| Llama-3.2-1B, prompting | 58.5 | 65.4 | not runnable (GPU memory) |
| Flan-T5-base + soft prompt | 30.2 | 19.8 | 20.8 |
| Flan-T5-large + soft prompt | 31.7 | 26.2 | 31.9 |
| **Llama-3.2-1B + soft prompt** | 50.3 | **75.0** | **82.0** |
| CRF, full training set (12,544 sentences, reference) | — | — | 94.2 |

Soft prompt tuning crosses over from significantly worse than prompting at k = 8 (−8.2 points) to
significantly better at k = 32 (+9.6); paired bootstrap over test sentences, 10,000 resamples.

## Setup

Data: UD_English-EWT, pinned release **r2.18**, included in `data/`.
Test set: fixed random subset of 300 test sentences (3,855 tokens, seed 0).
Pools: nested random subsets of the train split, k ∈ {8, 32, 128}, seeds {0, 1, 2}.

```bash
pip install -r requirements.txt            # plus torch, transformers, peft for the LLM scripts
huggingface-cli login                      # Llama 3.2 is a gated model
```

The LLM experiments need a GPU (developed on a free Colab T4, 16 GB).

## Reproducing the experiments

```bash
python baselines.py                        # MFT + CRF baselines (CPU, ~1 min)
python llm_prompt.py --grid                # zero-shot + few-shot prompting, all models
python rescore.py                          # add lenient scores to the prompting runs
python pt_train.py --grid                  # soft prompt tuning, 3 models x 3 k x 3 seeds
python summarize_all.py                    # combined results table
python crf_compare.py                      # CRF predictions + significance tests
python analyze.py compare_all              # paired-bootstrap comparisons
python analyze.py errors runs_pt/llama-1b_k32_s1_lr0.01   # confusions + 50-error sample
```

Finished runs are skipped automatically, so an interrupted grid can simply be restarted.

## Files

| File | Purpose |
|---|---|
| `common.py` | Data loading, test subset, pool sampling, evaluation metrics |
| `baselines.py` | Most-frequent-tag and CRF baselines |
| `llm_prompt.py` | Zero-shot and few-shot prompting of Flan-T5 and Llama (no training) |
| `pt_train.py` | Soft prompt tuning via PEFT; model frozen, 50 prompt vectors trained |
| `rescore.py` | Lenient re-parsing of saved model outputs |
| `summarize.py`, `summarize_all.py` | Results tables |
| `crf_compare.py` | CRF predictions in LLM format + significance tests |
| `analyze.py` | Paired bootstrap, confusion analysis, error sampling |
| `claude_run.py` | Unused. Early API-based approach, kept for the record |

| Folder | Contents |
|---|---|
| `runs_hf/` | Prompting runs (raw outputs, predictions, metrics) |
| `runs_pt/` | Prompt-tuning runs at the final learning rates |
| `runs_pilot/` | Learning-rate pilot runs |
| `runs_llama_lr0.03/` | Llama runs at the rejected learning rate, kept as evidence of instability |
| `runs_crf/` | CRF predictions for significance testing |
| `results/` | Baseline metrics and summary tables |

## Hyperparameters

Soft prompt tuning: 50 virtual tokens, text initialisation, 400 steps, batch size 4, AdamW,
gradient clipping 1.0, float32. Learning rate **0.3 for Flan-T5** and **0.01 for Llama**, chosen in
pilot runs on training loss only (0.03 diverged for Llama in 1 of 6 runs). Llama uses token-weighted
gradient accumulation and gradient checkpointing to fit in 16 GB; both were verified to leave
gradients unchanged.

## Evaluation protocol

- Every system sees only its k-sentence pool. Nothing is tuned on the test subset.
- Two parsers are reported: **strict** (exact `word/TAG` format, first line only) and **lenient**
  (three fixed tolerances). The lenient rules were fixed before any lenient score was computed.
- Unmatched tokens count as errors, not as skipped items.
- Raw model outputs are stored, so re-scoring never re-runs a model.

## Notes

Results were produced on a Colab T4. Prompt tuning is sensitive to the learning rate and to run-to-run
GPU nondeterminism (up to ~3.7 points for identical configurations), so small differences should not
be over-interpreted.
