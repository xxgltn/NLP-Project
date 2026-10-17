# Few-shot UPOS tagging — Week 1 pipeline

Data: UD_English-EWT, pinned release r2.18 (already in `data/`). Test = fixed random subset of
300 test sentences (3,855 tokens, seed 0). Pools = nested random train subsets, k ∈ {8, 32, 128}, seeds {0,1,2}.

    pip install -r requirements.txt
    python baselines.py                               # MFT + CRF (free, ~1 min) -> results/baselines.json
    export ANTHROPIC_API_KEY=...
    python claude_run.py --k 8 --dry-run              # inspect prompt + cost estimate (free)
    python claude_run.py --k 8 --limit 20 --sync      # smoke test, ~1 cent
    python claude_run.py --k 8 --seed 0 submit        # full run via Batch API (50% off)
    python claude_run.py --k 8 --seed 0 collect       # parse + evaluate -> runs/<name>/metrics.json
    python summarize.py > results/week1_table.md      # one results table

Rules: never tune anything on the test subset; systems only see their k-sentence pool;
raw API outputs are stored and never re-requested.
