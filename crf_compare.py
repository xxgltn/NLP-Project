import json
from pathlib import Path
import analyze, baselines
from common import pool, test_subset

test = test_subset()
for k in (8, 32, 128):
    for s in (0, 1, 2):
        d = Path(f"runs_crf/CRF_k{k}_s{s}")
        if (d / "predictions.json").exists():
            continue
        m = baselines.crf_train(pool(k, s))
        preds = baselines.crf_predict(m, test)
        d.mkdir(parents=True, exist_ok=True)
        (d / "predictions.json").write_text(json.dumps(
            [{"id": x["id"], "words": x["words"], "gold": x["tags"],
              "raw": " ".join(f"{w}/{t}" for w, t in zip(x["words"], p))}
             for x, p in zip(test, preds)]))
        print("wrote", d)

analyze.SYSTEMS["CRF"] = ("runs_crf", "CRF_k{k}_s{s}")
for k in (8, 32, 128):
    analyze.compare("llama-1b (prompt tuning)", "CRF", k)
for k in (8, 32):
    analyze.compare("llama-1b (prompting)", "CRF", k)
