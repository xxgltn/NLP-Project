import llm_prompt as L

tagger = L.Tagger("llama-1b")
for k, bs in [(32, 4), (128, 2)]:
    for s in [0, 1, 2]:
        L.run(tagger, "llama-1b", k, s, bs)
