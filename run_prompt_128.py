import sys
import llm_prompt as L
from torch.nn.attention import SDPBackend, sdpa_kernel

tagger = L.Tagger("llama-1b")
orig = tagger.tag_batch
tagger.tag_batch = lambda d, b: sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION).__enter__() or orig(d, b)

for seed in [int(x) for x in sys.argv[1:]]:
    L.run(tagger, "llama-1b", 128, seed, 1)
