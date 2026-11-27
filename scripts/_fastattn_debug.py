import os
import sys

import torch
from transformers import StaticCache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench import harness  # noqa: E402
from bench.fastattn import fill_cache_sentinels  # noqa: E402

MID = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = None
model, tok = harness.load_model(MID)
ids = harness.make_prompt_ids(tok, 128)
maxlen = 128 + 8 + 8

with torch.inference_mode():
    cache = StaticCache(config=model.config, max_cache_len=maxlen)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    first = out.logits[:, -1].argmax(-1, keepdim=True)
    layer = cache.layers[0]
    cum = int(layer.cumulative_length.item())
    print("fast_attn env:", os.environ.get("FAST_ATTN", "1"), "cum:", cum)
    print("keys shape:", tuple(layer.keys.shape))
    print("future key sample before fill:", layer.keys[0, 0, cum, :4].tolist())
    fill_cache_sentinels(cache)
    print("future key sample after fill :", layer.keys[0, 0, cum, :4].tolist())
    print("sentinel count in future:", int((layer.keys[:, :, cum:, :] == -1e4).sum()), "/", layer.keys[:, :, cum:, :].numel())

    cur = first
    toks = [cur.item()]
    for i in range(7):
        o = model(input_ids=cur, past_key_values=cache, use_cache=True)
        cur = o.logits[:, -1].argmax(-1, keepdim=True)
        toks.append(cur.item())
    torch.cuda.synchronize()
    print("tokens:", toks)
