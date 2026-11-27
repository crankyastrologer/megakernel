import torch

try:
    import flash_attn

    print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn: NOT AVAILABLE", type(e).__name__, e)

from transformers import AutoModelForCausalLM

mid = "HuggingFaceTB/SmolLM2-135M-Instruct"
for impl in [None, "flash_attention_2", "sdpa", "eager"]:
    try:
        kw = {} if impl is None else {"attn_implementation": impl}
        m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda", **kw)
        print(f"requested={impl!r} -> actual={m.config._attn_implementation}")
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"requested={impl!r} -> FAILED {type(e).__name__}: {str(e)[:200]}")
