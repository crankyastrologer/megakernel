import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(0)
dev = "cuda"
dtype = torch.bfloat16
H, KVH, D = 9, 3, 64


def bench(fn, iters=200, warmup=20):
    """Time a GPU callable and return milliseconds per call.

    Args:
        fn (Callable[[], None]): Work to enqueue on the current stream.
        iters (int): Timed iterations. Defaults to 200.
        warmup (int): Untimed iterations. Defaults to 20.

    Returns:
        float: Milliseconds per call.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


for L in [128, 512, 2048]:
    q = torch.randn(1, H, 1, D, device=dev, dtype=dtype)
    k = torch.randn(1, KVH, L, D, device=dev, dtype=dtype)
    v = torch.randn(1, KVH, L, D, device=dev, dtype=dtype)
    k9 = k.repeat_interleave(3, dim=1)
    v9 = v.repeat_interleave(3, dim=1)
    mask = torch.zeros(1, 1, 1, L, device=dev, dtype=torch.bool)

    print(f"\nL={L}")
    # no mask, GQA native
    t = bench(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False, enable_gqa=True))
    print(f"  no-mask enable_gqa: {t*1e3:8.1f} us")
    # no mask, expanded kv
    t = bench(lambda: F.scaled_dot_product_attention(q, k9, v9, attn_mask=None, is_causal=False))
    print(f"  no-mask expanded : {t*1e3:8.1f} us")
    # with mask
    t = bench(lambda: F.scaled_dot_product_attention(q, k9, v9, attn_mask=mask, is_causal=False))
    print(f"  4d-mask expanded : {t*1e3:8.1f} us")
    for name, backend in [("FLASH", SDPBackend.FLASH_ATTENTION), ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION), ("MATH", SDPBackend.MATH)]:
        try:
            with sdpa_kernel(backend):
                t = bench(lambda: F.scaled_dot_product_attention(q, k9, v9, attn_mask=None, is_causal=False))
            print(f"  forced {name:9s}: {t*1e3:8.1f} us")
        except Exception as e:
            print(f"  forced {name:9s}: FAILED {type(e).__name__}: {str(e)[:80]}")
