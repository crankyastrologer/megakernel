"""Parity test for norm_stats + qkv_gemv, including multi-partial input."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from kernels.triton.decode import norm_stats, qkv_gemv  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
TOL = 0.02
BN, BK, NW = 16, 64, 8


def main():
    """Check P=1 and P=4 partial sums against a torch reference."""
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    layer = model.model.layers[0]
    h = model.model.embed_tokens(torch.tensor([[441]], device="cuda"))[0, 0].contiguous()
    nw = layer.input_layernorm.weight
    w = torch.cat(
        [layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight],
        dim=0,
    ).contiguous()
    eps = model.config.rms_norm_eps
    ref = F.linear(F.rms_norm(h, (576,), nw, eps), w)

    ss = torch.empty(1, device="cuda", dtype=torch.float32)
    norm_stats[(1,)](h, ss, K=576, BLOCK=1024, num_warps=4)
    y1 = torch.empty(960, device="cuda", dtype=torch.bfloat16)
    qkv_gemv[(960 // BN,)](h, nw, ss, w, y1, eps, K=576, N=960,
                           P=1, PB=1, BN=BN, BK=BK, num_warps=NW)

    chunks = [h[i * 144:(i + 1) * 144] for i in range(4)]
    ss4 = torch.stack([torch.sum(c.float() ** 2) for c in chunks]).contiguous()
    y4 = torch.empty(960, device="cuda", dtype=torch.bfloat16)
    qkv_gemv[(960 // BN,)](h, nw, ss4, w, y4, eps, K=576, N=960,
                           P=4, PB=4, BN=BN, BK=BK, num_warps=NW)

    torch.cuda.synchronize()
    d1 = (y1 - ref).abs().max().item()
    d4 = (y4 - ref).abs().max().item()
    print(f"P=1 maxdiff {d1:.6f}")
    print(f"P=4 maxdiff {d4:.6f}")
    print("PASS" if d1 < TOL and d4 < TOL else "FAIL")
    return 0 if d1 < TOL and d4 < TOL else 1


if __name__ == "__main__":
    raise SystemExit(main())
