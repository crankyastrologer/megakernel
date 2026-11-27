"""Parity test for gemv_res: GEMV + residual + next-norm partial sums."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from kernels.triton.decode import gemv_res  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
BN, BK = 32, 64


def main():
    """Check y against torch and partials against sum(y_bf16^2)."""
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    layer = model.model.layers[0]

    cases = [
        ("o", 576, layer.self_attn.o_proj.weight),
        ("down", 1536, layer.mlp.down_proj.weight),
    ]
    ok = True
    for name, k, w in cases:
        n = 576
        x = torch.randn(k, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        y = torch.empty(n, device="cuda", dtype=torch.bfloat16)
        ss = torch.empty(n // BN, device="cuda", dtype=torch.float32)

        gemv_res[(n // BN,)](x, w, r, y, ss, K=k, N=n, BN=BN, BK=BK, num_warps=4)
        torch.cuda.synchronize()

        ref = (r.float() + F.linear(x, w).float()).to(torch.bfloat16)
        yd = (y - ref).abs().max().item()
        sd = (ss.sum() - (y.float() ** 2).sum()).abs().item()
        print(f"{name:4s} y maxdiff {yd:.6f}  ss diff {sd:.6f}")
        ok = ok and yd < 0.5 and sd < 1e-2

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
