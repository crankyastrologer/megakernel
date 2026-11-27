"""Parity test for rms_gate_up_swiglu and bump_seq."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from kernels.triton.decode import bump_seq, rms_gate_up_swiglu  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
K, N = 576, 1536
BN, BK, NW = 64, 64, 8
TOL = 0.1


def main():
    """Check gate/up output for P=1 and P=18, and bump_seq."""
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    layer = model.model.layers[0]
    h = torch.randn(K, device="cuda", dtype=torch.bfloat16) * 0.5
    nw = layer.post_attention_layernorm.weight
    wg = layer.mlp.gate_proj.weight
    wu = layer.mlp.up_proj.weight
    eps = model.config.rms_norm_eps

    xn = F.rms_norm(h, (K,), nw, eps)
    ref = F.silu(F.linear(xn, wg)) * F.linear(xn, wu)

    ok = True
    for p in (1, 18):
        if p == 1:
            ss = torch.tensor([(h.float() ** 2).sum()], device="cuda", dtype=torch.float32)
            pb = 1
        else:
            chunks = [h[i * 32:(i + 1) * 32] for i in range(18)]
            ss = torch.stack([(c.float() ** 2).sum() for c in chunks]).contiguous()
            pb = 32
        y = torch.empty(N, device="cuda", dtype=torch.bfloat16)
        rms_gate_up_swiglu[(N // BN,)](
            h, nw, ss, wg, wu, y, eps,
            K=K, N=N, P=p, PB=pb, BN=BN, BK=BK, num_warps=NW,
        )
        torch.cuda.synchronize()
        d = (y - ref).abs().max().item()
        print(f"P={p:2d} maxdiff {d:.6f}")
        ok = ok and d < TOL

    seq = torch.tensor([5], device="cuda", dtype=torch.int64)
    bump_seq[(1,)](seq)
    torch.cuda.synchronize()
    print("bump_seq:", seq.item(), "(expect 6)")
    ok = ok and seq.item() == 6

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
