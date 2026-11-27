"""Parity test for rope_kv_append.

Checks q/k against HF's apply_rotary_pos_emb, verifies k/v cache placement,
and asserts the position-0 identity case.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from transformers.models.llama import modeling_llama as ml  # noqa: E402

from kernels.triton.decode import rope_kv_append_launch  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
HQ, HKV, D = 9, 3, 64
L = 4096
TOL = 0.02


@torch.no_grad()
def build_tables(model, max_len):
    """Build bf16 cos/sin tables [max_len, D] from the model's rotary module."""
    dummy = torch.zeros(1, 1, model.config.hidden_size, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(max_len, device="cuda").unsqueeze(0)
    cos, sin = model.model.rotary_emb(dummy, position_ids=positions)
    return cos[0].contiguous(), sin[0].contiguous()


def main():
    """Run the rope parity checks and return a process exit code."""
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    cos, sin = build_tables(model, L)
    print("table", tuple(cos.shape), cos.dtype)

    failures = 0
    for pos in [0, 1, 127, 128, 2047]:
        q = torch.randn(HQ * D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(HKV * D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(HKV * D, device="cuda", dtype=torch.bfloat16)
        qkv = torch.cat([q, k, v]).contiguous()
        q0, k0, v0 = q.clone(), k.clone(), v.clone()

        kc = torch.zeros(HKV, L, D, device="cuda", dtype=torch.bfloat16)
        vc = torch.zeros_like(kc)
        seq = torch.tensor(pos + 1, device="cuda", dtype=torch.int64)

        rope_kv_append_launch(qkv, cos, sin, kc, vc, seq, L=L, num_warps=2)
        torch.cuda.synchronize()

        qr, kr = ml.apply_rotary_pos_emb(
            q0.view(1, HQ, 1, D),
            k0.view(1, HKV, 1, D),
            cos[pos].view(1, 1, D),
            sin[pos].view(1, 1, D),
        )
        qd = (qkv[: HQ * D].view(1, HQ, 1, D) - qr).abs().max().item()
        kd = (kc[:, pos, :].view(1, HKV, 1, D) - kr).abs().max().item()
        v_exact = torch.equal(vc[:, pos, :], v0.view(HKV, D))
        untouched = True if pos == 0 else kc[:, :pos, :].abs().max().item() == 0.0

        ok = qd < TOL and kd < TOL and v_exact and untouched
        if pos == 0:
            ok = (
                ok
                and torch.equal(qkv[: HQ * D], q0)
                and torch.equal(kc[:, 0, :], k0.view(HKV, D))
            )
        print(
            f"pos {pos:4d}  q {qd:.6f}  k {kd:.6f}"
            f"  v_exact {v_exact}  untouched {untouched}  ok {ok}"
        )
        failures += 0 if ok else 1

    print("PASS" if failures == 0 else f"FAIL ({failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
