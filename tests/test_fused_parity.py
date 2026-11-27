"""End-to-end parity gate for the fused graph: 50 greedy tokens at prompt 128."""

import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from bench import harness  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT_LEN = 128
STEPS = 50


def main():
    """Compare graph_fused tokens to manual eager and report the median."""
    model, tok = harness.load_model(MODEL)
    ids = harness.make_prompt_ids(tok, PROMPT_LEN)
    ref = harness.run_mode(model, ids, "eager", STEPS)["tokens"]
    res = harness.run_mode(model, ids, "graph_fused", STEPS)
    toks = res["tokens"]
    n = min(len(toks), len(ref))
    mism = next((i for i in range(n) if toks[i] != ref[i]), None)
    med = statistics.median(res["times"][10:])
    print(f"tokens {len(toks)} vs ref {len(ref)}  parity {mism is None}  first_diff {mism}")
    print(f"median {med:.3f} ms/token  {100 * 0.269025 / med / 263.5 * 1000:.1f}% roofline")
    print("PASS" if mism is None else "FAIL")
    return 0 if mism is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
