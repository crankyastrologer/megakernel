import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench import harness  # noqa: E402

MID = "Qwen/Qwen2.5-1.5B-Instruct"
PLEN = 128
STEPS = 60


def firstdiff(a, b):
    """Return the first differing index and nearby tokens for two id lists.

    Args:
        a (list[int]): First token sequence.
        b (list[int]): Second token sequence.

    Returns:
        tuple[int | None, list[int], list[int]]: ``(index, a_window, b_window)``;
        index is ``None`` if the overlapping prefix matches.
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, a[max(0, i - 3):i + 4], b[max(0, i - 3):i + 4]
    return None, a[:6], b[:6]


def main():
    """Compare Qwen2.5-1.5B tokens from generate, eager, graph, and vLLM."""
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda").eval()
    ids = harness.make_prompt_ids(tok, PLEN)
    ref = harness.reference_tokens(model, ids, STEPS + 4)
    e = harness.run_mode(model, ids, "eager", STEPS)["tokens"]
    g = harness.run_mode(model, ids, "graph", STEPS)["tokens"]

    with open(os.path.join(ROOT, "results", "vllm.json")) as f:
        v = json.load(f)["qwen2.5-1.5b|128"]["tokens"]

    print("lengths: ref", len(ref), "eager", len(e), "graph", len(g), "vllm", len(v))
    for name_a, a, name_b, b in [
        ("ref", ref, "eager", e),
        ("ref", ref, "graph", g),
        ("ref", ref, "vllm", v),
        ("eager", e, "graph", g),
        ("eager", e, "vllm", v),
        ("graph", g, "vllm", v),
    ]:
        i, aa, bb = firstdiff(a, b)
        print(f"{name_a} vs {name_b}: firstdiff={i}")
        if i is not None:
            print("   ", name_a, aa)
            print("   ", name_b, bb)

    print("\nref head:  ", ref[:12])
    print("eager head:", e[:12])
    print("graph head:", g[:12])
    print("vllm head: ", v[:12])


if __name__ == "__main__":
    main()
