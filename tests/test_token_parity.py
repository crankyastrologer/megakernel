"""Correctness gate: every decode mode must reproduce greedy reference tokens.

Run:  .venv/bin/python tests/test_token_parity.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from bench import harness  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT_LEN = 128
STEPS = 12
MODES = ["eager", "static", "graph", "compile"]


def main():
    """Check greedy token parity for eager/static/graph/compile on SmolLM2-135M.

    Returns:
        int: 0 if every mode matches the reference prefix, 1 otherwise.
    """
    model, tok = harness.load_model(MODEL)
    ids = harness.make_prompt_ids(tok, PROMPT_LEN)
    ref = harness.reference_tokens(model, ids, STEPS + 4)
    failures = []
    for mode in MODES:
        try:
            res = harness.run_mode(model, ids, mode, STEPS)
            n = min(len(res["tokens"]), len(ref))
            mism = next((i for i in range(n) if res["tokens"][i] != ref[i]), None)
            ok = mism is None
            print(f"{mode:8s} tokens={len(res['tokens']):3d} parity={ok} first_mismatch={mism}")
            if not ok:
                failures.append(mode)
        except Exception as exc:
            print(f"{mode:8s} ERROR {exc!r}")
            failures.append(mode)
    if failures:
        print(f"FAIL: {failures}")
        return 1
    print("PASS: all modes match greedy reference")
    return 0


if __name__ == "__main__":
    torch.cuda.init()
    raise SystemExit(main())
