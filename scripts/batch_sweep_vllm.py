"""Batch-size sweep for the vLLM baseline (run in .venv-vllm).

For B identical requests, reports TPOT (per-request ms/token), aggregate decode
throughput, and step-level weight/total roofline percentages.

Usage (via wrapper):
  ./scripts/run_batch_vllm.sh --batches 1,2,4,8 --prompt-lengths 128,512,2048
"""

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from bench.prompts import make_prompt_ids  # noqa: E402
from bench.roofline import kv_bytes_per_token  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
WEIGHT_GB = 0.269025  # SmolLM2-135M bf16
PEAK_GBS = 263.5


def tpot_ms(metrics, n_tokens):
    """Return per-output-token ms from vLLM engine timestamps, or None."""
    if metrics is None or n_tokens < 2:
        return None
    a = getattr(metrics, "first_token_ts", None)
    b = getattr(metrics, "last_token_ts", None)
    if a is not None and b is not None and b > a:
        return (b - a) / (n_tokens - 1) * 1e3
    x = getattr(metrics, "mean_time_per_output_token", None)
    return x * 1e3 if x is not None else None


def main():
    """Run the vLLM batch sweep and optionally write JSON records."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--prompt-lengths", default="128,512,2048")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    batches = [int(x) for x in args.batches.split(",")]
    plens = [int(x) for x in args.prompt_lengths.split(",")]
    maxlen = max(plens) + args.steps + 16
    cfg = AutoConfig.from_pretrained(MODEL)

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=maxlen,
        gpu_memory_utilization=0.8,
        enforce_eager=False,
        disable_log_stats=False,
        seed=0,
    )
    tok = llm.get_tokenizer()
    params = SamplingParams(
        temperature=0.0, max_tokens=args.steps, ignore_eos=True, detokenize=False, seed=0
    )
    rows = []

    print(
        f"{'plen':>5} {'B':>2} {'tpot ms':>9} {'per-req':>8} "
        f"{'tok/s':>9} {'w%':>6} {'tot%':>6} {'wall tok/s':>10}"
    )
    for plen in plens:
        ids = make_prompt_ids(tok, plen)[0].tolist()
        prompts = [TokensPrompt(prompt_token_ids=ids) for _ in range(max(batches))]
        llm.generate(
            prompts[:1],
            SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
            use_tqdm=False,
        )
        for batch in batches:
            use = prompts[:batch]
            tpots = []
            wall_tokens = 0
            walls = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                outs = llm.generate(use, params, use_tqdm=False)
                walls.append(time.perf_counter() - t0)
                for o in outs:
                    n = len(o.outputs[0].token_ids)
                    wall_tokens += n
                    x = tpot_ms(o.metrics, n)
                    if x is not None:
                        tpots.append(x)
            med = statistics.median(tpots) if tpots else None
            wall_tok_s = wall_tokens / sum(walls) if walls else None
            if med is None:
                print(f"{plen:5d} {batch:2d}  TPOT unavailable")
                continue
            kv = kv_bytes_per_token(cfg, torch.bfloat16, seq_len=plen + args.steps // 2) / 1e9
            w_gbs = WEIGHT_GB / (med / 1e3)
            t_gbs = (WEIGHT_GB + batch * kv) / (med / 1e3)
            rec = {
                "model": MODEL,
                "prompt_len": plen,
                "batch": batch,
                "mode": "vllm",
                "tpot_ms": round(med, 4),
                "per_request_ms": round(med, 4),
                "agg_tok_per_s": round(batch * 1e3 / med, 2),
                "wall_tok_per_s": round(wall_tok_s, 2) if wall_tok_s else None,
                "pct_weight_step": round(100 * w_gbs / PEAK_GBS, 2),
                "pct_total_step": round(100 * t_gbs / PEAK_GBS, 2),
                "kv_gb_per_request": round(kv, 6),
                "reps": args.reps,
            }
            rows.append(rec)
            print(
                f"{plen:5d} {batch:2d} {med:9.3f} {med:8.3f} "
                f"{batch * 1e3 / med:9.1f} "
                f"{100 * w_gbs / PEAK_GBS:6.1f} {100 * t_gbs / PEAK_GBS:6.1f} "
                f"{wall_tok_s:10.1f}"
            )

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
