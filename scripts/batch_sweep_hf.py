"""Batch-size sweep for the HF baseline: eager and CUDA graph.

For B identical requests, reports per-step latency (each request gets one token
per step), aggregate throughput, and both roofline percentages.

Usage:
  .venv/bin/python scripts/batch_sweep_hf.py
  .venv/bin/python scripts/batch_sweep_hf.py --batches 1,2,4,8 \
      --prompt-lengths 128,512,2048 --modes eager,graph --json results/batch_hf.json
"""

import argparse
import gc
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from transformers.cache_utils import StaticCache  # noqa: E402

from bench import harness, roofline  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def timed(fn, steps):
    """Run ``fn`` ``steps`` times with CUDA events and return per-step ms."""
    evs = []
    for _ in range(steps):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fn()
        t1.record()
        evs.append((t0, t1))
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in evs]


@torch.inference_mode()
def run_eager(model, ids, steps):
    """Decode with a DynamicCache and no graph."""
    out = model(input_ids=ids, use_cache=True)
    state = {"past": out.past_key_values, "cur": out.logits[:, -1].argmax(-1, keepdim=True)}
    harness.burn()

    def step():
        o = model(input_ids=state["cur"], past_key_values=state["past"], use_cache=True)
        state["past"] = o.past_key_values
        state["cur"] = o.logits[:, -1].argmax(-1, keepdim=True)

    times = timed(step, steps)
    del state, out, step
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return times


@torch.inference_mode()
def run_graph(model, ids, steps):
    """Decode with a StaticCache and a captured graph."""
    plen = ids.shape[1]
    cache = StaticCache(config=model.config, max_cache_len=plen + steps + 16)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    first = out.logits[:, -1].argmax(-1, keepdim=True).clone()
    static_ids = first.clone()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o = model(input_ids=static_ids, past_key_values=cache, use_cache=True)
        nxt = o.logits[:, -1].argmax(-1, keepdim=True)

    g.replay()
    torch.cuda.synchronize()
    static_ids.copy_(nxt)
    harness.burn()

    def step():
        static_ids.copy_(nxt)
        g.replay()

    times = timed(step, steps)
    del step, g, cache, static_ids, nxt, out
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return times


def main():
    """Run the batch sweep and optionally write JSON records."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--prompt-lengths", default="128,512,2048")
    ap.add_argument("--modes", default="eager,graph")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    batches = [int(x) for x in args.batches.split(",")]
    plens = [int(x) for x in args.prompt_lengths.split(",")]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    model, tok = harness.load_model(MODEL)
    w_bytes = roofline.weight_bytes(model)
    peak = roofline.load_peak_gbs()
    base = harness.make_prompt_ids(tok, plens[0])
    rows = []

    print(
        f"{'plen':>5} {'B':>2} {'mode':6s} {'step ms':>9} {'per-req':>8} "
        f"{'tok/s':>9} {'w%':>6} {'tot%':>6}"
    )
    for plen in plens:
        base = harness.make_prompt_ids(tok, plen)
        for batch in batches:
            ids = base.repeat(batch, 1)
            for mode in modes:
                try:
                    times = (
                        run_eager(model, ids, args.steps)
                        if mode == "eager"
                        else run_graph(model, ids, args.steps)
                    )
                    med = statistics.median(times[args.warmup:])
                    sec = med / 1e3
                    kv = roofline.kv_bytes_per_token(
                        model.config, harness.DTYPE, seq_len=plen + args.steps // 2
                    )
                    w_gbs = w_bytes / sec / 1e9
                    t_gbs = (w_bytes + batch * kv) / sec / 1e9
                    rec = {
                        "model": MODEL,
                        "prompt_len": plen,
                        "batch": batch,
                        "mode": mode,
                        "step_ms": round(med, 4),
                        "per_request_ms": round(med, 4),
                        "agg_tok_per_s": round(batch * 1e3 / med, 2),
                        "pct_weight_step": round(100 * w_gbs / peak, 2),
                        "pct_total_step": round(100 * t_gbs / peak, 2),
                        "kv_gb_per_request": round(kv / 1e9, 6),
                    }
                    rows.append(rec)
                    print(
                        f"{plen:5d} {batch:2d} {mode:6s} {med:9.3f} {med:8.3f} "
                        f"{batch * 1e3 / med:9.1f} "
                        f"{100 * w_gbs / peak:6.1f} {100 * t_gbs / peak:6.1f}"
                    )
                except Exception as exc:
                    print(f"{plen:5d} {batch:2d} {mode:6s}  FAILED {exc!r}")
                gc.collect()
                torch.cuda.empty_cache()

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
