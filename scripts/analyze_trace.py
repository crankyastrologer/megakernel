"""Summarize a torch.profiler chrome trace from bench/breakdown.py.

Prints per-kernel time/count, then the projection split:
  - HF trace: 210 `gemvx::kernel` events/step assigned by q/k/v/o/gate/up/down order
  - fused trace: `gemv_res` split into o and down by its alternating position

Usage:
  .venv/bin/python scripts/analyze_trace.py results/trace_breakdown_graph_128.json --steps 8
  .venv/bin/python scripts/analyze_trace.py results/trace_breakdown_graph_fused_128.json --steps 5
"""

import argparse
import collections
import json


def main():
    """Parse a chrome trace and print per-kernel and per-projection totals."""
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--steps", type=int, required=True, help="profile steps in the trace")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    events = json.load(open(args.trace)).get("traceEvents", [])
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    kernels.sort(key=lambda e: e["ts"])
    steps = args.steps
    print(f"{args.trace}: {len(kernels)} kernels, {len(kernels)/steps:.1f}/step")

    agg = collections.defaultdict(lambda: [0, 0.0])
    for e in kernels:
        name = e.get("name", "?")
        agg[name][0] += 1
        agg[name][1] += float(e.get("dur", 0.0))
    for name, (count, us) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: args.top]:
        print(f"{us/steps/1e3:9.4f} ms {count/steps:7.1f}  {name[:110]}")

    kind = "fused" if "graph_fused" in args.trace else "hf"
    print(f"\n--- {kind} projection split ---")
    if kind == "hf":
        g = [e for e in kernels if "gemvx::kernel" in e.get("name", "")]
        if not g or len(g) % steps != 0:
            print("gemvx events not found / not divisible by steps")
            return
        per_step = len(g) // steps
        if per_step != 210:
            print(f"warning: {per_step} gemvx/step, expected 210")
        stages = ["q", "k", "v", "o", "gate", "up", "down"]
        sums = collections.defaultdict(float)
        for i, e in enumerate(g):
            sums[stages[i % 7]] += e["dur"]
        for s in stages:
            print(f"{s:4s} {sums[s]/steps/1e3:9.4f} ms/step")
        print(f"qkv  {(sums['q']+sums['k']+sums['v'])/steps/1e3:9.4f} ms/step")
        print(f"gate+up {(sums['gate']+sums['up'])/steps/1e3:9.4f} ms/step")
    else:
        def total(name):
            return sum(e["dur"] for e in kernels if e["name"] == name) / steps / 1e3

        g = [e for e in kernels if e["name"] == "gemv_res"]
        o = sum(e["dur"] for i, e in enumerate(g) if i % 2 == 0) / steps / 1e3
        d = sum(e["dur"] for i, e in enumerate(g) if i % 2 == 1) / steps / 1e3
        print(f"qkv_gemv        {total('qkv_gemv'):9.4f} ms/step")
        print(f"gemv_res o      {o:9.4f} ms/step")
        print(f"gemv_res down   {d:9.4f} ms/step")
        print(f"gate_up_swiglu  {total('rms_gate_up_swiglu'):9.4f} ms/step")
        print(f"rope_kv_append  {total('rope_kv_append'):9.4f} ms/step")
        print(f"attn_split      {total('_decode_attn_split'):9.4f} ms/step")
        print(f"attn_reduce     {total('_decode_attn_reduce'):9.4f} ms/step")


if __name__ == "__main__":
    main()
