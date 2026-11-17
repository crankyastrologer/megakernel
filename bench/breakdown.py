"""Decode time breakdown: weight streaming vs launch/host vs sampling vs other.

Profiles a harness decoder and categorizes GPU kernels from the chrome trace.
"""

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from bench import harness, roofline  # noqa: E402

RESULTS = os.path.join(ROOT, "results")


def category(name):
    """Map a CUDA kernel name to a coarse timing category.

    Args:
        name (str): Kernel name from the chrome trace.

    Returns:
        str: One of ``attention``, ``gemm_weights``, ``sampling``, ``memcpy``,
        or ``elementwise_other``.
    """
    n = name.lower()
    if "flash" in n or "attention" in n or "attn" in n or "fmha" in n or "sdpa" in n:
        return "attention"
    if "gemm" in n or "sgemm" in n or "cutlass" in n or "cublas" in n or "gemv" in n or "wmma" in n or "ampere_" in n:
        return "gemm_weights"
    if "argmax" in n or "topk" in n or "sort" in n:
        return "sampling"
    if "memcpy" in n or "memset" in n:
        return "memcpy"
    return "elementwise_other"


def main():
    """Profile a decoder and write a kernel-category breakdown JSON.

    Builds the requested decode mode, warms up, burns clocks, times a window
    with CUDA events, then profiles a second window and categorizes kernels
    from the chrome trace.

    CLI flags: ``--model``, ``--prompt-len``, ``--mode``, ``--warmup``,
    ``--steps``, ``--profile-steps``, ``--attn-impl``.

    Returns:
        None. Writes ``results/breakdown_<mode>_<prompt_len>.json`` and the
        matching ``trace_breakdown_*`` trace.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--mode", default="graph")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--profile-steps", type=int, default=8)
    ap.add_argument("--attn-impl", default=None)
    args = ap.parse_args()

    import yaml

    with open(os.path.join(ROOT, "models.yaml")) as f:
        cfg = yaml.safe_load(f)
    mid = cfg["models"][args.model]["id"]

    if args.attn_impl:
        import transformers

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(mid)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        model = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.bfloat16, device_map="cuda", attn_implementation=args.attn_impl
        ).eval()
    else:
        model, tok = harness.load_model(mid)

    ids = harness.make_prompt_ids(tok, args.prompt_len)
    w_bytes = roofline.weight_bytes(model)
    peak = roofline.load_peak_gbs()
    total_steps = args.warmup + args.steps + args.profile_steps + 8

    prefix, step, info = harness.build_decoder(model, ids, args.mode, total_steps)
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    harness.burn()

    evs = []
    for _ in range(args.steps):
        t0 = torch.cuda.Event(enable_timing=True)
        t0.record()
        step()
        t1 = torch.cuda.Event(enable_timing=True)
        t1.record()
        evs.append((t0, t1))
    torch.cuda.synchronize()
    times = [a.elapsed_time(b) for a, b in evs]
    wall_ms = statistics.median(times)

    from torch.profiler import ProfilerActivity, profile

    suffix = f"_{args.prompt_len}" + (f"_{args.attn_impl}" if args.attn_impl else "")
    trace_path = os.path.join(RESULTS, f"trace_breakdown_{args.mode}{suffix}.json")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.profile_steps):
            step()
        torch.cuda.synchronize()
    prof.export_chrome_trace(trace_path)

    with open(trace_path) as f:
        events = json.load(f).get("traceEvents", [])
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    cats = {}
    counts = {}
    for e in kernels:
        c = category(e.get("name", ""))
        cats[c] = cats.get(c, 0.0) + float(e.get("dur", 0.0))
        counts[c] = counts.get(c, 0) + 1
    n = args.profile_steps
    cat_ms = {k: v / n / 1e3 for k, v in cats.items()}
    cat_count = {k: v / n for k, v in counts.items()}
    kernel_ms = sum(cat_ms.values())
    weight_floor_ms = w_bytes / peak / 1e9 * 1e3

    out = {
        "model": args.model,
        "mode": args.mode,
        "attn_impl": args.attn_impl or "default",
        "prompt_len": args.prompt_len,
        "wall_ms_per_step": round(wall_ms, 4),
        "kernel_ms_per_step": round(kernel_ms, 4),
        "host_gap_ms_per_step": round(max(0.0, wall_ms - kernel_ms), 4),
        "kernels_per_step": len(kernels) / n,
        "weight_floor_ms": round(weight_floor_ms, 4),
        "categories_ms": {k: round(v, 4) for k, v in sorted(cat_ms.items(), key=lambda kv: -kv[1])},
        "categories_count": {k: round(v, 1) for k, v in sorted(cat_count.items(), key=lambda kv: -kv[1])},
        "trace": trace_path,
    }
    path = os.path.join(RESULTS, f"breakdown_{args.mode}{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
