"""End-to-end Phase 2 verification: parity + timing at multiple prompt lengths.

Runs manual eager (reference), the HF CUDA graph, and the fused Triton graph,
and reports median/p95 plus both roofline percentages:
  - weight-only: weight bytes / time / peak (Part 1 convention)
  - total:       (weight + KV) bytes / time / peak

Usage:
  .venv/bin/python scripts/phase2_verify.py
  .venv/bin/python scripts/phase2_verify.py --prompt-lengths 128,512,2048 \
      --modes graph,graph_fused --json results/phase2_verify.json
"""

import argparse
import gc
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from bench import harness, roofline  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def clocks():
    """Return the nvidia-smi clock/power/temperature line, or 'unavailable'."""
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception as exc:  # pragma: no cover - WSL/driver dependent
        return f"unavailable: {exc}"


def parity(tokens, ref):
    """Compare generated token prefixes.

    Args:
        tokens (list[int]): Tokens from a decode mode (prefix included).
        ref (list[int]): Manual-eager reference tokens.

    Returns:
        tuple[bool, int | None]: match flag and first mismatch index.
    """
    n = min(len(tokens), len(ref))
    mism = next((i for i in range(n) if tokens[i] != ref[i]), None)
    return mism is None, mism


def main():
    """Run the parity/timing sweep and optionally write JSON records."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-lengths", default="128,512,2048")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--modes", default="graph,graph_fused")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    plens = [int(x) for x in args.prompt_lengths.split(",")]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    model, tok = harness.load_model(MODEL)
    w_bytes = roofline.weight_bytes(model)
    peak = roofline.load_peak_gbs()
    rows = []

    print(
        f"{'plen':>5} {'mode':12s} {'median':>8} {'p95':>8} "
        f"{'parity':>6} {'diff':>5} {'w%':>6} {'tot%':>6}"
    )
    for plen in plens:
        ids = harness.make_prompt_ids(tok, plen)
        ref_res = harness.run_mode(model, ids, "eager", args.steps)
        ref = ref_res["tokens"]
        del ref_res
        for mode in modes:
            res = harness.run_mode(model, ids, mode, args.steps)
            stats = harness.time_stats(res["times"], args.warmup)
            ok, mism = parity(res["tokens"], ref)
            kv = roofline.kv_bytes_per_token(
                model.config, harness.DTYPE, seq_len=plen + args.steps // 2
            )
            met = roofline.metrics(stats["median_ms"], w_bytes, kv, peak)
            rec = {
                "model": MODEL,
                "prompt_len": plen,
                "mode": mode,
                "median_ms": round(stats["median_ms"], 4),
                "p95_ms": round(stats["p95_ms"], 4),
                "tok_per_s": round(met["tok_per_s"], 3),
                "parity": ok,
                "first_mismatch": mism,
                "weight_gb": round(w_bytes / 1e9, 6),
                "kv_gb_per_token": round(kv / 1e9, 6),
                "pct_weight_roofline": round(met["pct_weight_roofline"], 2),
                "pct_total_roofline": round(met["pct_total_roofline"], 2),
                "peak_read_gbs": peak,
                "clocks": clocks(),
            }
            rows.append(rec)
            print(
                f"{plen:5d} {mode:12s} {stats['median_ms']:8.3f} {stats['p95_ms']:8.3f} "
                f"{str(ok):>6} {str(mism):>5} "
                f"{met['pct_weight_roofline']:6.1f} {met['pct_total_roofline']:6.1f}"
            )
            del res
            gc.collect()
            torch.cuda.empty_cache()
        del ref
        gc.collect()
        torch.cuda.empty_cache()

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
