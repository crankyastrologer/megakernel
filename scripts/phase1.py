"""Phase 1 orchestrator: run all torch decode modes, measure, and emit CSV/plots.

Usage (inside WSL):
  .venv/bin/python scripts/phase1.py [--models smollm2-135m,tinyllama-1.1b]
"""

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from bench import count, harness, roofline  # noqa: E402

RESULTS = os.path.join(ROOT, "results")


def clocks():
    """Query current SM/memory clocks, power and temperature.

    Returns:
        str: nvidia-smi CSV line, or an ``unavailable: ...`` string.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu", "--format=csv,noheader"],
            text=True,
        )
        return out.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def load_cfg():
    """Load ``models.yaml`` from the project root.

    Returns:
        dict: Parsed YAML with ``models``, ``runtime`` and ``roofline`` sections.
    """
    with open(os.path.join(ROOT, "models.yaml")) as f:
        return yaml.safe_load(f)


def parity(tokens, ref):
    """Compare generated tokens against the reference prefix.

    Args:
        tokens (list[int]): Tokens produced by a decode mode.
        ref (list[int]): Reference greedy tokens (may be longer).

    Returns:
        tuple[bool, int | None]: ``(matches, first_mismatch_index)``. The index
        is ``None`` when the overlapping prefix matches.
    """
    if not tokens:
        return False, None
    n = min(len(tokens), len(ref))
    for i in range(n):
        if tokens[i] != ref[i]:
            return False, i
    return True, None


def plot(rows, out_dir):
    """Write per-model latency and roofline PNGs from result rows.

    Args:
        rows (list[dict]): Result rows with ``model``, ``mode``, ``prompt_len``,
            ``parity``, ``median_ms`` and ``pct_weight_roofline``.
        out_dir (str): Directory for ``phase1_<model>_{latency,roofline}.png``.

    Returns:
        None. Plotting is skipped with a message if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("plot skipped:", repr(exc))
        return

    models = sorted({r["model"] for r in rows})
    modes = sorted({r["mode"] for r in rows})
    colors = {m: c for m, c in zip(modes, ["#888", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"])}

    for model in models:
        mr = [r for r in rows if r["model"] == model]
        if not mr:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        plens = sorted({r["prompt_len"] for r in mr})
        for mode in modes:
            xs, ys = [], []
            for plen in plens:
                vals = [r for r in mr if r["mode"] == mode and r["prompt_len"] == plen and r["parity"]]
                if vals:
                    xs.append(plen)
                    ys.append(vals[0]["median_ms"])
            if xs:
                ax.plot(xs, ys, marker="o", label=mode, color=colors.get(mode))
        ax.set_xlabel("prompt length (tokens)")
        ax.set_ylabel("ms / decode token (median)")
        ax.set_title(f"{model}: decode latency")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"phase1_{model}_latency.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        plen0 = min(plens)
        modes0 = [m for m in modes if any(r["model"] == model and r["mode"] == m and r["prompt_len"] == plen0 and r["parity"] for r in mr)]
        vals = [
            next(r["pct_weight_roofline"] for r in mr if r["mode"] == m and r["prompt_len"] == plen0 and r["parity"])
            for m in modes0
        ]
        bars = ax.bar(modes0, vals, color=[colors.get(m) for m in modes0])
        ax.axhline(100, color="k", linestyle="--", linewidth=1, label="HBM read roofline")
        ax.axhline(70, color="r", linestyle=":", linewidth=1, label="70% gate")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("% of sustained read roofline")
        ax.set_title(f"{model}: roofline at prompt {plen0}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"phase1_{model}_roofline.png"), dpi=150)
        plt.close(fig)


def main():
    """Run the Phase 1 sweep and write CSV, profile, meta and plot artifacts.

    For each model and prompt length, runs the requested torch decode modes,
    merges any vLLM results from ``results/vllm.json``, checks token parity
    against manual eager, and optionally profiles kernels. Rows are flushed
    incrementally so interrupted runs can be resumed.

    CLI flags: ``--models``, ``--prompt-lengths``, ``--steps``, ``--warmup``,
    ``--modes``, ``--profile``, ``--profile-steps``, ``--resume``.

    Returns:
        None. Writes ``results/phase1.csv``, ``results/phase1_profile.csv``,
        ``results/phase1_meta.json`` and PNG plots.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma-separated keys from models.yaml")
    ap.add_argument("--prompt-lengths", default=None, help="comma-separated ints")
    ap.add_argument("--steps", type=int, default=None, help="measured decode steps")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--modes", default="eager,static,graph,compile")
    ap.add_argument("--profile", action="store_true", help="also profile kernels at the first prompt length")
    ap.add_argument("--profile-steps", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="merge with existing phase1.csv instead of overwriting")
    args = ap.parse_args()

    cfg = load_cfg()
    model_keys = args.models.split(",") if args.models else list(cfg["models"].keys())
    rt = cfg["runtime"]
    plens = [int(x) for x in args.prompt_lengths.split(",")] if args.prompt_lengths else rt["prompt_lengths"]
    measure = args.steps or rt["measure_steps"]
    warmup = args.warmup if args.warmup is not None else rt["warmup_steps"]
    profile_steps = args.profile_steps or rt["profile_steps"]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    peak = roofline.load_peak_gbs()

    os.makedirs(RESULTS, exist_ok=True)
    vllm_path = os.path.join(RESULTS, "vllm.json")
    vllm_data = {}
    if os.path.exists(vllm_path):
        with open(vllm_path) as f:
            vllm_data = json.load(f)
    rows = []
    prof_rows = []
    prof_fields = [
        "model", "mode", "prompt_len", "steps", "median_ms", "kernels_per_step",
        "kernel_ms_per_step", "memcpy_ms_per_step", "gap_ms_per_step",
        "profiled_wall_ms_per_step", "profiler_gap_ms_per_step", "trace",
    ]
    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "peak_read_gbs": peak,
        "measure_steps": measure,
        "warmup_steps": warmup,
        "compile_mode": os.environ.get("COMPILE_MODE", "default"),
        "clocks_before": clocks(),
    }

    fields = [
        "model", "model_id", "prompt_len", "mode", "n",
        "median_ms", "mean_ms", "p95_ms", "min_ms", "max_ms", "tok_per_s",
        "weight_gb", "kv_gb_per_token", "weight_gbs", "total_gbs",
        "pct_weight_roofline", "pct_total_roofline", "parity", "first_mismatch", "wall_s", "error",
    ]
    phase1_csv = os.path.join(RESULTS, "phase1.csv")

    def write_csv(path, data, fields):
        """Write rows to a CSV with a fixed column order.

        Args:
            path (str): Output CSV path.
            data (list[dict]): Rows to write.
            fields (list[str]): Column order; extra keys are ignored.
        """
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in data:
                w.writerow(r)

    def flush():
        """Rewrite ``results/phase1.csv`` from the in-memory rows."""
        write_csv(phase1_csv, rows, fields)

    NUM_INT = {"prompt_len", "n", "first_mismatch"}
    NUM_FLOAT = {
        "median_ms", "mean_ms", "p95_ms", "min_ms", "max_ms", "tok_per_s",
        "weight_gb", "kv_gb_per_token", "weight_gbs", "total_gbs",
        "pct_weight_roofline", "pct_total_roofline", "wall_s",
    }

    def coerce_row(r):
        """Convert CSV strings back to bool/int/float for resumed rows.

        Args:
            r (dict): Row from ``csv.DictReader`` (string values).

        Returns:
            dict: The same row with ``parity`` and numeric fields coerced.
        """
        if isinstance(r.get("parity"), str):
            r["parity"] = r["parity"] == "True"
        for k in NUM_INT:
            if k in r and isinstance(r[k], str) and r[k] != "":
                try:
                    r[k] = int(float(r[k]))
                except ValueError:
                    r[k] = None
        for k in NUM_FLOAT:
            if k in r and isinstance(r[k], str) and r[k] != "":
                try:
                    r[k] = float(r[k])
                except ValueError:
                    r[k] = None
        return r

    def upsert(row):
        """Replace a row with the same ``(model, prompt_len, mode)`` and flush.

        Args:
            row (dict): Result row to insert or replace.
        """
        rows[:] = [
            r
            for r in rows
            if not (r.get("model") == row["model"] and r.get("prompt_len") == row["prompt_len"] and r.get("mode") == row["mode"])
        ]
        rows.append(row)
        flush()

    if args.resume and os.path.exists(phase1_csv):
        with open(phase1_csv, newline="") as f:
            for r in csv.DictReader(f):
                rows.append(coerce_row(r))
        print(f"resumed {len(rows)} rows from {phase1_csv}", flush=True)
        meta_path = os.path.join(RESULTS, "phase1_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                old = json.load(f)
            merged = {**old, **meta}
            merged["models"] = {**old.get("models", {}), **meta.get("models", {})}
            meta = merged

    prof_path = os.path.join(RESULTS, "phase1_profile.csv")

    def flush_prof():
        """Rewrite ``results/phase1_profile.csv`` when profile rows exist."""
        if prof_rows:
            write_csv(prof_path, prof_rows, prof_fields)

    if args.resume and os.path.exists(prof_path):
        with open(prof_path, newline="") as f:
            for r in csv.DictReader(f):
                prof_rows.append(r)
        print(f"resumed {len(prof_rows)} profile rows from {prof_path}", flush=True)

    for key in model_keys:
        mcfg = cfg["models"][key]
        mid = mcfg["id"]
        short = mcfg.get("short", key)
        print(f"\n=== loading {key} ({mid}) ===", flush=True)
        t0 = time.perf_counter()
        model, tok = harness.load_model(mid)
        w_bytes = roofline.weight_bytes(model)
        cfgm = model.config
        print(f"loaded in {time.perf_counter() - t0:.1f}s; weight bytes {w_bytes / 1e9:.3f} GB", flush=True)
        meta.setdefault("models", {})[key] = {
            "id": mid,
            "short": short,
            "weight_gb": w_bytes / 1e9,
            "num_layers": getattr(cfgm, "num_hidden_layers", None),
            "num_kv_heads": getattr(cfgm, "num_key_value_heads", None),
            "hidden_size": getattr(cfgm, "hidden_size", None),
            "params": sum(p.numel() for p in model.parameters()),
        }

        for plen in plens:
            ids = harness.make_prompt_ids(tok, plen)
            n_steps = warmup + measure
            gen_ref = harness.reference_tokens(model, ids, n_steps + 4)
            # HF generate can diverge from manual/StaticCache/vLLM for some models.
            # Use manual eager as the parity reference; they all share the same math.
            ref = harness.run_mode(model, ids, "eager", n_steps)["tokens"]
            gdiff = next((i for i, (a, b) in enumerate(zip(gen_ref, ref)) if a != b), None)
            print(
                f"-- prompt_len {plen}: ref tokens {len(ref)}"
                + (f"  (HF generate diverges at {gdiff})" if gdiff is not None else "  (HF generate matches)"),
                flush=True,
            )
            meta.setdefault("generate_divergence", {})[f"{key}|{plen}"] = gdiff
            kv_bytes = roofline.kv_bytes_per_token(cfgm, harness.DTYPE, seq_len=plen + n_steps // 2)

            for mode in modes:
                try:
                    t = time.perf_counter()
                    res = harness.run_mode(model, ids, mode, n_steps)
                    elapsed = time.perf_counter() - t
                    stats = harness.time_stats(res["times"], warmup)
                    m = roofline.metrics(stats["median_ms"], w_bytes, kv_bytes, peak)
                    ok, mismatch = parity(res["tokens"], ref)
                    row = {
                        "model": key,
                        "model_id": mid,
                        "prompt_len": plen,
                        "mode": mode,
                        "n": stats["n"],
                        "median_ms": round(stats["median_ms"], 4),
                        "mean_ms": round(stats["mean_ms"], 4),
                        "p95_ms": round(stats["p95_ms"], 4),
                        "min_ms": round(stats["min_ms"], 4),
                        "max_ms": round(stats["max_ms"], 4),
                        "tok_per_s": round(m["tok_per_s"], 3),
                        "weight_gb": round(m["weight_gb"], 4),
                        "kv_gb_per_token": round(m["kv_gb_per_token"], 5),
                        "weight_gbs": round(m["weight_gbs"], 2),
                        "total_gbs": round(m["total_gbs"], 2),
                        "pct_weight_roofline": round(m["pct_weight_roofline"], 2),
                        "pct_total_roofline": round(m["pct_total_roofline"], 2),
                        "parity": ok,
                        "first_mismatch": mismatch,
                        "wall_s": round(elapsed, 2),
                    }
                    upsert(row)
                    print(
                        f"   {mode:8s} med {stats['median_ms']:8.3f} ms  p95 {stats['p95_ms']:8.3f}"
                        f"  {m['tok_per_s']:7.1f} tok/s  {m['pct_weight_roofline']:6.1f}% roofline"
                        f"  parity={ok}",
                        flush=True,
                    )
                    del res
                except Exception as exc:
                    import traceback

                    traceback.print_exc()
                    upsert(
                        {
                            "model": key,
                            "model_id": mid,
                            "prompt_len": plen,
                            "mode": mode,
                            "n": 0,
                            "median_ms": None,
                            "mean_ms": None,
                            "p95_ms": None,
                            "min_ms": None,
                            "max_ms": None,
                            "tok_per_s": None,
                            "weight_gb": round(w_bytes / 1e9, 4),
                            "kv_gb_per_token": round(kv_bytes / 1e9, 5),
                            "weight_gbs": None,
                            "total_gbs": None,
                            "pct_weight_roofline": None,
                            "pct_total_roofline": None,
                            "parity": False,
                            "first_mismatch": None,
                            "wall_s": None,
                            "error": repr(exc)[:300],
                        }
                    )
                    print(f"   {mode:8s} FAILED: {exc!r}", flush=True)

            vrec = vllm_data.get(f"{key}|{plen}")
            if vrec and vrec.get("tpot_ms"):
                tpot = float(vrec["tpot_ms"])
                all_t = [x for x in vrec.get("tpot_all_ms", []) if x is not None]
                p95 = sorted(all_t)[min(len(all_t) - 1, int(0.95 * len(all_t)))] if all_t else tpot
                ok, mismatch = parity(vrec.get("tokens", []), ref)
                m = roofline.metrics(tpot, w_bytes, kv_bytes, peak)
                upsert(
                    {
                        "model": key,
                        "model_id": mid,
                        "prompt_len": plen,
                        "mode": "vllm",
                        "n": len(vrec.get("tokens", [])),
                        "median_ms": round(tpot, 4),
                        "mean_ms": round(sum(all_t) / len(all_t), 4) if all_t else None,
                        "p95_ms": round(p95, 4),
                        "min_ms": round(min(all_t), 4) if all_t else None,
                        "max_ms": round(max(all_t), 4) if all_t else None,
                        "tok_per_s": round(m["tok_per_s"], 3),
                        "weight_gb": round(m["weight_gb"], 4),
                        "kv_gb_per_token": round(m["kv_gb_per_token"], 5),
                        "weight_gbs": round(m["weight_gbs"], 2),
                        "total_gbs": round(m["total_gbs"], 2),
                        "pct_weight_roofline": round(m["pct_weight_roofline"], 2),
                        "pct_total_roofline": round(m["pct_total_roofline"], 2),
                        "parity": ok,
                        "first_mismatch": mismatch,
                        "wall_s": None,
                    }
                )
                print(
                    f"   {'vllm':8s} med {tpot:8.3f} ms  p95 {p95:8.3f}"
                    f"  {m['tok_per_s']:7.1f} tok/s  {m['pct_weight_roofline']:6.1f}% roofline"
                    f"  parity={ok}",
                    flush=True,
                )

        if args.profile:
            plen = plens[0]
            ids = harness.make_prompt_ids(tok, plen)
            for mode in modes:
                try:
                    prefix, step, info = harness.build_decoder(model, ids, mode, profile_steps)
                    trace = os.path.join(RESULTS, f"trace_{key}_{mode}.json")
                    pm = count.profile_steps(step, profile_steps, trace, warmup_steps=2)
                    pm.update({"model": key, "mode": mode, "prompt_len": plen})
                    timing = next(
                        (r for r in rows if r["model"] == key and r["mode"] == mode and r["prompt_len"] == plen and r.get("median_ms") is not None),
                        None,
                    )
                    if timing is not None:
                        pm["median_ms"] = timing["median_ms"]
                        pm["gap_ms_per_step"] = round(
                            max(0.0, timing["median_ms"] - pm["kernel_ms_per_step"] - pm["memcpy_ms_per_step"]), 4
                        )
                    prof_rows[:] = [
                        r
                        for r in prof_rows
                        if not (
                            r.get("model") == key
                            and r.get("mode") == mode
                            and str(r.get("prompt_len")) == str(plen)
                        )
                    ]
                    prof_rows.append(pm)
                    flush_prof()
                    print(
                        f"   [profile] {mode:8s} {pm['kernels_per_step']:.0f} kernels/step,"
                        f" gpu {pm['kernel_ms_per_step']:.3f} ms, gap {pm.get('gap_ms_per_step', float('nan')):.3f} ms",
                        flush=True,
                    )
                    del step, info
                except Exception as exc:
                    print(f"   [profile] {mode:8s} FAILED: {exc!r}", flush=True)

        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    meta["clocks_after"] = clocks()

    flush()
    flush_prof()
    with open(os.path.join(RESULTS, "phase1_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    plot([r for r in rows if r.get("parity")], RESULTS)
    print(f"\nwrote {phase1_csv}")


if __name__ == "__main__":
    main()
