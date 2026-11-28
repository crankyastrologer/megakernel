"""Generate the Phase 2 article figures from the measured results.

Outputs (PNG + PDF in ``results/``):

    fig4_phase2_waterfall   3.36 -> 1.69 ms, step by step
    fig5_projection_gap     per-projection measured time vs weight floor
    fig6_context_scaling    latency and % of roofline vs prompt length
    fig7_step_timeline      one decode step drawn to scale + SM coverage
    fig8_batch_plane        per-request latency vs aggregate throughput
    fig9_norm_design        where the RMSNorm reduction lives

Also writes small derived tables so the figures are reproducible without the
~200 MB chrome traces:

    results/phase2_per_projection.csv
    results/phase2_step_timeline.csv
    results/phase2_design_space.csv

Data sources: results/breakdown_graph_128.json,
results/breakdown_graph_fused_128.json, results/phase2_verify.json,
results/phase1.csv, results/llamacpp.csv, results/batch_hf_graph.json,
results/batch_vllm.json, results/trace_breakdown_*_128.json.

Run: .venv/bin/python scripts/make_phase2_figures.py
"""

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)

from bench.breakdown import category  # noqa: E402

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
GRAY = "#7f7f7f"
PURPLE = "#9467bd"
DARK = "#333333"

PEAK_GBS = 263.52
FLOOR_MS = 1.0209  # 269.03 MB / 263.52 GB/s

CAT_COLORS = {
    "gemm_weights": BLUE,
    "attention": ORANGE,
    "elementwise_other": GRAY,
    "sampling": RED,
    "memcpy": "#17becf",
}
CAT_LABELS = {
    "gemm_weights": "weight GEMV",
    "attention": "attention",
    "elementwise_other": "elementwise",
    "sampling": "sampling",
    "memcpy": "memcpy",
}

# Per-projection bf16 weight bytes; SmolLM2-135M shapes, 30 layers.
PROJECTIONS = [
    # name, bytes, measured ms (fused trace, 128 tokens), kernels, ctas
    ("qkv", 30 * 960 * 576 * 2, 0.2193, 30, 30),
    ("o_proj", 30 * 576 * 576 * 2, 0.1736, 30, 18),
    ("gate+up", 30 * 2 * 1536 * 576 * 2, 0.5853, 30, 48),
    ("down", 30 * 576 * 1536 * 2, 0.3209, 30, 18),
    ("lm_head", 576 * 49152 * 2, 0.2913, 1, None),
]


def load_json(name):
    """Load a JSON file from ``results/``."""
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def save(fig, stem):
    """Save a figure as PNG and PDF in ``results/``."""
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RESULTS, f"{stem}.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def titles(ax, main, sub=None):
    """Set a left-aligned bold title with an optional non-overlapping subtitle."""
    ax.set_title(main, loc="left", fontsize=11, fontweight="bold", pad=24 if sub else 8)
    if sub:
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=8,
                color="#555555", va="bottom")


def fmt(x):
    """Format a millisecond value with enough digits to be useful."""
    return f"{x:.3f}" if abs(x) < 0.1 else f"{x:.2f}"


# --------------------------------------------------------------------------
# fig 4: waterfall
# --------------------------------------------------------------------------
def fig4():
    """Decompose 3.36 ms -> 1.69 ms: what was removed and what was not."""
    hf = load_json("breakdown_graph_128.json")
    fu = load_json("breakdown_graph_fused_128.json")
    hw, fw = hf["wall_ms_per_step"], fu["wall_ms_per_step"]
    fk = fu["kernel_ms_per_step"]
    ch, cf = hf["categories_ms"], fu["categories_ms"]
    host = hf["host_gap_ms_per_step"]

    steps = [
        ("host gap\nremoved", -host, GREEN),
        ("1,100 elementwise\nkernels removed",
         -(ch["elementwise_other"] - cf["elementwise_other"]), GREEN),
        ("attention", cf["attention"] - ch["attention"], RED),
        ("sampling +\nmemcpy",
         (cf["sampling"] - ch["sampling"]) + cf.get("memcpy", 0.0), RED),
        ("weight GEMVs\n1.595 -> 1.591", cf["gemm_weights"] - ch["gemm_weights"], GRAY),
    ]

    labels = ["HF graph\n3.36 ms"]
    lefts, heights, bottoms, colors, stepsign, vals = [0], [hw], [0.0], [DARK], [""], [hw]
    cur = hw
    for i, (lab, delta, col) in enumerate(steps, start=1):
        lo = cur + delta if delta < 0 else cur
        labels.append(lab)
        lefts.append(i)
        heights.append(abs(delta))
        bottoms.append(lo)
        colors.append(col)
        stepsign.append("-" if delta < 0 else "+")
        cur += delta
        vals.append(cur)
    labels.extend(["fused\nkernel sum", "CUPTI\ninflation", "fused\n1.69 ms"])
    lefts.extend([6, 7, 8])
    heights.extend([cur, fk - fw, fw])
    bottoms.extend([0.0, fw, 0.0])
    colors.extend([DARK, GRAY, DARK])
    stepsign.extend(["", "-", ""])
    vals.extend([cur, fw, fw])

    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.bar(lefts, heights, bottom=bottoms, width=0.62, color=colors, zorder=3)
    for i in range(len(lefts) - 1):
        ax.plot([lefts[i] + 0.31, lefts[i + 1] - 0.31], [vals[i]] * 2,
                color="#bbbbbb", lw=0.8, ls=":", zorder=2)

    ax.axhline(FLOOR_MS, color=RED, ls="--", lw=1.1, zorder=2)
    ax.text(1.55, FLOOR_MS + 0.055, "weight floor 1.02 ms  (269 MB / 263.5 GB/s)",
            color=RED, ha="left", fontsize=8)

    for x, b, h, sign in zip(lefts, bottoms, heights, stepsign):
        ax.text(x, b + h + 0.045, sign + fmt(h), ha="center", va="bottom",
                fontsize=8, color="#222222")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ms per token")
    ax.set_ylim(0, 3.8)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    titles(
        ax,
        "Deleting kernels, not making the math faster",
        "SmolLM2-135M, 128-token prompt. Elementwise and host cost disappear; "
        "the weight stream does not move.",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    save(fig, "fig4_phase2_waterfall")
    print(f"fig4: wall {hw:.3f} -> {fw:.3f}; weights {ch['gemm_weights']:.4f} -> "
          f"{cf['gemm_weights']:.4f}; elementwise {ch['elementwise_other']:.3f} -> "
          f"{cf['elementwise_other']:.3f}")


# --------------------------------------------------------------------------
# fig 5: per-projection gap
# --------------------------------------------------------------------------
def fig5():
    """Show measured time vs bytes/peak for each weight kernel group."""
    rows = []
    for name, nbytes, ms, kernels, ctas in PROJECTIONS:
        floor = nbytes / (PEAK_GBS * 1e9) * 1e3
        rows.append((name, nbytes / 1e6, floor, ms, kernels, ctas))
    total_bytes = sum(r[1] for r in rows)
    total_floor = sum(r[2] for r in rows)
    total_ms = sum(r[3] for r in rows)

    with open(os.path.join(RESULTS, "phase2_per_projection.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["projection", "weight_mb", "floor_ms", "measured_ms", "pct_of_floor",
                    "kernels_per_step", "ctas_per_kernel", "overhead_us_per_kernel"])
        for name, mb, floor, ms, kernels, ctas in rows:
            w.writerow([name, f"{mb:.2f}", f"{floor:.4f}", f"{ms:.4f}",
                        f"{100 * floor / ms:.1f}", kernels, ctas if ctas else "n/a",
                        f"{1000 * (ms - floor) / kernels:.2f}"])
        w.writerow(["total", f"{total_bytes:.2f}", f"{total_floor:.4f}",
                    f"{total_ms:.4f}", f"{100 * total_floor / total_ms:.1f}",
                    sum(r[4] for r in rows), "", ""])

    order = sorted(rows, key=lambda r: -r[3])
    ys = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    for y, (name, mb, floor, ms, kernels, ctas) in zip(ys, order):
        ax.barh(y, floor, height=0.52, color=BLUE, zorder=3)
        ax.barh(y, ms - floor, left=floor, height=0.52, color="white",
                edgecolor=ORANGE, hatch="////", lw=0.0, zorder=3)
        ax.text(ms + 0.015, y, f"{ms:.3f} ms   {100 * floor / ms:.0f}%",
                va="center", fontsize=8.5)

    ytot = -1.25
    ax.barh(ytot, total_floor, height=0.52, color=BLUE, zorder=3)
    ax.barh(ytot, total_ms - total_floor, left=total_floor, height=0.52,
            color="white", edgecolor=ORANGE, hatch="////", lw=0.0, zorder=3)
    ax.text(total_ms + 0.015, ytot,
            f"{total_ms:.3f} ms   {100 * total_floor / total_ms:.0f}%",
            va="center", fontsize=8.5, fontweight="bold")

    ax.axvline(FLOOR_MS, color=RED, ls="--", lw=1.0, zorder=2)
    ticklabels = []
    for name, mb, floor, ms, kernels, ctas in order:
        if ctas:
            ticklabels.append(f"{name}\n{kernels} kern · {ctas} CTAs")
        else:
            ticklabels.append(f"{name}\n{kernels} cutlass kernel")
    ticklabels.append(f"total\n{sum(r[4] for r in rows)} kernels · ~4 us each")
    ax.set_yticks(list(ys) + [ytot])
    ax.set_yticklabels(ticklabels, fontsize=8)
    ax.set_xlabel("milliseconds per token")
    ax.set_xlim(0, 1.98)
    ax.set_ylim(-1.95, 4.65)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    titles(
        ax,
        "Where the 0.57 ms above the weight floor lives",
        "Solid = bytes / 263.5 GB/s; hatched = measured - floor. Red line = 1.02 ms floor.",
    )
    ax.legend(
        handles=[Patch(facecolor=BLUE, label="bytes / peak"),
                 Patch(facecolor="white", edgecolor=ORANGE, hatch="////", label="overhead")],
        loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2, fontsize=8, frameon=False,
    )
    save(fig, "fig5_projection_gap")
    print("fig5: wrote phase2_per_projection.csv")


# --------------------------------------------------------------------------
# fig 6: context scaling
# --------------------------------------------------------------------------
def fig6():
    """Latency and roofline efficiency vs prompt length."""
    verify = load_json("phase2_verify.json")
    p2 = {(r["prompt_len"], r["mode"]): r for r in verify}
    plens = sorted({r["prompt_len"] for r in verify})

    p1 = {}
    for r in csv.DictReader(open(os.path.join(RESULTS, "phase1.csv"))):
        if r["model"] == "smollm2-135m":
            p1[(int(r["prompt_len"]), r["mode"])] = float(r["median_ms"])

    llama = {}
    for r in csv.DictReader(open(os.path.join(RESULTS, "llamacpp.csv"))):
        if r["model"] == "SmolLM2-135M" and r["fa"] == "1":
            llama[(int(r["depth"]), r["quant"])] = float(r["ms_per_token"])

    x = np.array(plens)
    series = [
        ("fused Triton", [p2[(p, "graph_fused")]["median_ms"] for p in plens], GREEN, "o", "-", 2.2),
        ("HF CUDA graph", [p1[(p, "graph")] for p in plens], BLUE, "s", "-", 1.4),
        ("vLLM", [p1[(p, "vllm")] for p in plens], ORANGE, "^", "-", 1.4),
        ("llama.cpp F16+FA", [llama[(p, "F16")] for p in plens], PURPLE, "D", "-", 1.2),
        ("llama.cpp Q4_K_M (4-bit)", [llama[(p, "Q4_K_M")] for p in plens], GRAY, "v", ":", 1.2),
    ]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 4.45),
                                 gridspec_kw={"wspace": 0.26})

    for label, ys, color, marker, ls, lw in series:
        ax.plot(x, ys, color=color, marker=marker, ls=ls, lw=lw, ms=4.5, label=label, zorder=3)
    ax.axhline(FLOOR_MS, color=RED, ls="--", lw=1.0, zorder=2)
    ax.text(112, FLOOR_MS - 0.185, "weight floor 1.02 ms", color=RED, fontsize=8)
    ax.annotate(
        "+0.28 ms from 512 to 2048\n(all of it attention)",
        xy=(1500, 1.89), xytext=(150, 1.28),
        fontsize=7.5, color=GREEN,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.9),
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(plens)
    ax.set_xticklabels(plens)
    ax.set_xlim(110, 2400)
    ax.set_ylim(0.72, 4.1)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("ms / token")
    ax.set_title("Latency: one growing term", loc="left", fontsize=10)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    w_pct = [p2[(p, "graph_fused")]["pct_weight_roofline"] for p in plens]
    t_pct = [p2[(p, "graph_fused")]["pct_total_roofline"] for p in plens]
    hf_pct = [p2[(p, "graph")]["pct_total_roofline"] for p in plens]
    bx.plot(x, w_pct, color=BLUE, marker="o", ms=4.5, ls="--", lw=1.2,
            label="fused, weights only")
    bx.plot(x, t_pct, color=GREEN, marker="o", ms=4.5, lw=2.2,
            label="fused, weights + KV")
    bx.plot(x, hf_pct, color=GRAY, marker="s", ms=4.0, lw=1.2,
            label="HF graph, weights + KV")
    for xi, yi in zip(x, t_pct):
        bx.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5, color=GREEN)
    bx.annotate(f"{w_pct[-1]:.0f}%", (x[-1], w_pct[-1]), textcoords="offset points",
                xytext=(-2, -13), ha="center", fontsize=7.5, color=BLUE)
    bx.annotate("+9 pts\nfrom KV", xy=(2048, 55.5), xytext=(1150, 46),
                fontsize=7.5, color="#666666",
                arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.8))
    bx.set_xscale("log", base=2)
    bx.set_xticks(plens)
    bx.set_xticklabels(plens)
    bx.set_xlim(110, 2400)
    bx.set_ylim(20, 72)
    bx.set_xlabel("prompt length (tokens)")
    bx.set_ylabel("% of 263.5 GB/s")
    bx.set_title("Efficiency: flat once KV is counted", loc="left", fontsize=10)
    bx.grid(alpha=0.25)
    bx.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.11),
               fontsize=8, frameon=False)
    fig.suptitle("Context scaling at batch 1: the KV term grows, nothing else does",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    save(fig, "fig6_context_scaling")
    print(f"fig6: weight% {w_pct} total% {t_pct}")


# --------------------------------------------------------------------------
# fig 7: step timeline + SM coverage
# --------------------------------------------------------------------------
def _one_step(trace_name, kernels_per_step):
    """Return the last full step of a chrome trace as normalised kernel dicts."""
    ev = load_json(trace_name)["traceEvents"]
    k = [e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"]
    k.sort(key=lambda e: e["ts"])
    k = k[-kernels_per_step:]
    t0 = k[0]["ts"]
    out = []
    for e in k:
        blocks = 1
        for g in e["args"]["grid"]:
            blocks *= g
        out.append(
            {
                "name": e["name"],
                "cat": category(e["name"]),
                "start_us": e["ts"] - t0,
                "dur_us": max(float(e["dur"]), 0.35),
                "blocks": blocks,
                "coverage": min(blocks, 24) / 24.0,
            }
        )
    span = max(o["start_us"] + o["dur_us"] for o in out) / 1e3
    return out, span


def _step_from_csv(mode):
    """Load one normalised step from the derived timeline CSV (no traces needed)."""
    rows = [r for r in csv.DictReader(open(os.path.join(RESULTS, "phase2_step_timeline.csv")))
            if r["mode"] == mode]
    return [
        {
            "name": r["name"],
            "cat": r["category"],
            "start_us": float(r["start_us"]),
            "dur_us": float(r["dur_us"]),
            "blocks": int(r["blocks"]),
            "coverage": float(r["sm_coverage"]),
        }
        for r in rows
    ]


def fig7():
    """Draw one decode step to scale and the distribution of SM coverage."""
    hf_k, fu_k = 1372, 254
    try:
        hf, _ = _one_step("trace_breakdown_graph_128.json", hf_k)
        fu, _ = _one_step("trace_breakdown_graph_fused_128.json", fu_k)
        with open(os.path.join(RESULTS, "phase2_step_timeline.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mode", "name", "category", "start_us", "dur_us", "blocks",
                        "sm_coverage"])
            for mode, rows in (("fused", fu), ("hf_graph", hf)):
                for r in rows:
                    w.writerow([mode, r["name"][:120], r["cat"], f"{r['start_us']:.2f}",
                                f"{r['dur_us']:.2f}", r["blocks"], f"{r['coverage']:.3f}"])
    except (FileNotFoundError, OSError):
        print("fig7: traces missing, using results/phase2_step_timeline.csv")
        hf = _step_from_csv("hf_graph")
        fu = _step_from_csv("fused")
    hf_span = max(r["start_us"] + r["dur_us"] for r in hf) / 1e3
    fu_span = max(r["start_us"] + r["dur_us"] for r in fu) / 1e3

    fig = plt.figure(figsize=(11.0, 6.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.55], hspace=0.5)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2])

    def barcode(ax, rows, label):
        by_cat = {}
        for r in rows:
            by_cat.setdefault(r["cat"], []).append(r)
        for cat, rs in by_cat.items():
            ax.barh([0] * len(rs), [r["dur_us"] / 1e3 for r in rs],
                    left=[r["start_us"] / 1e3 for r in rs],
                    height=0.62, color=CAT_COLORS.get(cat, GRAY), lw=0)
        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=9)
        ax.set_ylim(-0.6, 1.6)
        ax.grid(axis="x", alpha=0.2)

    barcode(ax1, fu, f"fused\n{fu_k} kernels")
    barcode(ax2, hf, f"HF graph\n{hf_k} kernels")
    ax1.set_xlim(0, 3.5)
    ax1.tick_params(labelbottom=False)
    ax2.set_xlabel("time within the step (ms)")

    def annotate(ax, x, text):
        ax.annotate(text, xy=(x, 0.31), xytext=(x, 1.22), fontsize=8, ha="center",
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8))

    annotate(ax1, 1.30, "121 weight kernels, 1.59 ms")
    annotate(ax2, 1.10, "1,100 elementwise kernels, 1.34 ms")
    titles(ax1, "One decode step, drawn to scale", None)
    handles = [Patch(facecolor=CAT_COLORS[k], label=CAT_LABELS[k]) for k in CAT_COLORS]
    ax1.legend(handles=handles, ncol=2, fontsize=7.5, frameon=False,
               loc="center left", bbox_to_anchor=(0.56, 0.62))

    # SM-coverage histogram: how the step's kernel time splits across coverage bands
    edges = [0.0, 0.25, 0.5, 0.75, 1.0001]
    names = ["<25%", "25-50%", "50-75%", "75-100%"]
    width = 0.38
    for i, (rows, color, label) in enumerate(
        ((hf, GRAY, "HF graph"), (fu, GREEN, "fused Triton"))
    ):
        tot = sum(r["dur_us"] for r in rows)
        frac = [
            sum(r["dur_us"] for r in rows if lo <= r["coverage"] < hi) / tot * 100
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
        xs = np.arange(len(names)) + (i - 0.5) * width
        ax3.bar(xs, frac, width=width, color=color, label=label, zorder=3)
        for x, v in zip(xs, frac):
            if v >= 2.0:
                ax3.text(x, v + 1.2, f"{v:.0f}%", ha="center", fontsize=7.5, color=color)
        print(f"fig7 {label}: " + " ".join(f"{n}={v:.0f}%" for n, v in zip(names, frac)))

    ax3.set_xticks(np.arange(len(names)))
    ax3.set_xticklabels(names)
    ax3.set_ylim(0, 105)
    ax3.set_ylabel("% of kernel time")
    ax3.set_xlabel("fraction of the 24 SMs with at least one block resident")
    ax3.grid(axis="y", alpha=0.25)
    ax3.set_axisbelow(True)
    ax3.legend(fontsize=8, frameon=False, loc="upper left")
    ax3.set_title(
        "HF spends 40% of its kernel time below 25% SM coverage; the fused step spends 3%",
        loc="left", fontsize=9,
    )

    save(fig, "fig7_step_timeline")
    print(f"fig7: spans fused {fu_span:.3f} ms, hf {hf_span:.3f} ms")


# --------------------------------------------------------------------------
# fig 8: batch plane
# --------------------------------------------------------------------------
def fig8():
    """Per-request latency vs aggregate throughput across batch sizes."""
    hf = load_json("batch_hf_graph.json")
    vl = load_json("batch_vllm.json")
    verify = {(r["prompt_len"], r["mode"]): r for r in load_json("phase2_verify.json")}

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, plen in zip(axes, (128, 2048)):
        hfr = sorted([r for r in hf if r["prompt_len"] == plen], key=lambda r: r["batch"])
        vlr = sorted([r for r in vl if r["prompt_len"] == plen], key=lambda r: r["batch"])

        for rows, color, label, marker, dy in (
            (hfr, GRAY, "HF CUDA graph", "s", 6),
            (vlr, ORANGE, "vLLM", "^", -13),
        ):
            xs = [r["agg_tok_per_s"] for r in rows]
            ys = [r["per_request_ms"] for r in rows]
            ax.plot(xs, ys, color=color, marker=marker, ms=5, lw=1.3, label=label, zorder=3)
            for r, xi, yi in zip(rows, xs, ys):
                ax.annotate(f"B={r['batch']}", (xi, yi), textcoords="offset points",
                            xytext=(7, dy), fontsize=7, color=color)

        fused = verify[(plen, "graph_fused")]
        ms = fused["median_ms"]
        ax.plot([1000.0 / ms], [ms], marker="*", ms=17, color=GREEN, zorder=5,
                label="fused Triton (B=1)")
        ax.annotate("fused (B=1)", (1000.0 / ms, ms), textcoords="offset points",
                    xytext=(11, -3), ha="left", fontsize=7.5, color=GREEN)

        ax.axhline(FLOOR_MS, color=RED, ls="--", lw=1.0, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(230, 9500)
        ax.set_ylim(0.8, 6.2)
        ax.set_xticks([300, 500, 1000, 2000, 4000, 8000])
        ax.set_xticklabels(["300", "500", "1k", "2k", "4k", "8k"])
        ax.minorticks_off()
        ax.set_xlabel("aggregate decode throughput (tok/s)")
        ax.set_title(f"{plen}-token prompt", loc="left", fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("per-request latency (ms / token)")
    axes[0].text(245, FLOOR_MS + 0.12, "weight floor 1.02 ms", color=RED, fontsize=8)
    axes[1].text(9350, FLOOR_MS + 0.22, "weight-only ceiling: 979 x B tok/s",
                 color="#999999", fontsize=7.5, ha="right")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")

    fig.suptitle("B=1 and B=8 are different problems: latency vs aggregate throughput",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    save(fig, "fig8_batch_plane")
    print("fig8: done")


# --------------------------------------------------------------------------
# fig 9: norm design space
# --------------------------------------------------------------------------
def fig9():
    """Show that fusing the RMSNorm reduction into every block is a regression."""
    designs = [
        ("pure GEMV\n(no norm)", 0.212, GRAY, "reference"),
        ("1-CTA norm_stats\n+ GEMV", 0.258, ORANGE, "+46 us, extra launch"),
        ("norm reduction\nin every block", 0.312, RED, "+100 us: stream serialises"),
        ("producer partials\n+ GEMV (shipped)", 0.219, GREEN, "+7 us"),
    ]
    with open(os.path.join(RESULTS, "phase2_design_space.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "ms_30_layers", "delta_vs_pure_gemv_ms", "note"])
        for label, ms, _, note in designs:
            w.writerow([label.replace("\n", " "), f"{ms:.3f}", f"{ms - 0.212:.3f}", note])

    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    xs = np.arange(len(designs))
    ax.bar(xs, [d[1] for d in designs], width=0.55, color=[d[2] for d in designs], zorder=3)
    for x, (label, ms, _, note) in zip(xs, designs):
        ax.text(x, ms + 0.008, f"{ms:.3f} ms", ha="center", fontsize=8.5,
                fontweight="bold" if "shipped" in label else "normal")
        ax.text(x, ms + 0.024, note, ha="center", fontsize=7.5, color="#555555")
    ax.axhline(0.212, color=GRAY, ls=":", lw=0.9, zorder=2)
    ax.axhline(0.126, color=RED, ls="--", lw=1.0, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([d[0] for d in designs], fontsize=8)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel("qkv stage, 30 layers (ms)")
    ax.set_ylim(0, 0.375)
    titles(
        ax,
        "Fusion is free only when it does not move the reduction",
        "Cold-cache microbenchmark (256 MB read flush), 30 layers of real weights. "
        "Red dashed = 0.126 ms weight floor.",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    save(fig, "fig9_norm_design")
    print("fig9: wrote phase2_design_space.csv")


def main():
    """Render every Phase 2 figure."""
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
    })
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()


if __name__ == "__main__":
    main()
