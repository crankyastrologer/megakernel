"""Generate the Part 1 article figures from the measured results.

Outputs (PNG + PDF):
  results/fig1_roofline_by_engine.{png,pdf}
  results/fig2_context_scaling.{png,pdf}
  results/fig3_breakdown.{png,pdf}

Data sources: results/phase1.csv, results/vllm.json,
results/llamacpp.csv, results/breakdown_graph_128.json.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
GRAY = "#7f7f7f"
PURPLE = "#9467bd"


def save(fig, stem):
    """Save a figure as PNG and PDF in ``results/``.

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        stem (str): Output filename without extension.
    """
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RESULTS, f"{stem}.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig1():
    """Percent of roofline by engine for both models (prompt 128, min-max whiskers)."""
    panels = [
        {
            "title": "SmolLM2-135M",
            "labels": ["vLLM", "llama.cpp\nF16+FA", "ours\n(CUDA graph)"],
            "vals": [44.5, 44.0, 30.1],
            "lo": [42.3, 40.2, 27.5],
            "hi": [46.8, 44.0, 30.1],
            "colors": [BLUE, ORANGE, GREEN],
        },
        {
            "title": "Qwen2.5-1.5B",
            "labels": ["vLLM", "ours\n(CUDA graph)"],
            "vals": [93.8, 81.1],
            "lo": [93.4, 62.7],
            "hi": [95.0, 81.1],
            "colors": [BLUE, GREEN],
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), sharey=True, constrained_layout=True)
    for ax, p in zip(axes, panels):
        x = np.arange(len(p["labels"]))
        bars = ax.bar(x, p["vals"], color=p["colors"], width=0.6, zorder=3)
        lower = [v - lo for v, lo in zip(p["vals"], p["lo"])]
        upper = [hi - v for v, hi in zip(p["vals"], p["hi"])]
        ax.errorbar(x, p["vals"], yerr=[lower, upper], fmt="none", ecolor="black", capsize=4, lw=1, zorder=4)
        for b, v in zip(bars, p["vals"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(p["labels"], fontsize=9)
        ax.set_title(p["title"], fontsize=12)
        ax.set_ylim(0, 112)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(70, color=RED, ls=":", lw=1.2, zorder=2)
        ax.axhline(85, color="#8c1c13", ls=":", lw=1.2, zorder=2)

    axes[0].set_ylabel("% of 263.5 GB/s read roofline")
    axes[0].text(0.02, 0.97, "70% gate", transform=axes[0].transAxes, color=RED, fontsize=8, va="top")
    axes[0].text(0.02, 0.88, "85% gate", transform=axes[0].transAxes, color="#8c1c13", fontsize=8, va="top")
    axes[1].text(
        0.5,
        6,
        "llama.cpp\nnot run",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=GRAY,
        bbox=dict(facecolor="white", edgecolor="none", pad=2),
    )
    fig.suptitle("Percent of roofline by engine (bars: 128 tokens; whiskers: min–max over 128/512/2048)", fontsize=11)
    save(fig, "fig1_roofline_by_engine")


def fig2():
    """Decode latency vs context, before/after split-K, with vLLM and llama.cpp."""
    ctx = [128, 512, 2048]
    before = [4.80, 4.30, 6.69]
    after = [3.39, 3.43, 3.72]
    vllm = [2.2955, 2.1837, 2.4143]
    llama = [2.3224, 2.3667, 2.5428]

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.plot(ctx, before, "o--", color=RED, lw=2, ms=6, label="ours, pre-fix (4D-mask attention)")
    ax.plot(ctx, after, "o-", color=GREEN, lw=2.2, ms=6, label="ours, split-K Triton attention")
    ax.plot(ctx, vllm, "s-", color=BLUE, lw=1.8, ms=5, label="vLLM")
    ax.plot(ctx, llama, "^-", color=ORANGE, lw=1.8, ms=5, label="llama.cpp F16 + flash-attn")
    ax.axhline(1.02, color=GRAY, ls=":", lw=1.1, label="weight-read floor (1.02 ms)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ctx)
    ax.set_xticklabels([str(c) for c in ctx])
    ax.minorticks_off()
    ax.set_xlim(100, 2600)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("ms / token (median)")
    ax.set_ylim(0, 7.6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", frameon=False, labelspacing=0.35)

    ax.annotate(f"{before[-1]:.2f}", xy=(2048, before[-1]), xytext=(6, 8), textcoords="offset points", color=RED, fontsize=8)
    ax.annotate(f"{after[-1]:.2f}", xy=(2048, after[-1]), xytext=(6, -12), textcoords="offset points", color=GREEN, fontsize=8)
    ax.set_title("Context scaling: before and after the split-K attention fix", fontsize=11)
    save(fig, "fig2_context_scaling")


def fig3():
    """Stacked per-token breakdown of 3.36 ms with the 1.02 ms floor marked."""
    components = [
        ("Weight streaming (GEMM/GEMV)", 1.59, BLUE),
        ("Elementwise/other (~1,100 kernels)", 1.33, ORANGE),
        ("Host gap", 0.26, GRAY),
        ("Attention", 0.17, GREEN),
        ("Sampling (argmax)", 0.011, PURPLE),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    bottom = 0.0
    for label, value, color in components:
        ax.bar(
            [0],
            [value],
            bottom=bottom,
            color=color,
            width=0.42,
            edgecolor="white",
            linewidth=1.2,
            label=f"{label} — {value:.3g} ms",
            zorder=3,
        )
        if value >= 0.15:
            ax.text(
                0,
                bottom + value / 2,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontweight="bold",
            )
        bottom += value

    ax.axhline(1.02, color="black", ls="--", lw=1.2, zorder=4)
    ax.text(
        0.24,
        1.04,
        "weight-read floor 1.02 ms",
        va="bottom",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=1),
    )
    ax.text(0, bottom + 0.06, f"total {bottom:.2f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks([0])
    ax.set_xticklabels(["SmolLM2-135M\n128 tokens, CUDA graph"], fontsize=9)
    ax.set_ylabel("ms / token")
    ax.set_ylim(0, 3.85)
    ax.set_xlim(-0.55, 0.55)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        labelspacing=0.55,
    )
    ax.set_title("Where 3.36 ms goes", fontsize=11)
    save(fig, "fig3_breakdown")


def main():
    """Create all three figures and print their paths."""
    os.makedirs(RESULTS, exist_ok=True)
    fig1()
    fig2()
    fig3()
    for stem in ("fig1_roofline_by_engine", "fig2_context_scaling", "fig3_breakdown"):
        for ext in ("png", "pdf"):
            print(os.path.join(RESULTS, f"{stem}.{ext}"))


if __name__ == "__main__":
    main()
